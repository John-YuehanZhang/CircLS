"""Sequential PPM driver — run a sequence of joint Pauli-product measurements on
rotated routed surface-code patches; between PPMs measure out ONLY the bus
(corridor), target patches persist with state.

Orchestrates the existing single-PPM IR primitives (RoutedMultiPatchLSExperiment
is the per-step template; CNOTLSExperiment is the two-PPM precedent); it does NOT
modify either.
"""
from dataclasses import dataclass, replace
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import stim

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.utils.linear_algebra import solve_linear_decomposition
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.bent_joint_se import se_round_chunk
from .multi_patch_coupler import (
    route_and_build, conjugate_patch_records, conjugate_layout, cell_index,
    origin_of, PatchSpec, RotatedRoutedMultiPatchCoupler, BentLayoutError,
    probe_wall_precheck)
from .common import _CONJ, _FLIP_O, _ROTATION_SEARCH_CAP  # noqa: F401
from circls.compiler.liveness import LivenessMixin, compute_lifetimes  # noqa: F401
from circls.compiler.parallel_windows import ParallelWindowsMixin
from circls.compiler.rotation_plan import RotationPlannerMixin
from .joint_merge import (
    RotatedSeamWallCoupler, SeamRuleError, classify_seam, patch_view,
    wall_spec, TEXTBOOK, CONJUGATE)
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.swap_rotation import rotate_patches_swap
from lightstim.qec_code.surface_code.rotated.litinski_rotation import rotate_patches_litinski
from lightstim.qec_code.surface_code.rotated.SE_block import (
    RotatedSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.y_boundary_patch import (
    make_degenerate_y_boundary_patch)
from lightstim.qec_code.surface_code.rotated.y_transition_round import (
    make_y_transition_chunk)

__all__ = ["PPMStep", "compute_lifetimes", "SequentialPPMExperiment"]


@dataclass
class PPMStep:
    """One joint PPM in a sequence.

    interaction_type: [(patch_name, Pauli), ...] with Pauli in {'X', 'Z'}.
    route:            optional explicit corridor cell list (else auto-routed;
                      cell-adjacent targets default to the zero-cell seam).
    construction:     'auto' (rule table decides) | 'merge' | 'wall' —
                      only meaningful for cell-adjacent targets.
    schedule:         per-step override of the merged-round schedule
                      ('bent' | 'diagonal'); None = the experiment's policy.
    """
    interaction_type: List[Tuple[str, str]]
    route: Optional[list] = None
    construction: str = 'auto'
    schedule: Optional[str] = None
    #: adjacencies that stay UNSTITCHED — ``('q1', (1, 0))`` (patch | corridor
    #: cell) or ``('q4', 'q6')`` (patch | patch).  The two boundaries sit next
    #: to each other without merging: no seam qubits, no seam-orientation rule
    #: (a corridor cell may need to pass BY a target to reach another patch).
    unstitch: Optional[list] = None


def _zip_round_chunks(chunks, pad=False):
    """Interleave qubit-disjoint single-round chunks tick-by-tick into ONE
    round.  With ``pad=False`` all chunks must have the same layer count
    (simultaneous Y-birth transition rounds, same distance).  With
    ``pad=True`` shorter chunks are padded with empty layers (idle ticks,
    honestly noise-counted) — used to composite a gidney-scheduled birth
    round with the bent/diagonal round of the rest of the system."""
    if len(chunks) == 1:
        return chunks[0]

    def layers(c):
        out, cur = [], stim.Circuit()
        for inst in c:
            if inst.name == 'TICK':
                out.append(cur)
                cur = stim.Circuit()
            else:
                cur.append(inst)
        out.append(cur)
        return out

    split = [layers(c) for c in chunks]
    counts = {len(l) for l in split}
    if len(counts) != 1:
        if not pad:
            raise ValueError(
                f"cannot tick-zip transition chunks with unequal layer counts "
                f"{sorted(len(l) for l in split)}")
        k = max(counts)
        for l in split:
            # pad BEFORE the last layer: every chunk's measurement layer must
            # land in the SAME final layer (the SE path's back-propagation
            # treats the last instruction as THE round's measurements)
            while len(l) < k:
                l.insert(len(l) - 1, stim.Circuit())
    merged = stim.Circuit()
    k = max(counts)
    for j in range(k):
        # coalesce same-(name,args,tag) instructions across the sub-chunks
        # into ONE instruction per group: the SE path's back-propagation
        # contract is "the chunk's measurements = the LAST instruction", so
        # a layer ending in two separate M instructions silently drops half
        # the round's measurements from the tracker (record indices shift,
        # every later detector mis-compares — found the hard way)
        groups, order = {}, []
        for l in split:
            for inst in l[j]:
                args = tuple(inst.gate_args_copy())
                if args == (0.0,) and inst.name in (
                        'M', 'MX', 'MY', 'MR', 'MRX', 'MRY'):
                    args = ()   # M(0) == M: canonicalise so they coalesce
                key = (inst.name, args, inst.tag)
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].extend(inst.targets_copy())
        for name, args, tag in order:
            merged.append(name, groups[(name, args, tag)], list(args), tag=tag)
        if j < k - 1:
            merged.append('TICK')
    return merged


class SequentialPPMExperiment(RotationPlannerMixin,
                              ParallelWindowsMixin, LivenessMixin):
    def __init__(self, patches, ppm_sequence, *, initial_states,
                 final_measure_states, rounds=3, rounds_init=1, idle_rounds=0,
                 noise_params=None, noise_model='circuit_level',
                 liveness: bool = False, first_use_init: bool = True,
                 keep_patches=None, auto_rotate: bool = False,
                 rotate_saving_threshold: int = 1,
                 rotation_kind: str = 'litinski', schedule: str = 'auto',
                 parallel_steps: bool = False,
                 colour_swapped=frozenset(), lifetime_overrides=None,
                 router=None):
        self.patches = list(patches)
        self.ppm_sequence = list(ppm_sequence)
        # parallel_steps: execute contiguous runs of patch-disjoint steps in
        # ONE shared merge window (batch) — corridors coexist, one SE block
        # drives every active coupler (the emission is system-global anyway)
        self.parallel_steps = parallel_steps
        self.initial_states = {k: v.upper() for k, v in initial_states.items()}
        self.final_measure_states = {k: v.upper()
                                     for k, v in final_measure_states.items()}
        # initial 'Y' = the fault-tolerant in-place Gidney birth (arXiv:
        # 2302.07395): degenerate XXZZ patch + mixed-basis reset + reversed
        # rounds + transition, run in a prologue before anything else exists
        # (the corner-injection Y init was removed 2026-08-02 — a uniform
        # transversal 'Y' reset is a non-codespace product state, and stays
        # rejected).  Final 'Y' (death transition) is not wired up: the
        # pipeline reads gadget ancillas out in X.
        bad = {k: v for k, v in self.initial_states.items()
               if v not in ('X', 'Z', 'Y')}
        if bad:
            raise ValueError(f"initial_states letters must be X/Z/Y; got {bad}")
        bad = {k: v for k, v in self.final_measure_states.items()
               if v not in ('X', 'Z')}
        if bad:
            raise ValueError(
                f"final_measure_states letters must be 'X' or 'Z'; got {bad} "
                f"(Y-basis readout needs the measure-direction transition — "
                f"not wired into the experiment; gadget ancillas end in 'X')")
        self._y_names = frozenset(k for k, v in self.initial_states.items()
                                  if v == 'Y')
        assert rounds_init >= 1, (
            f"rounds_init must be >= 1 (a freshly initialised patch needs at least one "
            f"standalone SE round to establish its stabilizers before the merge); "
            f"got {rounds_init}")
        self.rounds = rounds
        self.rounds_init = rounds_init
        self.idle_rounds = idle_rounds
        self.noise_params = noise_params
        self.noise_model = noise_model
        self.liveness = liveness
        # first_use_init: allocate + initialise each patch at its FIRST PPM use instead of all
        # up front — the TIMING mirror of the liveness path retiring a patch at its LAST
        # use. It saves the patch's qubit allocation / reset / baseline-SE cost until the
        # patch is needed. NOTE it does NOT free the patch's layout cell: a not-yet-used
        # patch's coarse cell is still an obstacle to EARLIER PPMs' routing. Only the
        # liveness/retire path frees a cell (for a later corridor to reuse); first_use_init is
        # a time saver, not a space saver.
        self.first_use_init = first_use_init
        # keep_patches: the OUTPUT patches (final data readout) that must NEVER be retired
        # mid-sequence — the liveness path retires only patches NOT listed here (the consumed
        # / magic-state / ancilla patches). Retiring an OUTPUT patch loses its observable
        # (its output is absorbed by a later PPM and then emitted as a detector, not an
        # observable — a known limitation), so the caller MUST declare its outputs whenever
        # retirement is on. Required when liveness=True; ignored (nothing is retired) when
        # liveness=False.
        if liveness and keep_patches is None:
            raise ValueError(
                "keep_patches is required when liveness=True: list the OUTPUT patches "
                "(those with a final data readout) so the liveness path retires only the "
                "consumed / magic-state patches. Retiring an output patch would silently "
                "drop its observable. Pass keep_patches=set() only if NO patch is an output.")
        self.keep_patches = set(keep_patches) if keep_patches is not None else set()
        for step in self.ppm_sequence:
            for name, P in step.interaction_type:
                if P not in ('X', 'Z'):
                    raise ValueError(
                        f"PPM Pauli must be 'X' or 'Z', got {P!r} on patch {name}")
        patch_names = {s.name for s in self.patches}
        missing_init = patch_names - self.initial_states.keys()
        if missing_init:
            raise ValueError(
                f"patch(es) {sorted(missing_init)} missing from initial_states")
        missing_final = patch_names - self.final_measure_states.keys()
        if missing_final:
            raise ValueError(
                f"patch(es) {sorted(missing_final)} missing from "
                f"final_measure_states")
        unknown_keep = self.keep_patches - patch_names
        if unknown_keep:
            raise ValueError(
                f"keep_patches names {sorted(unknown_keep)} are not patches in this "
                f"experiment (patches: {sorted(patch_names)})")
        # auto_rotate: between two PPMs, physically rotate target patches with the
        # REAL protocols (rotate_90 = SWAP network: flips orientation and swaps the
        # red/blue COLOURS, weight-2 positions unchanged; litinski = 5-step
        # deformation: flips orientation and MOVES the weight-2 positions
        # (textbook<->conjugate), colours unchanged) when the targets' required
        # buses disagree, when the step does not route in the current orientations
        # (blocked faces), or when a rotation saves >= rotate_saving_threshold bus
        # cells. Off (default) keeps the historical behaviour verbatim.
        self.auto_rotate = auto_rotate
        # rotate_saving_threshold: with auto_rotate, ALSO rotate a routable step's
        # target when the rotated route saves at least this many corridor cells
        # (bus length).  The cost trade (rotation rounds/ticks vs bus cells) is
        # deliberately left as this single knob; default 1 = rotate whenever even
        # one bus cell is saved.
        self.rotate_saving_threshold = rotate_saving_threshold
        # rotation_kind: restrict the planner to ONE rotation protocol (so an
        # experiment can use a single rotate method globally, e.g. for the paper's
        # cost comparison).  'auto' (default) picks by constraint: a red/blue
        # colour-convention switch needs rotate_90 (the only move that swaps the
        # colours), a pure orientation flip prefers litinski (singles, pairs) with
        # the all-targets rotate_90 group (everyone's colours swap together, the
        # bus flips) as the fallback.  'litinski': steps needing a colour switch
        # raise (litinski cannot swap the colours).  'rotate_90': orientation
        # fixes only via the all-targets group; raises when it does not route.
        if rotation_kind not in ('auto', 'rotate_90', 'litinski'):
            raise ValueError(
                f"rotation_kind must be 'auto', 'rotate_90' or 'litinski', "
                f"got {rotation_kind!r}")
        self.rotation_kind = rotation_kind
        self.rotation_count = 0
        self.rotations: List[Tuple[int, str]] = []   # (ppm_index, patch_name) rotated before
        self.rotation_log: List[Tuple[int, str, str]] = []   # + kind ('rotate_90'|'litinski')
        # _derived_bus: {ppm_index: 'X'|'Z'} — with auto_rotate, each step's bus is
        # DERIVED from the targets' live red/blue colour conventions: a
        # standard-coloured patch requires bus == its Pauli, a colour-swapped one
        # requires the opposite bus; all targets must require the SAME bus
        # (disagreement = some patch needs its colours swapped, which only
        # rotate_90 does).  Without auto_rotate the majority vote stands.
        self._derived_bus = {}
        # schedule policy for MERGED rounds (the user's rule: decide right
        # after routing — any bend in the corridor, or a wall in the step,
        # forces the WHOLE merged block onto the diagonal schedule; pure
        # straight corridors and the row-4 recoloured column stay bent):
        # 'auto' (default) | 'bent' (historical everywhere; wall steps raise)
        # | 'diagonal' (everywhere; row-4 columns raise in the validator).
        if schedule not in ('auto', 'bent', 'diagonal'):
            raise ValueError(
                f"schedule must be 'auto', 'bent' or 'diagonal', got {schedule!r}")
        self.schedule = schedule
        # colour_swapped: patches whose red/blue COLOURS are swapped at
        # allocation (conjugate_patch_records on top of the standard build).
        # This is a COLOUR knob — weight-2 positions (textbook/conjugate)
        # are untouched; the live logical orientation flips with the labels.
        self.colour_swapped = frozenset(colour_swapped or ())
        unknown_cs = self.colour_swapped - patch_names
        if unknown_cs:
            raise ValueError(
                f"colour_swapped names {sorted(unknown_cs)} are not patches")
        # router= (docs/API_HOOKS.md): strategy injection for the one
        # state-dependent decision.  Called once per step with the LIVE
        # specs of the first routing attempt; returning None falls back
        # to the internal corridor search; returned cells go through
        # route_and_build's explicit-route path and its full oracle.
        # NOT consulted where no search would run: an explicit
        # PPMStep.route, the zero-cell direct seam of a cell-adjacent
        # pair, and a parallel window's planned corridor all precede it
        # (precedence table: docs/API_HOOKS.md).  The answer cache is
        # keyed by the step's SEQUENCE POSITION, never id(step) — a
        # recycled object address serves one step another step's
        # corridor (the C2 failure mode, see _route_result's memo key).
        self.router = router
        self._router_cache = {}
        self.lifetimes = compute_lifetimes(self.ppm_sequence)
        if lifetime_overrides:
            # result-injection hook (docs/API_HOOKS.md): a caller-chosen
            # {name: (init_layer, free_layer)} replacing the derived
            # first/last-use pair.  An override may only WIDEN a
            # lifetime: the patch must exist at every layer that uses it.
            for nm, pair in lifetime_overrides.items():
                if nm not in patch_names:
                    raise ValueError(
                        f"lifetime override for unknown patch {nm!r}")
                a, b = int(pair[0]), int(pair[1])
                if nm in self.lifetimes:
                    fu, lu = self.lifetimes[nm]
                    if a > fu or b < lu:
                        raise ValueError(
                            f"lifetime override for {nm} ({a},{b}) must "
                            f"contain its uses ({fu},{lu})")
                if not (0 <= a <= b < max(len(self.ppm_sequence), 1)):
                    raise ValueError(
                        f"lifetime override for {nm} out of range: "
                        f"({a},{b})")
                self.lifetimes[nm] = (a, b)
        self._orient = {s.name: (_FLIP_O[s.orientation]
                                 if s.name in self.colour_swapped
                                 else s.orientation)
                        for s in self.patches}

    @staticmethod
    def _bus_of(interaction_type):
        n_x = sum(1 for _, P in interaction_type if P == 'X')
        return 'X' if n_x >= len(interaction_type) - n_x else 'Z'

    def _bus_for(self, i, step):
        """The native basis of PPM ``i``: with auto_rotate the bus DERIVED from the
        targets' live conventions (set by _plan_rotations), else the historical
        majority target basis (tie -> X)."""
        if i in self._derived_bus:
            return self._derived_bus[i]
        return self._bus_of(step.interaction_type)

    def _resolve_registration(self):
        """name -> (is_conjugate, geometry_orientation) for the INITIAL
        allocation — the DECLARED input state, nothing inferred."""
        # STRICT BIRTHS (design decision 2026-07-31): every patch is allocated
        # EXACTLY as the input declares — orientation from its spec, colours
        # from ``colour_swapped`` (applied separately at allocation).  The
        # old first-use minority-role inference (a planner-invented birth
        # conjugation) is retired; letter mismatches are step-2 constructs
        # (walls / recolour columns) or costed physical rotations.
        return {s.name: (False, s.orientation) for s in self.patches}

    def _specs_with(self, orient_map):
        """The patch specs re-stamped with the effective orientation in
        ``orient_map`` (name -> orientation) — PatchSpec is frozen, so a rotated
        patch is represented to the router by a spec with its orientation flipped."""
        return [replace(s, orientation=orient_map[s.name]) for s in self.patches]

    def _specs_for_step(self, i, orient_map, force_present=frozenset()):
        """The specs the router sees for PPM ``i``.  Free ground (NOT an
        obstacle) includes, besides empty cells:

        ``force_present`` overrides the absence rules: a shared-window batch
        passes every member's patch names, because a batch-mate's patch IS
        alive during this step's window even when its first use is a later
        step index (the serial borrow-the-cell contract does not hold under
        co-activity).

        * a patch first used LATER (``first_use_init``): not allocated yet —
          the corridor borrows its cell and is measured out at the split,
          before the patch is born there;
        * a patch already RETIRED (``liveness``: last use < i, not an output
          in ``keep_patches``): its cell has been freed — the corridor
          reuses it (the v2m2 reuse path; registration goes lazy)."""
        def _absent(nm):
            if nm in force_present:
                return False
            lt = self.lifetimes.get(nm)
            if lt is None:
                return False
            fu, lu = lt
            if self.liveness and lu < i and nm not in self.keep_patches:
                return True
            if nm in self._y_names:
                # deferred Gidney birth starts (mixed reset) during step
                # fu-1; the cell must not serve as a bus from the round
                # BEFORE that reset (design decision 2026-08-03), i.e. it is an
                # obstacle for steps >= fu-2 regardless of first_use_init
                return i < fu - 2
            if self.first_use_init and fu > i \
                    and nm not in self.system.patches:
                # the ledger is the PHYSICAL allocation, not the lifetime
                # alone: a demoted shared-window batch leaves its preamble
                # allocations standing (_try_run_batch: "allocations
                # stick"), so a patch first used at a later step can
                # already be alive here -- routing through its cell as
                # borrowed ground then collides at add_patch (measured
                # 2026-09-07: multiply_n13 d3 liveness off, row_major:
                # batch [4..9] allocated q4 (fu=9) at step 4, demoted,
                # and ppm_4's serial corridor ran through q4's cell)
                return True
            return False
        return [replace(s, orientation=orient_map[s.name])
                for s in self.patches if not _absent(s.name)]

    def _route_result(self, specs, step, bus=None, raise_errors=False,
                      probe=False, conj=None, blocked_cells=frozenset(),
                      step_index=None):
        """The SubsetRoute of PPM ``step`` for the given specs, or None if it does
        not route (a perpendicular-seam / no-legal-side placement surfaces as a
        BentLayoutError or a non-'ok' status; either counts as infeasible).

        ``conj``: the conjugation-registration TRUTH for the step's targets —
        a frozenset of conj-registered names.  The planner passes its live
        hypothesis per candidate; ``None`` reads ``self._conj_live`` (the
        applied state).  It reaches route_and_build as ``conj_names``, so the
        seam table is classified against the REAL registrations instead of
        the historical minority-letter inference.

        Colour-swapped corridor: a SAME-Pauli step whose required bus is the
        OPPOSITE of the measured Pauli means every target is colour-swapped
        (after rotate_90s) — the corridor it needs is the standard construction
        under a global colour swap.  Route + verify the colour-conjugate
        PRE-IMAGE (orientations flipped, Paulis conjugated, plain majority bus)
        and emit ``conjugate_layout`` of the result; colour swapping is an exact
        relabeling symmetry, so no re-verification is needed."""
        if self.router is not None and step.route is None:
            # cache by ppm_sequence position, passed in by every caller:
            # id(step) is unique only while the object lives, and this
            # file already paid for that once (C2 — the route memo's
            # dangling-id key, fixed 2026-08-03 below)
            if step_index is None:
                cells = self.router(specs, step)
            else:
                if step_index not in self._router_cache:
                    self._router_cache[step_index] = self.router(specs, step)
                cells = self._router_cache[step_index]
            if cells is not None:
                step = replace(step, route=[tuple(c) for c in cells])
        if conj is None:
            conj = frozenset(nm for nm, _ in step.interaction_type
                             if getattr(self, '_conj_live', {}).get(nm))
        paulis = {P for _, P in step.interaction_type}
        # the global-conjugate shortcut is ONLY the all-swapped case: every
        # target conj-registered, one measured letter, bus = its conjugate.
        # A PARTIAL conj set under the conjugate bus is a normal mixed-row
        # construction (walls / recolour columns) — feeding it the global
        # trick builds a semantically wrong circuit (measured: tracker
        # census breaks downstream).
        swapped = (bus is not None and len(paulis) == 1
                   and bus == _CONJ[next(iter(paulis))]
                   and conj == frozenset(nm for nm, _
                                         in step.interaction_type))
        # route_and_build is a pure function of (specs, step, bus): memoize
        # successful REAL results so the planner's confirmation build is
        # reused verbatim at registration instead of being rebuilt
        key = None
        if not probe:
            # cache key = STEP CONTENT, never object identity: the route is a
            # pure function of (specs, step content), while id(step) of the
            # throwaway replace() copies dangles after GC — a recycled
            # address handed one step another step's route (C2, fixed
            # 2026-08-03; the old key's only discriminating component was
            # the dangling id).
            key = ((tuple(step.interaction_type), step.construction,
                    step.schedule,
                    tuple(map(tuple, step.unstitch)) if step.unstitch else None),
                   bus, conj, tuple(sorted(map(tuple, blocked_cells))),
                   tuple((sp.name, tuple(sp.origin), sp.distance,
                          sp.orientation) for sp in specs),
                   tuple(map(tuple, step.route)) if step.route else None)
            hit = getattr(self, '_rr_cache', {}).get(key)
            if hit is not None:
                return hit
        # intra-step geometry memo (graph / tree pools shared across the
        # planner's per-candidate probes; pure geometry, hit == recompute)
        if not hasattr(self, '_geom_cache'):
            self._geom_cache = {}
        try:
            if swapped:
                pre_specs = [replace(s, orientation=_FLIP_O[s.orientation])
                             for s in specs]
                pre_target = [(nm, _CONJ[P]) for nm, P in step.interaction_type]
                r = route_and_build(pre_specs, pre_target, seam=True,
                                    route=step.route,
                                    blocked_cells=blocked_cells,
                                    no_stitch=step.unstitch, probe=probe,
                                    geom_cache=self._geom_cache)
                if r.status == 'ok' and not probe:
                    r = replace(r, layout=conjugate_layout(r.layout))
            else:
                if probe and not probe_wall_precheck(
                        step.interaction_type,
                        {sp.name: sp.orientation for sp in specs},
                        conj, bus):
                    # tree-independent dispatch verdict: no corridor can
                    # host this (rotation, bus, conj) candidate — skip the
                    # full candidate-tree scan (planner hot loop)
                    return None
                r = route_and_build(specs, step.interaction_type, seam=True,
                                    route=step.route, bus=bus,
                                    conj_names=conj,
                                    blocked_cells=blocked_cells,
                                    no_stitch=step.unstitch, probe=probe,
                                    geom_cache=self._geom_cache)
        except BentLayoutError:
            if raise_errors:
                raise
            return None
        if key is not None and r.status == 'ok':
            if not hasattr(self, '_rr_cache'):
                self._rr_cache = {}
            self._rr_cache[key] = r
        if raise_errors:
            return r
        return r if r.status == 'ok' else None

    def _apply_rotation_moves(self, i, moves):
        """Run ``moves = [(name, kind), ...]`` on the LIVE system and update
        every book: the two protocols apply CONCURRENTLY (SWAP networks
        merged tick-for-tick, litinski protocols advancing stage-by-stage in
        lockstep sharing the system-wide SE rounds), so each kind costs one
        batch.  rotate_90 flips the conjugation bit and keeps the type;
        litinski flips the type and keeps the colours; both flip the
        orientation."""
        r90 = [nm for nm, kind in moves if kind == 'rotate_90']
        lit = [nm for nm, kind in moves if kind == 'litinski']
        if r90:
            rotate_patches_swap(self.system, self.builder, r90)
        if lit:
            rotate_patches_litinski(
                self.system, self.builder, lit,
                directions={nm: self._lit_direction(
                    self._eff_orient[nm], self._conj_live[nm])
                            for nm in lit})
        for nm, kind in moves:
            if kind == 'rotate_90':
                self._conj_live[nm] = not self._conj_live[nm]
            else:
                self._type_live[nm] = (CONJUGATE
                                       if self._type_live[nm] == TEXTBOOK
                                       else TEXTBOOK)
            self._eff_orient[nm] = _FLIP_O[self._eff_orient[nm]]
            self._orient[nm] = self._eff_orient[nm]
            self.rotation_count += 1
            self.rotations.append((i, nm))
            self.rotation_log.append((i, nm, kind))

    def _plan_snake(self, i, step, eff, conj):
        """Plan the SNAKE construction for a colour-parity-violating routed
        step (any number of same-letter targets): EVERY colour-swapped
        target attaches the corridor through a stretched (kf) wall ON ITS
        OWN attach seam (the K&F Fig 4 a/b arrangement — each stretched
        weight-4's side pair carries the OPPOSITE letter of its
        neighbouring cell stabilizer; the stretched weight-2 end lobe sits
        at the end away from the patch's own lobes), while every
        standard-colour target attaches per the corridor painting.  One
        corridor, one coupler, one merge window, zero rotations.

        Painting: a wall on a VERTICAL seam (patch and wall cell side by
        side) hosts horizontal dominoes, whose pure family has no compact-7
        schedule (variant search exhausted) — the corridor joins the flipped
        gauge (``flip_cells``) so every wall lands on the MIXED seam-table
        row (#4 -> #6, the verified K&F Fig 4a family) and every
        standard-colour seam becomes the recoloured column (#1 -> #7).
        With only horizontal-seam walls the pure painting stays (the
        vertical-domino pure family is verified).  Returns the plan dict or
        None (fall back to rotations).

        A wall fixes a COLOUR mismatch, never an ORIENTATION one: every
        attach seam — walled or painted — still owes the parallel law, so a
        target whose live orientation has no legal attach cell on this
        corridor is NOT a snake case and falls back to the rotation search.
        The step-1 route probe does not see that (``_auto_stitch`` counts a
        walled seam as an attachment without re-checking the law), which is
        how an infeasible snake used to reach ``_register_snake`` and die
        there as a hard error (grover_n2's second CX gadget, 2026-08-16)."""
        tgt = step.interaction_type
        if len(tgt) < 2 or not step.route:
            return None
        paulis = {P for _, P in tgt}
        if len(paulis) != 1:
            return None
        P = next(iter(paulis))
        conj_t = [nm for nm, _ in tgt if conj[nm]]
        if not conj_t:
            return None
        cells = [tuple(c) for c in step.route]

        def _attach(nm, pauli, p_cell):
            """(corridor cells edge-adjacent to this target, the subset whose
            seam obeys the parallel law for the measured letter)."""
            adj = [c for c in cells
                   if abs(c[0] - p_cell[0]) + abs(c[1] - p_cell[1]) == 1]
            return adj, [c for c in adj
                         if self._parallel_orientation(
                             pauli, self._seam_axis_of((p_cell, c)))
                         == eff[nm]]

        p_cells = {nm: cell_index(self._by_name[nm].origin,
                                  self._by_name[nm].distance, seam=True)
                   for nm, _ in tgt}
        # a target that touches the corridor must touch it LEGALLY (a target
        # that does not touch it at all is the route probe's business)
        for nm, Pn in tgt:
            adj, ok = _attach(nm, Pn, p_cells[nm])
            if adj and not ok:
                return None
        walls = []                     # (name, wall cell, patch cell)
        for nm in conj_t:
            p_cell = p_cells[nm]
            _adj, ok = _attach(nm, P, p_cell)
            if not ok:                 # the wall has no legal seam to sit on
                return None
            walls.append((nm, ok[0], p_cell))
        bus = P                        # shared-letter painting (pure set)
        flip = ([tuple(c) for c in step.route]
                if any(wc[0] != pc[0] for _, wc, pc in walls) else None)
        specs = self._specs_for_step(i, eff)
        # STEP-1 admission ONLY (iron rule 2026-07-31: path finding never
        # constructs a stabilizer).  The probe runs the explicit-route
        # dispatch + stitch + connectivity checks with the walls counted
        # as attachments; the REAL build happens once, at registration
        # (_register_snake), and its failure is a hard error there.
        try:
            r = route_and_build(specs, list(tgt), seam=True, route=step.route,
                                bus=bus, conj_names=frozenset(conj_t),
                                flip_cells=flip, no_stitch=step.unstitch,
                                arm_walls=[(nm, wc) for nm, wc, _ in walls],
                                probe=True)
        except (BentLayoutError, ValueError):
            return None
        if r.status != 'ok':
            return None
        return dict(conj=frozenset(conj_t), bus=bus,
                    walls=[(nm, wc) for nm, wc, _ in walls], flip=flip)

    def _setup(self):
        self.tracker = SyndromeTracker(num_qubits=self.system.num_qubits,
                                       expected_num_logicals=self.system.num_logicals)
        self.builder = CircuitBuilder(tracker=self.tracker,
                                      system_config=self.system,
                                      if_detector=True)
        self.system.register_tracker(self.tracker)
        self.system.register_builder(self.builder)

    def _live_y_uids(self):
        """Interior checks of the live |Y>-born patches.  Rounds that cover
        a freshly-sealed Y patch must run these on Gidney's order
        (SE_block contract: a bent round surrounding the Y transition
        costs one unit of fault distance -- measured on toffoli/bell/
        fredkin/simon at d = 5, all four at graphlike distance 4 until
        the surrounding rounds were re-scheduled)."""
        owner = self.system.index_to_owner_map
        yq = {q for q, o in owner.items() if o in self._y_names}
        return {u for u in self.system.active_stabilizer_indices
                if self.system.stabilizers[u].get('pauli')
                and all(dq in yq for dq in self.system.stabilizers[u]['pauli'])}

    def _y_owned_coords(self):
        """Coordinates owned by the |Y>-born patches -- the scope of the
        diagonal block's kf near-side mirror (``y_coords``): the mirror may
        only fire where a vertical relay wall's near side rests both feet
        on a Y wedge.  The scoping is mandatory -- an unscoped mirror
        collides where the wall abuts plain-K&F checks (measured)."""
        return frozenset(
            tuple(self.system.qubit_coords[q])
            for q, o in self.system.index_to_owner_map.items()
            if o in self._y_names)

    def _standalone_se(self, n_rounds):
        if n_rounds < 1:
            return
        yuids = self._live_y_uids()
        if yuids:
            # first_use_init baseline round over a live Y patch: same
            # spectator+gidney zip the seal round itself uses
            chunk = _zip_round_chunks(
                [self._spectator_chunk(yuids),
                 self._gidney_chunk(yuids, 'gidney')], pad=True)
            self.builder.apply_syndrome_extraction(circuit_chunk=chunk,
                                                   rounds=n_rounds)
            return
        owner = self.system.index_to_owner_map
        domains = {tuple(self.system.qubit_coords[q]): self._orient[owner[q]]
                   for q in self.system.data_indices
                   if owner.get(q) in self._orient}
        chunk = se_round_chunk(self.system, domains=domains)
        self.builder.apply_syndrome_extraction(circuit_chunk=chunk, rounds=n_rounds)


    def _pauli_vec(self, targets):
        """[X|Z] GF(2) vector of the joint logical named by [(patch, letter)]."""
        n = self.tracker.num_qubits
        v = np.zeros(2 * n, dtype=np.uint8)
        for nm, letter in targets:
            for rec in (x for x in self.system.logical_ops
                        if x.get('patch_name') == nm and x.get('type') == letter):
                for q, pp in rec['pauli'].items():
                    if pp in ('X', 'Y'):
                        v[q] ^= 1
                    if pp in ('Z', 'Y'):
                        v[n + q] ^= 1
        return v

    def _parity_records(self, vec):
        """Read-only: if vec is in the CURRENT stabilizer span, return the
        sorted XOR of the pinning rows' banked records (its deterministic
        reconstruction); else None (a free outcome, not a parity check)."""
        # stabilizer rows PLUS seeded logical rows (both carry records —
        # "deterministic since initialisation" parities live in the latter);
        # same stacking precedent as tracker.py logical_canonicalization
        from lightstim.ir.tracker import UNMEASURED_STAB_RECORD
        stab, logs = self.tracker.stabilizers, self.tracker.logicals
        # deferred allocation can grow the qubit count between the two
        # trackers' refreshes (and vec is built at the CURRENT count).
        # Symplectic rows are [X | Z] halves, so the narrower matrix is
        # widened by placing each half at its new offset — appended qubits
        # keep identity support, existing indices are stable.
        wide = max(stab.matrix.shape[1], logs.matrix.shape[1]
                   if logs.matrix.shape[0] > 0 else 0, vec.shape[0]) // 2

        def _widen(M):
            n0 = M.shape[1] // 2
            if n0 == wide:
                return M
            P = np.zeros((M.shape[0], 2 * wide), dtype=M.dtype)
            P[:, :n0] = M[:, :n0]
            P[:, wide:wide + n0] = M[:, n0:]
            return P

        if vec.shape[0] != 2 * wide:
            v = np.zeros(2 * wide, dtype=vec.dtype)
            n0 = vec.shape[0] // 2
            v[:n0] = vec[:n0]
            v[wide:wide + n0] = vec[n0:]
            vec = v
        if logs.matrix.shape[0] > 0:
            M = np.vstack([_widen(stab.matrix), _widen(logs.matrix)])
            recs_of = stab.records + logs.records
        else:
            M = _widen(stab.matrix)
            recs_of = stab.records
        # UNMEASURED rows carry the sentinel, not a value — a closure that
        # uses one XORs in an unbanked (per-shot random) gauge.  Serial
        # flows never landed on them; a parallel batch window offers the
        # solver batch-mates' fresh joint rows and reduce_weight happily
        # took one (measured 2026-08-05: ghz-mixed batch 2, the b3^b5
        # deterministic parity read random while the circuit's own
        # observables stayed exact).  Excluding them can only remove wrong
        # reconstructions: a vec only reachable through unmeasured rows is
        # genuinely not record-deterministic yet.
        n_stab = stab.matrix.shape[0]
        rows_ok = [k for k in range(M.shape[0])
                   if k >= n_stab
                   or (recs_of[k]
                       and UNMEASURED_STAB_RECORD not in recs_of[k])]
        if len(rows_ok) != M.shape[0]:
            M = M[rows_ok]
            recs_of = [recs_of[k] for k in rows_ok]
        if M.shape[0] == 0:
            return None
        out = solve_linear_decomposition(basis=M, targets=vec.reshape(1, -1),
                                         reduce_weight=True)
        coeffs = out[0]
        if coeffs is None:
            return None
        row = np.asarray(coeffs[0], dtype=np.uint8)
        if not np.array_equal((row.astype(int) @ M.astype(int)) % 2,
                              vec.astype(int)):
            return None
        recs = set()
        for j in np.nonzero(row)[0]:
            recs ^= set(recs_of[j])
        return sorted(recs)


    def _apply_ppm_step(self, i, step, birth_names=None):
        self.step_joint_records_pre[i] = self._parity_records(
            self._pauli_vec(step.interaction_type))
        subset = step.interaction_type
        bus = self._bus_for(i, step)
        cname = f'ppm_{i}'
        # idle standalone SE between PPMs (debug knob; default 0)
        if self.idle_rounds:
            self._standalone_se(self.idle_rounds)
        # merge (coupler already registered up front)
        self.builder.activate_coupler(cname)
        coupler_patch = self.system.coupler_patches[cname]
        l2g = self.system.local_to_global_map[cname]
        coupler_init = {l2g[q]: _CONJ[bus] for q in coupler_patch.data_indices}
        if coupler_init:
            self.builder.initialize(init_dict=coupler_init, n=self.system.num_qubits)
        # deferred-birth ride: the NEXT step's Y patches are allocated+reset
        # AFTER the first merged round (the absorb round must stay clean —
        # seeding in the same round breaks the merge bookkeeping, measured),
        # then their reversed pre-rounds ride the remaining merged rounds.
        # mchunk is built BEFORE the birth, so it never contains the degens.
        mchunk = self._merged_chunk(i)
        if birth_names and self.rounds >= 2:
            self.builder.apply_syndrome_extraction(circuit_chunk=mchunk,
                                                   rounds=1)
            rider = self._birth_begin(birth_names)
            composite = _zip_round_chunks(
                [mchunk, rider['rev_chunk']], pad=True)
            self.builder.apply_syndrome_extraction(circuit_chunk=composite,
                                                   rounds=self.rounds - 1)
            rider['rides'] = self.rounds - 1
        else:
            if birth_names:
                self._birth_begin(birth_names)   # rounds==1: no room to ride
            self.builder.apply_syndrome_extraction(circuit_chunk=mchunk,
                                                   rounds=self.rounds)
        # split: measure out the corridor only (patches persist)
        self.builder.deactivate_coupler(cname)
        self.step_joint_records_post[i] = self._parity_records(
            self._pauli_vec(step.interaction_type))
        # split: the bus/corridor readout measures out the corridor ONLY; the joint
        # patch-patch relation it created persists, so it must not resolve absorbed_ops.
        self.builder.apply_data_readout(final_measurements=dict(coupler_init),
                                        resolve_absorbed=False)

    def _merged_chunk(self, i):
        """The one-round chunk of PPM ``i``'s merged system (all three
        schedule variants), WITHOUT emitting it.

        Every diagonal-schedule variant hands the live-|Y> interior checks
        to the block as ``gidney_uids``: a merged round covering a live Y
        patch must run those checks on Gidney's order (SE_block contract),
        and the block derives conflict-free custom orders where the merged
        layout's seam checks make pure Gidney infeasible (see
        ``DiagonalSurfaceCodeExtractionBlock._gidney_slot_maps``); the
        |Y>-owned coordinates ride along as ``y_coords``, scoping the
        block's kf near-side mirror."""
        yuids = self._live_y_uids()
        ycoords = self._y_owned_coords()
        if self._sched.get(i, 'bent') == 'diagonal':
            # any bend in the corridor (or a forced override) puts the whole
            # merged block on the diagonal schedule — one stretched/bent
            # check's tick budget is the global tick budget
            return DiagonalSurfaceCodeExtractionBlock(
                self.system, gidney_uids=yuids, y_coords=ycoords).circuit
        else:
            lay = self._routes[i].layout
            if any(c.get('kf') for c in lay.checks):
                # a dispatch-built stretched wall in the layout: the bent
                # chunk cannot express kf relay checks
                return DiagonalSurfaceCodeExtractionBlock(
                    self.system, gidney_uids=yuids, y_coords=ycoords).circuit
            else:
                if yuids:
                    # a bent merged round covering a freshly-born |Y>
                    # patch violates the SE_block Y-transition contract
                    # (the bent chunk cannot express Gidney's order for
                    # the Y checks; a zip variant conflicts with the
                    # builder's reset-basis back-propagation) -- demote
                    # the step to the diagonal schedule
                    return DiagonalSurfaceCodeExtractionBlock(
                        self.system, gidney_uids=yuids,
                        y_coords=ycoords).circuit
                domains_merged = dict(getattr(lay, 'domains', {}) or {})
                # SPECTATOR hook-safety: the routed layout's domains cover
                # only the step's patches + corridor.  Every OTHER live
                # patch must still run its orientation-correct schedule —
                # a defaulted hook direction aligned with the idle
                # logical halves its distance to (d+1)/2 (measured:
                # exp1 full circuit graphlike 2 at d=3 / 3 at d=5, the
                # weight-2 mechanism living entirely on spectator q3)
                _owner = self.system.index_to_owner_map
                for _q in self.system.data_indices:
                    _nm = _owner.get(_q)
                    if _nm in self._orient:
                        domains_merged.setdefault(
                            tuple(self.system.qubit_coords[_q]),
                            self._orient[_nm])
                return se_round_chunk(self.system, domains=domains_merged)

    def _apply_wall_step(self, i, step):
        """A stretched-stabilizer wall step: activate the wall coupler (its
        apparatus adds no data qubits, so there is no corridor init/readout),
        run the merged rounds on the DIAGONAL schedule (mandatory for kf
        checks), then split by deactivating — the paused facing lobes come
        back automatically."""
        self.step_joint_records_pre[i] = self._parity_records(
            self._pauli_vec(step.interaction_type))
        cname = f'ppm_{i}'
        if self.idle_rounds:
            self._standalone_se(self.idle_rounds)
        # live-|Y> interior checks ride the wall rounds on Gidney's order
        # (same SE_block contract as _merged_chunk); gathered before the
        # activation, which only pauses facing lobes / adds kf checks and
        # leaves the Y interiors untouched.  The |Y>-owned coordinates go
        # along too: a vertical wall whose near side rests on the Y wedge
        # mirrors that side's subslot order (the block's kf near-side
        # mirror; see DiagonalSurfaceCodeExtractionBlock).
        yuids = self._live_y_uids()
        ycoords = self._y_owned_coords()
        self.builder.activate_coupler(cname)
        self.builder.apply_syndrome_extraction(
            DiagonalSurfaceCodeExtractionBlock(
                self.system, gidney_uids=yuids, y_coords=ycoords).circuit,
            rounds=self.rounds)
        # extract BEFORE the split: deactivation restores the original patch
        # checks and the stretched-wall kf rows leave the live span (M3)
        self.step_joint_records_post[i] = self._parity_records(
            self._pauli_vec(step.interaction_type))
        self.builder.deactivate_coupler(cname)

    def _register_snake(self, i, step, plan):
        """Register a planned SNAKE step: build the coupler NOW from the
        plan's archived geometry (walls/bus/conj/flip) — step 2 is the ONLY
        place the stabilizers are constructed (iron rule), and a failed
        construction is a hard error, never a re-route.  The step then runs
        through the standard _apply_ppm_step (uniform corridor init,
        diagonal schedule — the wall's kf checks demand it)."""
        specs = self._specs_for_step(i, self._eff_orient)
        r = route_and_build(specs, step.interaction_type, seam=True,
                            route=step.route, bus=plan['bus'],
                            conj_names=plan['conj'],
                            flip_cells=plan['flip'],
                            no_stitch=step.unstitch,
                            arm_walls=list(plan['walls']))
        if r.status != 'ok':
            raise BentLayoutError(
                f"snake PPM {i}: the planned corridor does not pass the "
                f"rule-based construction — {r.message}")
        plan['route_result'] = r        # post-build record (tests read it)
        self.system.register_coupler(
            RotatedRoutedMultiPatchCoupler(),
            patch_names=[nm for nm, _ in step.interaction_type],
            name=f'ppm_{i}', specs=specs, target=step.interaction_type,
            subset_route=r, seam=True,
            minority_names=plan['conj'])
        self._routes[i] = r
        self._sched[i] = 'diagonal'

    def _adjacent_pair(self, step):
        """True iff the step's targets are exactly two patches on
        edge-adjacent coarse cells — the direct-seam territory of the rule
        table (rows 1-4); distant targets go through the routed corridor.
        An EXPLICIT route overrides the zero-cell-seam default (the route
        field's contract): a caller-injected corridor must be built even
        when the pair happens to touch (foreign placements: the corridor
        end cell may accidentally border the far patch)."""
        if step.route:
            return False
        names = [nm for nm, _ in step.interaction_type]
        if len(names) != 2:
            return False
        cells = [cell_index(self._by_name[nm].origin,
                            self._by_name[nm].distance, seam=True)
                 for nm in names]
        (a1, b1), (a2, b2) = cells
        return abs(a1 - a2) + abs(b1 - b2) == 1

    def _register_adjacent(self, i, step):
        """Classify the direct seam by the rule table (live probe = ground
        truth) and register the chosen construction: rows 2/3 -> the
        stretched-stabilizer wall coupler; rows 1/4 -> the plain/recoloured
        merge through the zero-cell corridor (route=[])."""
        names = [nm for nm, _ in step.interaction_type]
        views = {nm: patch_view(self.system, nm) for nm in names}
        tgt = dict(step.interaction_type)
        rule = classify_seam(views[names[0]], tgt[names[0]],
                             views[names[1]], tgt[names[1]])
        cons = step.construction if step.construction != 'auto' \
            else rule.construction
        if cons == 'wall':
            spec = wall_spec(views[names[0]], views[names[1]], tgt)
            self.system.register_coupler(
                RotatedSeamWallCoupler(), patch_names=names, name=f'ppm_{i}',
                spec=spec, occupied=frozenset(self.system.index_map))
            self._walls[i] = spec
            self._sched[i] = self._pick_schedule(step, is_wall=True,
                                                 collinear=True)
        else:
            self._register_ppm(i, step)
            self._sched[i] = self._pick_schedule(step, is_wall=False,
                                                 collinear=True)
        self._rules[i] = rule

    def _invalidate_downstream_registration(self, i, rotated_name):
        """After an UNPLANNED runtime rotation of ``rotated_name`` at step
        ``i`` (the feasibility-only fallback, ``auto_rotate`` off), every later
        step touching that patch was pinned/registered for the PRE-rotation
        orientation.  Drop the planner pins AND any coupler already registered
        UP FRONT (remove it and null the route) so the in-loop register
        rebuilds it against the new orientation — the ``auto_rotate``-off twin
        of the planner's ``_rot_lazy`` laziness.  Without this the stale
        coupler is activated as-is: its seam band binds the pre-rotation
        qubits, so the merge is non-deterministic (it crashes in hook-benign
        scheduling, or the readout is not deterministic)."""
        for j in range(i + 1, len(self.ppm_sequence)):
            if not any(n == rotated_name
                       for n, _ in self.ppm_sequence[j].interaction_type):
                continue
            self._planned_tree.pop(j, None)
            self._planned_conj.pop(j, None)
            # the memoized router cells and the pinned schedule were both
            # derived from the pre-rotation geometry: left in place, the
            # rebuild would replay the stale corridor (router= hook) or run
            # the stale schedule on a differently-bent fresh route
            self._router_cache.pop(j, None)
            self._sched.pop(j, None)
            if self._routes[j] is not None:
                # pre-registered up front for the OLD orientation: remove the
                # stale coupler (frees its name + corridor coords) and null the
                # route so the in-loop `_routes[j] is None` fallback rebuilds it
                self.system.remove_coupler(f'ppm_{j}')
                self._routes[j] = None

    def _register_adjacent_repaired(self, i, step):
        """:meth:`_register_adjacent` plus the direct-seam twin of the
        feasibility-only rotation fallback (the routed twin is
        :meth:`_register_with_blocked_repair`): independent of
        ``auto_rotate``, a seam whose live orientations violate the parallel
        law gets its wrong-facing targets physically rotated HERE and is then
        classified again.  With ``auto_rotate`` on, :meth:`_plan_rotations`
        already planned exactly these moves, so the repair finds nothing to
        do and this is a plain call; with it off, this is the only rotation
        the step ever gets.  An unplanned rotation invalidates the planner's
        pins (tree/conj) on every later step touching that patch — those were
        chosen for the pre-rotation orientation, and re-routing is honest
        while re-using them is silently wrong."""
        try:
            return self._register_adjacent(i, step)
        except (BentLayoutError, ValueError) as original:
            moves = self._seam_parallel_moves(step, self._eff_orient,
                                              self._conj_live)
            if not moves:
                raise
            try:
                self._apply_rotation_moves(i, moves)
            except (BentLayoutError, ValueError, RuntimeError):
                raise original from None
            for nm, _kind in moves:
                self._invalidate_downstream_registration(i, nm)
            return self._register_adjacent(i, step)

    def _register_with_blocked_repair(self, i, step, exclude=frozenset()):
        """Register PPM ``i``; on failure, the FEASIBILITY-ONLY rotation
        fallback (design decision 2026-08-05): independent of ``auto_rotate``,
        when a step cannot be constructed in the live orientations (blocked
        legal faces), probe each non-Y target with its orientation flipped
        and litinski-rotate the first one that makes the step routable —
        physically HERE, between the previous split and this merge.
        Optimization rotations (shorter-bus trades) stay behind
        ``auto_rotate``/``rotate_saving_threshold``; this path only ever
        fires when the alternative is a hard error, which it re-raises when
        no single flip repairs the step."""
        try:
            return self._register_ppm(i, step, exclude=exclude)
        except (BentLayoutError, ValueError) as original:
            for nm, _ in step.interaction_type:
                if nm in self._y_names:
                    continue        # |Y> ancillas never rotate (protocol-fixed)
                trial = dict(self._eff_orient)
                trial[nm] = _FLIP_O[trial[nm]]
                sp = [s for s in self._specs_for_step(i, trial)
                      if s.name not in exclude]
                cj = frozenset(n for n in self._conj_live
                               if self._conj_live[n])
                ok = None
                for bus_c in ('X', 'Z'):
                    ok = self._route_result(sp, step, bus_c, probe=True,
                                            conj=cj, step_index=i)
                    if ok is not None:
                        break
                if ok is None:
                    continue
                try:
                    rotate_patches_litinski(
                        self.system, self.builder, [nm],
                        directions={nm: self._lit_direction(
                            self._eff_orient[nm], self._conj_live[nm])})
                except (BentLayoutError, ValueError, RuntimeError):
                    continue        # protocol precondition refused; next patch
                self._type_live[nm] = (CONJUGATE
                                       if self._type_live[nm] == TEXTBOOK
                                       else TEXTBOOK)
                self._eff_orient[nm] = _FLIP_O[self._eff_orient[nm]]
                self._orient[nm] = self._eff_orient[nm]
                self.rotation_count += 1
                self.rotations.append((i, nm))
                self.rotation_log.append((i, nm, 'litinski'))
                self._invalidate_downstream_registration(i, nm)
                return self._register_ppm(
                    i, step, exclude=exclude,
                    repair_route=(bus_c, cj, sorted(tuple(c) for c in ok.tree)))
            raise original

    def _pick_schedule(self, step, *, is_wall, collinear):
        """The merged-round schedule, decided right after routing (the
        user's rule): a wall or any corridor bend forces the whole merged
        block onto the diagonal schedule; straight corridors stay bent."""
        if is_wall:
            if step.schedule == 'bent' or self.schedule == 'bent':
                raise SeamRuleError(
                    "a wall step cannot run the bent schedule: the bent chunk "
                    "cannot express stretched (kf) checks — use 'diagonal'")
            return 'diagonal'
        if step.schedule is not None:
            return step.schedule
        if self.schedule in ('bent', 'diagonal'):
            return self.schedule
        return 'bent' if collinear else 'diagonal'

    def _register_ppm(self, i, step, exclude=frozenset(),
                      blocked_cells=frozenset(), force_present=frozenset(),
                      repair_route=None):
        """Route and register PPM ``i``'s coupler.  ``exclude`` names retired patches
        whose freed cells this corridor reuses — dropping them from the spec/obstacle
        set lets the router pass THROUGH their coarse cells.  Specs carry each patch's
        EFFECTIVE orientation (``self._eff_orient``), so a rotated patch is routed and
        its coupler native-locked in the flipped orientation it now physically has."""
        specs = [sp for sp in self._specs_for_step(i, self._eff_orient,
                                                   force_present=force_present)
                 if sp.name not in exclude]
        bus = self._bus_for(i, step)
        if step.route is None and i in self._adj_steps:
            step = replace(step, route=[])      # zero-cell direct seam
        if step.route is None:
            _pt = getattr(self, '_planned_tree', {}).get(i)
            if _pt is not None:
                step = replace(step, route=_pt)   # build the PLANNED corridor
        try:
            r = self._route_result(
                specs, step, bus=self._derived_bus.get(i),
                conj=getattr(self, '_planned_conj', {}).get(i),
                raise_errors=True, blocked_cells=blocked_cells, step_index=i)
        except BentLayoutError:
            if repair_route is None:
                raise
            # Preserve a successful legacy retry verbatim. Only a geometry
            # failure, before schedule/registry mutations below, may replay
            # the feasibility probe's (bus, conjugation, corridor) choice.
            repair_bus, repair_conj, repair_tree = repair_route
            repair_step = replace(step, route=repair_tree)
            r = self._route_result(
                specs, repair_step, bus=repair_bus, conj=repair_conj,
                raise_errors=True, blocked_cells=blocked_cells, step_index=i)
            if r.status == 'ok':
                self._derived_bus[i] = repair_bus
                self._planned_conj[i] = repair_conj
                self._planned_tree[i] = repair_tree
        if r.status != 'ok':
            raise ValueError(
                f'route_and_build failed at PPM {i}: {r.status} — {r.message}')
        # a dispatch-raised stretched wall in the layout makes this a WALL
        # step: kf records demand the diagonal schedule and the wall-aware
        # coupler metadata (same treatment as the snake path)
        has_kf = any(ch.get('kf')
                     for ch in (r.layout.checks if r.layout is not None else ()))
        if i not in self._sched:
            # schedule decision straight off the route geometry: any bend
            # (cells not collinear) forces the diagonal schedule
            names = [nm for nm, _ in step.interaction_type]
            cells = [cell_index(self._by_name[nm].origin,
                                self._by_name[nm].distance, seam=True)
                     for nm in names] + [tuple(c) for c in (r.tree or ())]
            collinear = (len({a for a, _ in cells}) == 1
                         or len({b for _, b in cells}) == 1)
            self._sched[i] = self._pick_schedule(step, is_wall=has_kf,
                                                 collinear=collinear)
        planned = getattr(self, '_planned_conj', {}).get(i)
        minority = planned if planned is not None else frozenset(
            nm for nm, _ in step.interaction_type
            if getattr(self, '_conj_live', {}).get(nm))
        self.system.register_coupler(
            RotatedRoutedMultiPatchCoupler(),
            patch_names=[nm for nm, _ in step.interaction_type],
            name=f'ppm_{i}', specs=specs,
            target=step.interaction_type, subset_route=r, seam=True,
            minority_names=minority)
        self._routes[i] = r
        if exclude:
            # the reused cells were masked & flagged retired when their patch was
            # measured out; as this coupler's corridor they are active again — clear
            # them from the tracker's retired set so num_active_qubits stays honest.
            reused = set(self.system.local_to_global_map[f'ppm_{i}'].values())
            self.tracker.retired_qubits.difference_update(reused)

    def _alloc_patch(self, name, reg):
        """Create one patch's physical qubits in the system (define-by-run auto-expands
        the tracker when this runs mid-sequence). Applies the registration's conjugation
        / geometry flip, same as the up-front allocation path."""
        s = self._by_name[name]
        is_conj, geo = reg[name]
        p = RotatedSurfaceCode(distance=s.distance)
        if geo == 'X_horizontal':
            p.transpose_coords()
        if is_conj:
            conjugate_patch_records(p)
        if name in self.colour_swapped:
            conjugate_patch_records(p)      # colour knob: swap red/blue only
        self.system.add_patch(p, name=name, offset=(s.origin[0] - 1, s.origin[1] - 1))

    def _alloc_y_degenerate(self, name):
        """Allocate a 'Y' patch as Gidney's degenerate XXZZ-boundary patch (the
        pre-birth form; ``_gidney_birth_prologue`` grows it into the real code)."""
        s = self._by_name[name]
        if s.orientation != 'X_vertical':
            raise ValueError(
                f"Y initial state on {name!r}: the Gidney birth (v1) supports "
                f"only X_vertical orientation — the degenerate XXZZ patch has "
                f"a fixed boundary layout; got {s.orientation!r}")
        if name in self.colour_swapped:
            raise ValueError(
                f"Y initial state on {name!r} cannot be colour-swapped (v1)")
        degen = make_degenerate_y_boundary_patch(s.distance)
        gp = self.system.add_patch(
            degen, name=name, offset=(s.origin[0] - 1, s.origin[1] - 1))
        self._y_gps[name] = gp      # PRE-grow view: the transition chunk needs it

    def _mask_active(self, keep=None, drop=None):
        """Snapshot active_stabilizer_indices and set it to ``keep`` (exact
        set) or active-minus-``drop``.  Returns the snapshot for restore."""
        sysm = self.system
        saved = set(sysm.active_stabilizer_indices)
        sysm.active_stabilizer_indices.clear()
        sysm.active_stabilizer_indices.update(
            set(keep) if keep is not None else saved - set(drop))
        return saved

    def _restore_active(self, saved):
        self.system.active_stabilizer_indices.clear()
        self.system.active_stabilizer_indices.update(saved)

    def _gidney_chunk(self, uids, scheduling):
        """One gidney-scheduled SE round covering exactly ``uids``."""
        saved = self._mask_active(keep=uids)
        try:
            return RotatedSurfaceCodeExtractionBlock(
                self.system, scheduling=scheduling).circuit
        finally:
            self._restore_active(saved)

    def _spectator_chunk(self, exclude_uids):
        """One bent SE round for every active check EXCEPT ``exclude_uids``
        (the birthing patches, which get their own gidney chunk)."""
        sysm = self.system
        saved = self._mask_active(drop=exclude_uids)
        try:
            if not sysm.active_stabilizer_indices:
                return stim.Circuit()
            owner = sysm.index_to_owner_map
            domains = {tuple(sysm.qubit_coords[q]): self._orient[owner[q]]
                       for q in sysm.data_indices
                       if owner.get(q) in self._orient}
            return se_round_chunk(sysm, domains=domains)
        finally:
            self._restore_active(saved)

    def _birth_begin(self, names):
        """Start a deferred Gidney birth batch: allocate the degenerate
        patches, emit the anti-diagonal mixed reset, and prepare the
        reversed-order pre-round chunk.  The reversed rounds themselves run
        either zipped onto the host step's merged rounds (ride) or inside
        ``_birth_finish`` (standalone fallback)."""
        names = sorted(names)
        dists = {self._by_name[nm].distance for nm in names}
        if len(dists) != 1:
            raise ValueError(
                f"a Y birth batch must share one distance (transition rounds "
                f"are tick-zipped): got {sorted(dists)} for {names}")
        d = dists.pop()
        for nm in names:
            self._alloc_y_degenerate(nm)
        uids = set()
        for nm in names:
            uids |= set(self._y_gps[nm]._registered_stabilizer_uids)
        init = {}
        for nm in names:
            ox = self._by_name[nm].origin[0] - 1
            oy = self._by_name[nm].origin[1] - 1
            for q in self._y_gps[nm].data_indices:
                x, y = self.system.qubit_coords[q]
                init[q] = ('Z' if (x - ox - 1) // 2 + (y - oy - 1) // 2 < d
                           else 'X')
        self.builder.initialize(init_dict=init, n=self.system.num_qubits)
        self._birth_state = {
            'names': names, 'd': d, 'degen_uids': uids, 'rides': 0,
            'pre_rounds': 1 + d // 2,
            'rev_chunk': self._gidney_chunk(uids, 'gidney_reversed')}
        return self._birth_state

    def _birth_finish(self):
        """Finish the current birth batch: any reversed pre-rounds not yet
        ridden, then grow -> ONE composite relay round (all transitions
        tick-zipped strictly, then pad-zipped with a spectator SE round so
        nobody gets a blind round) -> declare_logical(Y_L) -> one forward
        'gidney' seal round (composited likewise) -> release the
        degenerate-only ghost syndrome sites."""
        st = self._birth_state
        self._birth_state = None
        sysm, builder = self.system, self.builder
        names, d, uids = st['names'], st['d'], st['degen_uids']
        for _ in range(max(0, st['pre_rounds'] - st['rides'])):
            builder.apply_syndrome_extraction(
                circuit_chunk=_zip_round_chunks(
                    [self._spectator_chunk(uids), st['rev_chunk']], pad=True),
                rounds=1)
        others = set(sysm.active_stabilizer_indices) - uids
        for nm in names:
            sp = self._by_name[nm]
            sysm.grow_patch(nm, RotatedSurfaceCode(distance=d),
                            offset=(sp.origin[0] - 1, sp.origin[1] - 1))
        grown = set(sysm.active_stabilizer_indices) - others
        trans = _zip_round_chunks(
            [make_y_transition_chunk(sysm, nm, self._y_gps[nm],
                                     direction='init') for nm in names])
        builder.apply_relay_chunk(_zip_round_chunks(
            [trans, self._spectator_chunk(grown)], pad=True))
        n = self.tracker.num_qubits
        for nm in names:
            yl = np.zeros(2 * n, dtype=np.uint8)
            for rec in (x for x in sysm.logical_ops
                        if x.get('patch_name') == nm):
                for q, pp in rec['pauli'].items():
                    if pp in ('X', 'Y'):
                        yl[q] ^= 1
                    if pp in ('Z', 'Y'):
                        yl[n + q] ^= 1
            self.tracker.declare_logical(yl)
        builder.apply_syndrome_extraction(
            circuit_chunk=_zip_round_chunks(
                [self._spectator_chunk(grown),
                 self._gidney_chunk(grown, 'gidney')], pad=True),
            rounds=1)
        # release the degenerate-only syndrome sites: grow never prunes, but
        # a normal patch has no qubit at e.g. local (4,0)/(0,4) — leaving
        # them ACTIVE makes later coupler placement collide with ghosts
        live_syn = set(sysm.active_syndrome_indices)
        for nm in names:
            leftover = set(self._y_gps[nm].syndrome_indices) - live_syn
            sysm.active_qubit_indices.difference_update(leftover)

    def _readout_with_capture(self, meas_dict):
        """apply_data_readout for PATCH readouts (liveness retire + final),
        capturing (a) each read patch's deterministic pre-readout parity of
        its final letter — solved BEFORE process_data_measurement reconciles
        the tracker rows — and (b) the absolute record index of every data
        qubit (builder emission order: MX block, MY block, M block, each in
        dict insertion order).  Corridor readouts do NOT go through here."""
        owner = self.system.index_to_owner_map
        for nm in {owner.get(q) for q in meas_dict}:
            letter = self.final_measure_states.get(nm) if nm else None
            if letter and (nm, letter) not in self.prereadout_parity:
                self.prereadout_parity[(nm, letter)] = self._parity_records(
                    self._pauli_vec([(nm, letter)]))
        base = self.tracker.total_measurements
        self.builder.apply_data_readout(final_measurements=meas_dict)
        ordered = ([q for q, b in meas_dict.items() if b == 'X']
                   + [q for q, b in meas_dict.items() if b == 'Y']
                   + [q for q, b in meas_dict.items() if b == 'Z'])
        for off, q in enumerate(ordered):
            self.final_readout_recs[q] = base + off
        # per-(patch, letter) snapshot of the logical support's readout
        # records: final_readout_recs is keyed by QUBIT INDEX, so a later
        # patch reborn on the same cells (scripted tile reuse) overwrites
        # a retired patch's entries; this map keeps each life's own records
        rec_of = {q: base + off for off, q in enumerate(ordered)}
        for nm in {owner.get(q) for q in meas_dict}:
            letter = self.final_measure_states.get(nm) if nm else None
            if not letter:
                continue
            support = [dq for rec in self.system.logical_ops
                       if rec.get('patch_name') == nm
                       and rec.get('type') == letter
                       for dq in rec['pauli']]
            if support and all(dq in rec_of for dq in support):
                self.patch_support_recs[(nm, letter)] = \
                    [rec_of[dq] for dq in support]

    def _init_and_baseline(self, names):
        """Reset the data qubits owned by ``names`` to their initial states, then run the
        baseline SE (``rounds_init``) that establishes their stabilizers."""
        owner = self.system.index_to_owner_map
        init_dict = {q: self.initial_states[owner[q]]
                     for q in self.system.data_indices if owner.get(q) in names}
        if init_dict:
            self.builder.initialize(init_dict=init_dict, n=self.system.num_qubits)
        self._standalone_se(self.rounds_init)
        # virgin-moment seed closures (see build()): the init-letter logical
        # is pinned by the reset just banked, before any window can frame it
        for nm in names:
            letter = self.initial_states.get(nm)
            if letter and (nm, letter) not in self.seed_closure:
                c = self._parity_records(self._pauli_vec([(nm, letter)]))
                if c is not None:
                    self.seed_closure[(nm, letter)] = c

    def build(self):
        reg = self._resolve_registration()
        # build() is re-runnable: everything below re-derives from the specs.
        # Two pieces of state are MUTATED during a run and must be reset here
        # or a second build() silently degrades (M5): _orient (rotations
        # rewrite it in place; a stale post-rotation orientation feeds the
        # WRONG hook-direction table to every SE round — distance halves with
        # no other symptom) and the rotation accumulators.
        self._orient = {s.name: (_FLIP_O[s.orientation]
                                 if s.name in self.colour_swapped
                                 else s.orientation)
                        for s in self.patches}
        self.rotation_count = 0
        self.rotations = []
        self.rotation_log = []
        self.system = QECSystem()
        self._by_name = {s.name: s for s in self.patches}
        first_use = {nm: self.lifetimes[nm][0] for nm in self.lifetimes}
        # effective orientation of each patch, updated as rotations are applied to the
        # live system; _register_ppm routes/native-locks every coupler through it.
        # a colour-swapped patch's labels flip at allocation, so its live
        # orientation starts flipped too
        self._eff_orient = {s.name: (_FLIP_O[s.orientation]
                                     if s.name in self.colour_swapped
                                     else s.orientation)
                            for s in self.patches}
        # auto_rotate: plan the rotations up front (route_and_build is a pure function
        # of specs+target). _rotate_plan[i] = [(patch, kind), ...] moves before step i;
        # _rot_lazy = steps whose coupler must be registered lazily (post-rotation);
        # _derived_bus[i] = the step's bus under the live conventions.
        # _conj_live tracks each patch's live conjugation bit (= checkerboard phase),
        # flipped by every rotate_90; litinski reads it for its growth direction.
        self._conj_live = {s.name: reg[s.name][0] ^ (s.name in self.colour_swapped)
                           for s in self.patches}
        # live TYPE (weight-2 positions): textbook iff the BUILT geometry is
        # X_vertical (untransposed); litinski flips it, rotate_90/colour
        # swaps do not.  The live probe (patch_view) is ground truth at
        # registration; this bit exists for planning before allocation.
        self._type_live = {s.name: (TEXTBOOK if reg[s.name][1] == 'X_vertical'
                                    else CONJUGATE)
                           for s in self.patches}
        # direct-seam steps (two cell-adjacent targets): classified by the
        # rule table against the LIVE system, so always registered lazily
        self._adj_steps = {i for i, st in enumerate(self.ppm_sequence)
                           if self._adjacent_pair(st)}
        self._walls = {}
        self._rules = {}
        self._sched = {}
        self._snake_plans = {}
        self._planned_conj, self._planned_tree = {}, {}
        if self.auto_rotate:
            self._rotate_plan, self._rot_lazy, self._derived_bus = (
                self._plan_rotations(reg))
        else:
            self._rotate_plan, self._rot_lazy = {}, set()
            self._derived_bus = {}  # reset runtime-rescue choices on rebuild

        # Allocate patches: all up front, or — with first_use_init — only those first used at
        # step 0. The rest are allocated + initialised at their first PPM in the loop below
        # (the timing mirror of the liveness path retiring a patch at its last PPM — it
        # saves alloc/reset/baseline cost, but a deferred patch's cell is still an obstacle
        # to earlier routes; only retire frees a cell for corridor reuse).
        allocated = set()
        self._y_gps = {}
        # Y patches are never allocated up front: each is born in place
        # (deferred Gidney birth) during the step before its first use
        for s in self.patches:
            if s.name in self._y_names:
                continue
            if not self.first_use_init or first_use.get(s.name, 0) == 0:
                self._alloc_patch(s.name, reg)
                allocated.add(s.name)

        # Register couplers up front (BEFORE the tracker exists) — EXCEPT: (a) the ones
        # that reuse a retired patch's freed space as corridor, and (b) with first_use_init,
        # any whose patches are not all allocated yet. Those are registered LAZILY in the
        # sequence loop, after the retirement / deferred allocation, so registration cannot
        # steal a still-live patch's qubits.
        self._routes = [None] * len(self.ppm_sequence)
        self._lazy = self._lazy_reuse_map()
        # deferred Gidney births: batch Y patches by first-use step; each
        # batch RIDES the previous routed step's merged rounds (reversed
        # pre-rounds zipped in), with transition+seal at the step boundary;
        # fallback = standalone composite birth at the top of the use step
        self._birth_batch = {}
        for nm in sorted(self._y_names):
            self._birth_batch.setdefault(first_use.get(nm, 0), []).append(nm)
        # batch planning happens BEFORE up-front registration so that
        # multi-batch members can skip it: an up-front route is built under
        # the SERIAL contract (cell borrowing, retirement reuse) and cannot
        # share a window — the portfolio routes those members fresh at
        # batch time, against the batch's true obstacle set
        if self.parallel_steps:
            self._plan_step_batches()
        else:
            self._step_batches = [[i] for i in range(len(self.ppm_sequence))]
            self._batch_first = {}
            self._batch_done = set()
            self._batch_members = set()
        for i, step in enumerate(self.ppm_sequence):
            if i in self._lazy or i in self._rot_lazy or i in self._adj_steps \
                    or i in self._snake_plans or i in self._batch_members:
                continue
            if any(nm not in allocated for nm, _ in step.interaction_type):
                continue    # includes every step targeting a Y patch (born
                # in the loop); the `self._routes[i] is None` fallback
                # registers those post-birth
            try:
                self._register_ppm(i, step)
            except (BentLayoutError, ValueError):
                # defer to the in-loop lazy register, where the
                # feasibility-only rotation fallback can act at the right
                # physical moment (between the previous split and this
                # merge); if the fallback cannot repair it either, the
                # error re-raises THERE with full context
                self._routes[i] = None

        self._setup()
        self.builder.write_coordinates()

        # {(patch, init_letter): records} captured at each patch's VIRGIN
        # moment (seeded, before any merge window).  A later window's split
        # leaves a Pauli byproduct frame on its patches, so a post-closure
        # reads the frame-contaminated actual value; the seed closure is the
        # frame-free ideal anchor the reporting layer reconstructs program
        # bits from (measured 2026-08-05: scheduling hoisted dj_8's X^8 step
        # ahead of four Z steps and their post-closures went per-shot random
        # while the ideal values are constants).
        self.seed_closure = {}

        # init the already-allocated patches, then baseline SE
        self._init_and_baseline(allocated)

        self._born = set()
        self._birth_state = None

        self.step_joint_records_pre = {}
        self.step_joint_records_post = {}
        # terminal-readout capture (folded weight-1 program bits, ancilla
        # readouts): deterministic pre-readout parity per (patch, letter) and
        # the absolute record index of every read data qubit
        self.prereadout_parity = {}
        self.final_readout_recs = {}
        self.patch_support_recs = {}
        owner = self.system.index_to_owner_map  # live dict — add_patch below mutates it
        measured = set()
        for i, step in enumerate(self.ppm_sequence):
            # fallback birth: any Y patch first used HERE that did not get to
            # ride the previous step (step 0 / wall / snake host)
            todo = [nm for nm in self._birth_batch.get(i, ())
                    if nm not in self._born]
            if todo:
                self._birth_begin(todo)
                self._birth_finish()
                self._born.update(todo)
                allocated.update(todo)

            if i in self._batch_done:
                continue
            b = self._batch_first.get(i)
            if b is not None and self._try_run_batch(b, reg, owner, measured,
                                                     allocated):
                continue
            # deferred allocation: bring in + initialise any patch first used at this step
            newly = [nm for nm, _ in step.interaction_type if nm not in allocated]
            assert not any(nm in self._y_names for nm in newly), \
                f"Y patch reached deferred alloc without a birth: {newly}"
            # lifetime_overrides init side: a widened init layer allocates
            # the patch HERE (idle until its first use), so the physical
            # birth and the router's presence rule (_absent: fu > i) read
            # one ledger.  Empty without overrides: an un-widened patch
            # with fu == i is one of this step's targets already.
            newly += sorted(nm for nm, (fu, _lu) in self.lifetimes.items()
                            if fu <= i and nm not in allocated
                            and nm not in newly and nm not in self._y_names)
            for nm in newly:
                self._alloc_patch(nm, reg)
                allocated.add(nm)
            if newly:
                self._init_and_baseline(set(newly))

            # auto_rotate: apply the planned PHYSICAL rotations BEFORE this PPM's
            # coupler is registered, so the merged checks bind to the rotated qubits.
            # rotate_90 = the SWAP-network protocol (flips orientation + conjugation
            # bit); litinski = the 5-step deformation protocol (flips orientation,
            # keeps the conjugation bit; growth direction fixed by the live phase).
            # CONCURRENT application: disjoint patches rotate in the SAME
            # time window — SWAP networks merged tick-for-tick (depth of
            # one rotation), litinski protocols advance stage-by-stage in
            # lockstep sharing the system-wide SE rounds (5*d rounds total)
            self._apply_rotation_moves(i, self._rotate_plan.get(i, []))

            if i in self._snake_plans:
                self._register_snake(i, step, self._snake_plans[i])
                self._apply_ppm_step(i, step)
                self._retire_consumed(i, step, owner, measured, allocated)
                continue
            if i in self._adj_steps and i not in self._walls \
                    and self._routes[i] is None:
                self._register_adjacent_repaired(i, step)
            if i in self._walls:
                self._apply_wall_step(i, step)
                self._retire_consumed(i, step, owner, measured, allocated)
                continue
            if self._routes[i] is None:
                # coupler not registered up front — a reuse step and/or a deferred step.
                # A reuse step passes THROUGH a retired patch's freed cells, so exclude
                # them (drop from the obstacle set) to let the router route through.
                if i in self._lazy:
                    # Budget invariant the lazy register relies on — the
                    # tracker's own: standing + absorbed rank == expected.
                    # NOT the stronger "no absorbed DOF outstanding": a
                    # persisting joint relation (resolve_absorbed=False)
                    # legitimately stays absorbed until its patches retire,
                    # and the unscheduled program order reaches a reuse step
                    # with such relations open (ghz_16_mixed no_sched,
                    # 2026-08-06: guard fired at expected=9 standing=8
                    # absorbed=1 while the compile is healthy — full
                    # distance, faithful 0/56018 once admitted).
                    absorbed = self.tracker.num_absorbed_dof()
                    assert (self.tracker.expected_num_logicals
                            == self.tracker.logicals.count + absorbed), (
                        f"logical budget desync before lazy register of ppm_{i}: "
                        f"expected={self.tracker.expected_num_logicals} "
                        f"standing={self.tracker.logicals.count} "
                        f"absorbed={absorbed}")
                self._register_with_blocked_repair(
                    i, step, exclude=self._lazy.get(i, frozenset()))
            ride = [nm for nm in self._birth_batch.get(i + 1, ())
                    if nm not in self._born]
            self._apply_ppm_step(i, step, birth_names=ride or None)
            if ride:
                self._birth_finish()
                self._born.update(ride)
                allocated.update(ride)
            self._retire_consumed(i, step, owner, measured, allocated)
        leftover_y = [nm for nm in sorted(self._y_names)
                      if nm not in self._born]
        if leftover_y:
            self._birth_begin(leftover_y)
            self._birth_finish()
            self._born.update(leftover_y)
        meas_dict = {q: self.final_measure_states[owner[q]]
                     for q in self.system.data_indices
                     if owner.get(q) in self.final_measure_states and owner.get(q) not in measured}
        if meas_dict:
            self._readout_with_capture(meas_dict)

        if self.noise_params is not None:
            return self.builder.build_noisy_circuit(
                noise_params=self.noise_params, noise_model=self.noise_model)
        return self.builder.circuit
