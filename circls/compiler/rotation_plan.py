"""Paper Section 3: the rotation planner.

Decides which patches to rotate before which PPM (litinski batches,
seam-parallel moves, the candidate search under
_ROTATION_SEARCH_CAP).  Pure planning: protocol EXECUTION
(_apply_rotation_moves) stays in the core driver.  Mixed into
SequentialPPMExperiment.
"""
from __future__ import annotations

from itertools import product

from circls.core.common import _CONJ, _FLIP_O, _ROTATION_SEARCH_CAP
from circls.core.multi_patch_coupler import BentLayoutError, cell_index


class RotationPlannerMixin:
    """Rotation planning (docs/ARCHITECTURE.md)."""

    @staticmethod
    def _lit_direction(orientation, conj):
        """The Litinski growth direction fixed by (live orientation, live
        conjugation bit): X̄-horizontal starts grow along ±y (the paper
        protocol), X̄-vertical starts along ±x (its transpose); the sign is
        the checkerboard phase = the conjugation bit."""
        axis = 'x' if orientation == 'X_vertical' else 'y'
        return ('+' if conj else '-') + axis

    def _bar_free(self, nm, conj, orientation):
        """Litinski precondition: the (2d-1)×d bar grows one coarse cell past
        the home tile in the phase-fixed direction (see _lit_direction).
        Free = no OTHER patch's home cell there (conservative: liveness
        timing is not consulted)."""
        s = self._by_name[nm]
        cell = cell_index(s.origin, s.distance, seam=True)
        direction = self._lit_direction(orientation, conj)
        step_ = 1 if direction[0] == '+' else -1
        bar_cell = ((cell[0] + step_, cell[1]) if direction[1] == 'x'
                    else (cell[0], cell[1] + step_))
        return all(cell_index(o.origin, o.distance, seam=True) != bar_cell
                   for o in self.patches if o.name != nm)

    @staticmethod
    def _seam_axis_of(cells):
        """The seam LINE direction between two cell-adjacent coarse cells —
        the same convention as ``joint_merge._seam_axis`` ('vertical' =
        patches side by side, 'horizontal' = stacked)."""
        (_ax, ay), (_bx, by) = cells
        return 'vertical' if ay == by else 'horizontal'

    @staticmethod
    def _parallel_orientation(pauli, axis):
        """The ONE orientation a direct-seam target may have: the measured
        logical must run PARALLEL to the seam line (``_check_parallel``), and
        X̄ runs horizontal exactly when the patch is X_horizontal — so
        (axis, letter) fixes the orientation with no freedom left."""
        return ('X_horizontal' if (axis == 'horizontal') == (pauli == 'X')
                else 'X_vertical')

    def _seam_parallel_moves(self, step, eff, conj):
        """The rotations a DIRECT-SEAM step FORCES, given the orientations in
        ``eff`` (and the colours in ``conj``).

        A direct seam carries no free knob — :func:`_parallel_orientation`
        fixes each target's orientation — so this is a constraint, not a
        cost choice: a target already facing that way stays put, one facing
        the wrong way must be physically rotated HERE (between the previous
        split and this merge).  That is the case a single birth orientation
        cannot cover: a data patch joint-measured in X̄ across one seam and
        in Z̄ across another needs both orientations over its life (the
        foreign compact layouts: DASCOT's CNOT gadgets sandwich an H between
        two CXs on the same patch).

        Protocol choice respects ``rotation_kind`` and the existing
        constraints: litinski first (it keeps the COLOURS, so no downstream
        corridor painting is disturbed) when its (2d-1)xd bar has room,
        rotate_90 (in place, swaps the colours) otherwise; a |Y> ancilla
        never rotates.  Returns ``[(name, kind), ...]`` — empty when the live
        orientations already satisfy the parallel law, or when no protocol is
        allowed (registration then raises the parallel-law error, as before).
        """
        names = [nm for nm, _ in step.interaction_type]
        cells = [cell_index(self._by_name[nm].origin,
                            self._by_name[nm].distance, seam=True)
                 for nm in names]
        axis = self._seam_axis_of(cells)
        allow_lit = self.rotation_kind in ('auto', 'litinski')
        allow_r90 = self.rotation_kind in ('auto', 'rotate_90')
        moves = []
        for nm, pauli in step.interaction_type:
            if nm in self._y_names:
                continue        # |Y> ancillas never rotate (protocol-fixed)
            if eff[nm] == self._parallel_orientation(pauli, axis):
                continue
            if allow_lit and self._bar_free(nm, conj[nm], eff[nm]):
                moves.append((nm, 'litinski'))
            elif allow_r90:
                moves.append((nm, 'rotate_90'))
        return moves

    def _plan_rotations(self, reg):
        """Rotation plan (auto_rotate) — the user's unified step-1 search
        (2026-07-31): per step, enumerate EVERY candidate

            (per-target rotation choice in {stay, rotate_90, litinski})
            x (bus letter in {X, Z})

        and score each feasible one with the GEOMETRIC probe only (route +
        parallel law + seam table; zero construction):

            score = corridor cells + rotate_saving_threshold x rotation BATCHES

        (a batch = one protocol kind used this step: both protocols apply
        their rotations concurrently, so time cost is per batch, not per
        rotated patch — 2026-08-04 model fix).  Lowest score wins; ties ->
        shorter corridor (so a rotation batch is adopted exactly when it
        saves >= the threshold), then FEWER ROTATIONS (each rotated patch
        pays real error exposure even inside a shared batch), then fewer
        stretched walls, then the majority-letter bus.  Births carry NO free knobs: every patch is
        born exactly as the input declares (orientation from its spec,
        colours from ``colour_swapped``); any orientation change is a
        PHYSICAL rotation and is costed.  ``rotation_kind`` restricts the
        candidate kinds ('auto' allows both).

        The snake gateway stays ahead of the search: an explicit-route
        step whose input-declared colour swap violates the corridor
        parity law takes the stretched-wall plan (zero rotations).

        Returns ``(plan, lazy, derived_bus)``: ``plan[i] = [(name, kind), ...]``
        moves to apply on the LIVE system before step ``i`` (kind in
        {'rotate_90', 'litinski'}); ``lazy`` = steps to register after the first
        rotation (real rotation protocols move qubits, so later couplers must bind
        post-rotation); ``derived_bus[i]`` = the step's bus."""
        declared = {s.name: s.orientation for s in self.patches}
        # effective orientation mirrors _eff_orient: a colour-swapped
        # patch's labels flip at allocation, so its live orientation
        # starts flipped too (probes and registration must agree)
        eff = {nm: (_FLIP_O[o] if nm in self.colour_swapped else o)
               for nm, o in declared.items()}
        # live conj = registration bit XOR the user's colour_swapped input
        # (under strict births the ONLY colour source is the declared input)
        conj = {s.name: reg[s.name][0] ^ (s.name in self.colour_swapped)
                for s in self.patches}
        allow_lit = self.rotation_kind in ('auto', 'litinski')
        allow_r90 = self.rotation_kind in ('auto', 'rotate_90')
        plan, lazy, derived_bus = {}, set(), {}
        self._snake_plans = {}
        #: step -> conj-registration frozenset the planner CONFIRMED with;
        #: registration passes it back so the memoized build is reused
        #: verbatim (recomputing from _conj_live could diverge mid-plan)
        self._planned_conj = {}
        #: step -> the corridor tree the winning probe chose.  Registration
        #: builds EXACTLY this route — step 1's decision includes the route
        #: itself (the user's architecture); re-routing at registration can
        #: wander to a different tree and fail where the probed one builds
        #: (measured: threshold=5 15-patch, q12 attach lost on the re-route)
        self._planned_tree = {}
        #: RETIRED (design decision 2026-07-31: births follow the declared
        #: input exactly — no free orientation/colour knobs).  Kept empty
        #: for API compatibility.
        self.birth_reorientations = []
        first_rotation_step = None
        for i, step in enumerate(self.ppm_sequence):
            if self._adjacent_pair(step):
                # direct-seam step: rule-table dispatch, no bus — so it is
                # never SCORED (no corridor to trade against).  It is still
                # a CONSTRAINT: the parallel law fixes both targets'
                # orientations, and a target facing the wrong way must
                # rotate HERE.  Carrying that forward is what lets a later
                # step be planned against the orientation the patch will
                # actually have (before 2026-08-16 this `continue` dropped
                # the constraint AND the plan's book-keeping, so a patch
                # measured X̄ on one seam and Z̄ on another simply raised
                # SeamRuleError at registration — no orientation can serve
                # both).
                moves = self._seam_parallel_moves(step, eff, conj)
                for nm, kind in moves:
                    eff[nm] = _FLIP_O[eff[nm]]
                    if kind == 'rotate_90':
                        conj[nm] = not conj[nm]
                if moves:
                    plan[i] = moves
                    if first_rotation_step is None:
                        first_rotation_step = i
                continue
            moves = []
            names = [nm for nm, _ in step.interaction_type]
            # snake gateway: an explicit route + an INPUT-declared colour
            # swap violating the corridor parity law -> the stretched-wall
            # plan (the user's zero-rotation construction for that family)
            need = {nm: (_CONJ[P] if conj[nm] else P)
                    for nm, P in step.interaction_type}
            if len(set(need.values())) > 1 and step.route:
                snake = self._plan_snake(i, step, eff, conj)
                if snake is not None:
                    self._snake_plans[i] = snake
                    derived_bus[i] = snake['bus']
                    continue

            # ---- unified candidate search (geometric probes only) ----------
            def can_lit(nm):
                return self._bar_free(nm, conj[nm], eff[nm])

            if len(names) > _ROTATION_SEARCH_CAP:
                # High-weight steps are not probed AT ALL (2026-08-04
                # profile: the SINGLE-probe cost explodes with weight —
                # w=10 is ~60 s/probe, w=16 is hours — while the capped
                # candidate set is just {no-rotation} x bus, i.e. there is
                # nothing to choose.  Majority-vote the bus (the pre-planner
                # behaviour) and let registration route the step once for
                # real (measured: the w=16 bv_16 step routes in ~13 s
                # there).  An infeasible big step fails loudly at
                # registration, same as pre-planner.
                derived_bus[i] = self._bus_of(step.interaction_type)
                continue
            kinds_of = {}
            for nm in names:
                ks = [None]
                if nm in self._y_names:
                    # |Y> ancillas never rotate: the Gidney birth layout is
                    # protocol-fixed (v1: X_vertical only) and rotating a
                    # just-born ancilla collides its post-protocol geometry
                    # with the coupler's registration (measured: s-twist
                    # ppm_1 vs active y0 at (24,28), 2026-08-04).  The
                    # planner routes around it instead.
                    kinds_of[nm] = ks
                    continue
                if allow_lit and can_lit(nm):
                    ks.append('litinski')   # tie preference: colour-keeping
                if allow_r90:
                    ks.append('rotate_90')
                kinds_of[nm] = ks
            maj = self._bus_of(step.interaction_type)
            cands = []
            for assign in product(*(kinds_of[nm] for nm in names)):
                kinds = dict(zip(names, assign))
                r_n = sum(1 for k in assign if k)
                trial = dict(eff)
                for nm, k in kinds.items():
                    if k:
                        trial[nm] = _FLIP_O[trial[nm]]
                cj = frozenset(nm for nm in names
                               if conj[nm] ^ (kinds[nm] == 'rotate_90'))
                sp = self._specs_for_step(i, trial)
                for bus_c in ('X', 'Z'):
                    # ONE probe per candidate: corridor length AND wall
                    # surcharge come from the same SubsetRoute — the old
                    # walls=False/walls=True _route_len pair re-ran an
                    # identical route_and_build twice (measured: 133
                    # probes / 526 s on the 30-patch 6-body step, half
                    # redundant), and the winner's tree is kept here so
                    # pinning needs no third probe either
                    r_c = self._route_result(sp, step, bus_c, probe=True,
                                             conj=cj, step_index=i)
                    if r_c is None:
                        continue
                    ln = len(r_c.tree)
                    # TIME cost of rotations is batched: both protocols apply
                    # a step's rotations concurrently (SWAP networks merged
                    # tick-for-tick; litinski stages in lockstep sharing the
                    # system-wide SE rounds), so the fee is per protocol
                    # BATCH, not per rotated patch.  The rotation COUNT stays
                    # as a tie-breaker: each rotated patch still pays real
                    # error exposure (exp1: the rotation window dominates the
                    # intra-layer LER budget).
                    n_batch = ((1 if any(k == 'rotate_90' for k in assign) else 0)
                               + (1 if any(k == 'litinski' for k in assign) else 0))
                    cands.append((
                        ln + self.rotate_saving_threshold * n_batch,
                        ln, r_n, ln + getattr(r_c, 'n_walls', 0),
                        0 if bus_c == maj else 1,
                        [(nm, k) for nm, k in kinds.items() if k],
                        bus_c, cj, sorted(tuple(c) for c in r_c.tree)))
            if not cands:
                orients = {nm: eff[nm] for nm in names}
                raise BentLayoutError(
                    f"PPM {i} has no feasible (rotation, bus) candidate "
                    f"(orientations {orients}, conj "
                    f"{sorted(nm for nm in names if conj[nm])}, "
                    f"rotation_kind={self.rotation_kind!r}) — re-place the "
                    f"patches or give the step an explicit route")
            cands.sort(key=lambda c: c[:5])
            _score, _ln, _rn, _lw, _nb, group, bus, cj, _wtree = cands[0]
            if step.route is None and _wtree:
                self._planned_tree[i] = _wtree
            import os as _os
            if _os.environ.get('LIGHTSTIM_DEBUG_PLAN'):
                import sys as _sys
                print(f"[plan] step{i} pick score={_score} L={_ln} "
                      f"Lw={_lw} bus={bus} rot={group} conj={sorted(cj)} "
                      f"(cands={len(cands)})", file=_sys.stderr)
            for nm, kind in group:
                moves.append((nm, kind))
                eff[nm] = _FLIP_O[eff[nm]]
                if kind == 'rotate_90':
                    conj[nm] = not conj[nm]
            self._planned_conj[i] = cj
            derived_bus[i] = bus
            if moves:
                plan[i] = moves
                if first_rotation_step is None:
                    first_rotation_step = i
            # real rotation protocols move/allocate qubits: every step from the
            # first rotation onward binds its coupler lazily (post-rotation)
            if first_rotation_step is not None and i >= first_rotation_step:
                lazy.add(i)
        return plan, lazy, derived_bus

