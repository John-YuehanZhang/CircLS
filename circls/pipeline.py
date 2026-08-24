"""One-call Clifford QASM -> stim pipeline.

Promoted from ``notebooks/qasm_to_stim.ipynb`` (cell 1) so benchmarks and
experiments import it instead of redefining it.  Placement comes from
``circls.compiler.mapping`` (Square Sparse floor, row-major assignment).
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
from typing import Dict, Optional

import stim

from circls.interop.ir.gosc_gadgets import (append_program_observables,
                                         expand_gadgets,
                                         program_step_interactions,
                                         to_experiment_inputs)
from circls.interop.nwqec.frontend import (load_clifford_mpauli,
                                           load_clifford_t_as_s)
from circls.interop.ir.pauli_algebra import (drop_free_prefix,
                                          reorder_weight1_last, y_free_sweep)
from circls.core.sequential_ppm_ls import SequentialPPMExperiment


@dataclasses.dataclass
class CompiledProgram:
    circuit: stim.Circuit
    observables: Dict[int, Optional[int]]   # program bit -> observable index (None = free coin)
    experiment: SequentialPPMExperiment
    program: object                          # the expanded PPMProgram
    reconstruction: Optional[object] = None  # measure_reduce.Reconstruction (None = pass off)
    reorder_perm: Optional[list] = None      # weight-1 reorder perm[new_pos] = executed index
    m_load_trace: Optional[list] = None      # out-bit i -> LOAD-order m index (full perm chain)
    source_qasm: Optional[str] = None        # the input program (verify's logical oracle)
    t_as_s: bool = False                     # Y-state approximation compile
    graphlike_closures: bool = False         # cross-window closures stripped

    # read-only views of the compiler's decisions (docs/API_HOOKS.md)
    @property
    def placement(self):
        '''{patch name: coarse cell} the mapper chose.'''
        from lightstim.protocols.ppm.spec import cell_index
        return {sp.name: cell_index(sp.origin, sp.distance, seam=True)
                for sp in self.experiment.patches}

    @property
    def routes(self):
        '''Per PPM step: sorted corridor cells of the verified route
        (None where a step needed no corridor).'''
        return [sorted(r.tree) if r is not None and getattr(r, 'tree', None)
                else None for r in self.experiment._routes]

    @property
    def lifetimes(self):
        '''{patch name: (first_use, last_use)} in PPM-layer indices.'''
        return dict(self.experiment.lifetimes)

    def stats(self):
        '''The full metric set (allocated volume, qubit-rounds, ...).'''
        from circls.metrics import experiment_stats
        from circls.metrics.report import to_dict
        return to_dict(experiment_stats(self.experiment, self.circuit))


@dataclasses.dataclass
class ProgramAnalysis:
    """Front-end-only view of a program: what compile_qasm would compile.

    ``program`` is the expanded PPMProgram (feed it to
    ``compile_ppm_sequence``); ``interactions`` is the per-step list of
    ``(patch name, Pauli)`` target sets — the input an external mapper,
    scheduler, or router optimizes against (docs/API_HOOKS.md, the
    recommended extension route)."""
    program: object
    interactions: list

    @property
    def patch_names(self):
        """Every patch the steps touch, sorted."""
        return sorted({nm for step in self.interactions for nm, _ in step})

    @property
    def first_letters(self):
        """{patch name: Pauli of its first use} — the orientation rule's
        input."""
        out: Dict[str, str] = {}
        for step in self.interactions:
            for nm, letter in step:
                out.setdefault(nm, letter)
        return out


def analyze_qasm(qasm: str, *, measure_reduction: bool = True,
                 step_scheduling: bool = True,
                 parallel_steps: bool = True) -> ProgramAnalysis:
    """Run ONLY the front-end passes and return the PPM structure,
    without compiling anything.

    This is the first half of ``compile_qasm`` (load, re-selection,
    ordering, Y-elimination, scheduling, gadget expansion), for callers
    who compute their own placement/orientation/routing and want the
    structure those decisions are made against.  Pass the SAME flags you
    will pass to ``compile_qasm`` so the analyzed steps match the
    compiled ones."""
    raw = load_clifford_mpauli(qasm)
    if measure_reduction:
        from circls.compiler.measure_reduce import reduce_measurements
        raw, _ = reduce_measurements(raw)
    ordered, _ = reorder_weight1_last(raw)
    swept = drop_free_prefix(y_free_sweep(ordered))
    if step_scheduling:
        from circls.compiler.scheduling import schedule_ops
        swept, _ = schedule_ops(swept, parallel=parallel_steps)
    prog = expand_gadgets(swept)
    return ProgramAnalysis(program=prog,
                           interactions=program_step_interactions(prog))


def compile_ppm_sequence(program, distance: int = 3,
                         rounds: Optional[int] = None, noise=None,
                         assignment: str = "row_major",
                         placement=None, orientation=None, lifetime=None,
                         router=None,
                         **experiment_kwargs) -> CompiledProgram:
    """PPMProgram -> compiled program: the mid-pipeline entry point.

    Enter with a PPM sequence from ANY front-end (docs/API_HOOKS.md)
    and skip the QASM loading, re-selection and scheduling passes;
    the placement/orientation/lifetime/router hooks apply exactly as in
    ``compile_qasm``.  Returns a CompiledProgram whose reconstruction
    and reorder metadata are identity (no front-end rewrites
    happened here)."""
    specs, steps, init, final, _ = to_experiment_inputs(
        program, distance=distance, assignment=assignment,
        placement=placement, orientation=orientation)
    if lifetime is not None:
        experiment_kwargs["lifetime_overrides"] = lifetime
    if router is not None:
        experiment_kwargs["router"] = router
    exp = SequentialPPMExperiment(
        specs, steps, initial_states=init, final_measure_states=final,
        rounds=rounds or distance,
        rounds_init=experiment_kwargs.pop("rounds_init", 1),
        noise_params=noise,
        **experiment_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        circuit = exp.build()
    obs = append_program_observables(circuit, exp, program)
    n_bits = len(program.out_bits)
    return CompiledProgram(circuit=circuit, observables=obs,
                           experiment=exp, program=program,
                           reconstruction=None,
                           reorder_perm=list(range(n_bits)),
                           m_load_trace=list(range(n_bits)))


_STRIP_M_KINDS = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}


def strip_cross_window_closures(circuit: stim.Circuit,
                                span_rounds: int = 3,
                                verbose: bool = True) -> stim.Circuit:
    """The ``graphlike_closures`` post-pass: drop DETECTORs whose records
    span more than ``span_rounds`` syndrome windows.

    A k-body PPM whose split leaves m >= 2 target patches alive conserves
    m - 1 relations (products of the survivors' logicals) that no
    per-round check can measure; the tracker banks each one and emits it
    at the readout that finally determines it, as a detector whose
    records span from the split to that readout.  Those long-range
    parities are real syndrome (LER is ~2x better with them on the
    mixed-basis GHZ probe): a fault on the conserved surface picks
    up their symptom on top of its local pair, and stim cannot decompose
    the triple.  Stripping them trades the LER margin for a graphlike
    DEM on circuits where they are the only non-graphlike content
    (measured true for the mixed-basis GHZ; NOT true for the Y-state
    gadget circuits, whose hyperedges live within a window and survive
    this pass untouched).  Ordinary closures (a
    consumption readout against its own merge window) span at most ~2
    windows and are kept; the window length is estimated from the
    circuit's own measurement cadence.  Returns a flattened circuit;
    observables are untouched, so p = 0 determinism and the logical
    check are preserved by construction.  Dropped spans are printed --
    never strip silently.
    """
    flat = circuit.flattened()
    meas_tick, tick = [], 0
    for inst in flat:
        if inst.name == "TICK":
            tick += 1
        elif inst.name == "MPP":
            raise ValueError("strip_cross_window_closures: MPP unsupported")
        elif inst.name in _STRIP_M_KINDS:
            meas_tick.extend([tick] * len(inst.targets_copy()))
    m_ticks = sorted(set(meas_tick))
    gaps = [b - a for a, b in zip(m_ticks, m_ticks[1:])]
    window = max(sorted(gaps)[len(gaps) // 2] if gaps else 1, 1)
    limit = span_rounds * window
    out = stim.Circuit()
    n, dropped = 0, []
    for inst in flat:
        if inst.name in _STRIP_M_KINDS:
            n += len(inst.targets_copy())
            out.append(inst)
        elif inst.name == "DETECTOR":
            ts = [meas_tick[n + t.value] for t in inst.targets_copy()]
            if max(ts) - min(ts) > limit:
                dropped.append(max(ts) - min(ts))
                continue
            out.append(inst)
        else:
            out.append(inst)
    if verbose and dropped:
        print(f"graphlike_closures: dropped {len(dropped)} cross-window "
              f"closure detector(s), record spans {sorted(dropped)} ticks "
              f"(window ~{window} ticks, limit {limit})")
    return out


def compile_qasm(qasm: str, distance: int = 3, rounds: Optional[int] = None,
                 noise=None, assignment: str = "row_major",
                 auto_rotate: bool = False, rotation_kind: str = "litinski",
                 birth_colouring: bool = False,
                 measure_reduction: Optional[bool] = None,
                 step_scheduling: Optional[bool] = None,
                 parallel_steps: bool = True,
                 reselector=None, scheduler=None, placement=None,
                 orientation=None, lifetime=None, router=None,
                 t_as_s: bool = False,
                 graphlike_closures: bool = False,
                 **experiment_kwargs) -> CompiledProgram:
    """Clifford QASM -> compiled program (stim circuit + program observables).

    ``assignment``: 'row_major' (baseline) or 'optimized' (the
    circls.compiler.assignment portfolio).  The rotation PLANNER is OFF by default:
    with measure reduction and wall attachment in place, planned rotations
    measured as a net loss on the benchmark suite (each litinski batch
    burns 5d rounds for a few corridor cells); the planner stays opt-in
    via ``auto_rotate=True`` with its ``rotate_saving_threshold`` knob.  Rotation as a FEASIBILITY fallback
    is always on regardless: a step whose live orientations admit no legal
    construction gets its blocked patch litinski-rotated at the last
    moment — routed steps through ``_register_with_blocked_repair``, direct
    (cell-adjacent) seams through ``_register_adjacent_repaired``, where the
    parallel law fixes the orientation outright.  ``rotation_kind='litinski'``
    is the ONLY fault-tolerant rotation protocol: swap rotation has an
    unprotected unitary window of depth 2(d-1) and must never be the
    default.  ``experiment_kwargs`` pass through to
    ``SequentialPPMExperiment`` (e.g. ``liveness=True, keep_patches={...}``).
    Non-Clifford input is rejected loudly by the front-end.

    ``measure_reduction`` (default ON): re-select the terminal measurement
    set as a minimum-weight generating set of the same commuting group
    (identity rewrite; the executed bits relate to the original program bits
    through ``reconstruction`` — output-side XOR, purely classical).

    ``parallel_steps`` (default ON since 2026-08-06): contiguous
    patch-disjoint steps share one merge window (wall-clock = batches x d).
    Gated green before the flip: truth-oracle extraction equality,
    parallel-vs-serial distribution equality on both placements, decoder
    faithfulness, full graphlike distance.  Batches demote to the serial
    path whenever the planner or router cannot prove window disjointness.
    """
    # customization hooks (docs/API_HOOKS.md): a hook and its shorthand
    # flag together are an error — the flag IS the built-in hook.
    if reselector is not None and measure_reduction is not None:
        raise ValueError("pass reselector= or measure_reduction=, not both")
    if scheduler is not None and step_scheduling is not None:
        raise ValueError("pass scheduler= or step_scheduling=, not both")
    do_reduce = (measure_reduction if measure_reduction is not None
                 else reselector is None)
    do_sched = step_scheduling if step_scheduling is not None else True
    # t_as_s (the S-state proxy): accept Clifford+T, replace every
    # pi/8 rotation by the pi/4 rotation of the same axis and sign, and
    # run the m-block passes on the measurement suffix only.  The
    # compiled program is the PROXY program; source_qasm becomes the
    # T->S substituted flat circuit (pure Clifford), so verify()'s
    # logical oracle runs against the proxy semantics rather than the
    # original non-Clifford program.
    s_head = []
    proxy_qasm = None
    if t_as_s:
        raw_full, proxy_qasm = load_clifford_t_as_s(qasm,
                                                    return_proxy_qasm=True)
        s_head = [op for op in raw_full.ops if op.kind != "m"]
        from circls.interop.ir.pauli_ir import PauliCircuit as _PC
        raw = _PC(raw_full.num_qubits,
                  [op for op in raw_full.ops if op.kind == "m"])
    else:
        raw = load_clifford_mpauli(qasm)
    recon = None
    if reselector is not None:
        from circls.compiler.measure_reduce import verify_reconstruction
        pre_reselect = raw
        raw, recon = reselector(raw)
        verify_reconstruction(pre_reselect, raw, recon)
    elif do_reduce:
        from circls.compiler.measure_reduce import reduce_measurements
        raw, recon = reduce_measurements(raw)
    ordered, perm = reorder_weight1_last(raw)
    m_only_ordered = ordered
    if s_head:
        from circls.interop.ir.pauli_ir import PauliCircuit as _PC
        ordered = _PC(ordered.num_qubits, list(s_head) + list(ordered.ops))
    swept = drop_free_prefix(y_free_sweep(ordered))
    # the m-bit permutation chain (2026-08-05, found by the equivalence
    # verifier as a bit-alignment mismatch): reorder_weight1_last permutes
    # the m stream (perm), the sweep and the prefix drop PRESERVE m order
    # (documented contracts), and step scheduling permutes again below.
    # m_load_trace[i] = LOAD-order index of the i-th m op of the final
    # stream = the canonical identity of program out-bit i.
    pre_sched_m_load = [perm[p] for p, op in enumerate(m_only_ordered.ops)
                        if op.kind == "m"]
    # sweep/drop insert 's' ops but keep every m, in order:
    assert len([o for o in swept.ops if o.kind == "m"]) == \
        len(pre_sched_m_load), "sweep/drop changed the m count"
    if scheduler is not None or do_sched:
        # lifetime-aware step scheduling (2026-08-05): reorder commuting
        # joint steps so births hug first uses and retirements hug last
        # uses; anticommuting pairs keep their order (measured, never
        # assumed), fixed weight-1 ops keep their absolute positions, and
        # the pass can only tie or beat the input order.  Scheduling BEFORE
        # expand_gadgets keeps every record index consistent for free.
        # An injected scheduler= replaces the built-in order but must
        # satisfy the same contract (verify_schedule).
        pre = swept
        if scheduler is not None:
            from circls.compiler.scheduling import verify_schedule
            swept, sched_perm = scheduler(swept)
            verify_schedule(pre, swept, sched_perm)
        else:
            from circls.compiler.scheduling import schedule_ops
            swept, sched_perm = schedule_ops(swept, parallel=parallel_steps)
        # t_as_s guard (review 2026-08-20): the proxy rotation heads are
        # spliced AFTER reorder_weight1_last, so the scheduler may float
        # a commuting 's' head to behind a weight-1 terminal 'm' on the
        # same qubit — folded_out_bits would then reject a valid input.
        # Scheduling is an optimisation: on violation, keep the
        # pre-schedule order.
        def _floats_past_weight1(stream):
            for pos, op in enumerate(stream.ops):
                if op.kind == "m" and len(op.paulis) == 1:
                    (q, _l), = op.paulis.items()
                    if any(q in later.paulis
                           for later in stream.ops[pos + 1:]):
                        return True
            return False
        if s_head and _floats_past_weight1(swept):
            swept, sched_perm = pre, list(range(len(pre.ops)))
        m_ordinal = {}                      # pre-schedule op pos -> m ordinal
        k = 0
        for pos, op in enumerate(pre.ops):
            if op.kind == "m":
                m_ordinal[pos] = k
                k += 1
        m_load_trace = [pre_sched_m_load[m_ordinal[sched_perm[pos]]]
                        for pos, op in enumerate(swept.ops)
                        if op.kind == "m"]
    else:
        m_load_trace = pre_sched_m_load
    prog = expand_gadgets(swept)
    specs, steps, init, final, _ = to_experiment_inputs(
        prog, distance=distance, assignment=assignment,
        placement=placement, orientation=orientation)
    if birth_colouring and "colour_swapped" not in experiment_kwargs:
        from circls.core.colouring import plan_birth_colours
        experiment_kwargs["colour_swapped"] = plan_birth_colours(
            [s.interaction_type for s in steps])
    if lifetime is not None:
        experiment_kwargs["lifetime_overrides"] = lifetime
    if router is not None:
        experiment_kwargs["router"] = router
    exp = SequentialPPMExperiment(
        specs, steps, initial_states=init, final_measure_states=final,
        rounds=rounds or distance,
        rounds_init=experiment_kwargs.pop("rounds_init", 1),
        noise_params=noise,
        auto_rotate=auto_rotate, rotation_kind=rotation_kind,
        parallel_steps=parallel_steps,
        **experiment_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        circuit = exp.build()
    obs = append_program_observables(circuit, exp, prog)
    if graphlike_closures:
        circuit = strip_cross_window_closures(circuit)
    return CompiledProgram(circuit=circuit, observables=obs,
                           experiment=exp, program=prog,
                           reconstruction=recon, reorder_perm=perm,
                           m_load_trace=m_load_trace,
                           source_qasm=proxy_qasm if t_as_s else qasm,
                           t_as_s=t_as_s,
                           graphlike_closures=graphlike_closures)
