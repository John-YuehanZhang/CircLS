"""One-call Clifford QASM -> stim pipeline.

Promoted from ``notebooks/qasm_to_stim.ipynb`` (cell 1) so benchmarks and
experiments import it instead of redefining it.  Placement comes from
``circls.compiler.mapping`` (Square Sparse floor, row-major assignment).
"""
from __future__ import annotations

import contextlib
import warnings
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
    graphlike_detectors: bool = False        # True once graphlike_detectors_pass ran on `circuit` (compile_qasm)

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


def _index_detectors(circuit: stim.Circuit):
    """(flattened circuit, tick of every measurement record, [(position, records)] per DETECTOR).
    Records are counted with stim's own per-instruction count, so every measuring instruction
    (M/MX/MY/MZ, MR*, MPP, MXX/MYY/MZZ, MPAD) is covered."""
    flat = circuit.flattened()
    n, tick, rec_tick, dets = 0, 0, [], []       # dets: (position, records)
    for i, inst in enumerate(flat):
        if inst.name == "TICK":
            tick += 1
        elif inst.num_measurements:
            k = inst.num_measurements
            rec_tick.extend([tick] * k)
            n += k
        elif inst.name == "DETECTOR":
            dets.append((i, [n + t.value for t in inst.targets_copy()
                             if t.is_measurement_record_target]))
    return flat, rec_tick, dets


def _undecomposable(dem: stim.DetectorErrorModel):
    """Detector-id tuples of the error mechanisms stim could not split into pieces of <= 2 symptoms."""
    bad = set()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        piece, pieces = [], []
        pieces.append(piece)
        for t in inst.targets_copy():
            if t.is_separator():
                piece = []
                pieces.append(piece)
            elif t.is_relative_detector_id():
                piece.append(t.val)
        bad.update(tuple(sorted(pc)) for pc in pieces if len(pc) > 2)
    return sorted(bad)


def graphlike_detectors_pass(circuit: stim.Circuit,
                             verbose: bool = True,
                             probe_p: float = 1e-3) -> stim.Circuit:
    """Keep the detector error model graph-decodable: while stim reports an
    error mechanism it cannot decompose into pieces of at most two
    symptoms, re-emit the least local DETECTOR of that mechanism (the one
    whose records span the most ticks) as an OBSERVABLE_INCLUDE.

    A graph decoder (PyMatching) needs every fault to flip at most two
    detectors, after stim has split larger symptom sets by subtracting
    known two-symptom mechanisms.  Local checks satisfy this by
    construction.  A banked conservation relation -- a k-body PPM whose
    split leaves m >= 2 target patches alive conserves m - 1 products of
    the survivors' logicals, which the tracker emits at the readout that
    finally determines it -- puts whole logical strings into one row, so a
    fault on a shared record flips the relation together with the local
    checks and nothing in the model splits it off.  Such a relation is
    real logical information, so it is not dropped: it becomes an
    observable (a checked logical relation, as tqec records its cross-time
    correlation surfaces), which counts a flip as a logical failure but is
    not offered to the decoder as syndrome.

    Decision rule.  The pass first scans record ownership (linear in the
    number of detector records): a circuit in which no measurement record
    belongs to three or more detectors cannot contain such a relation and
    is returned untouched, without building an error model.  Otherwise
    the circuit is analysed under nominal uniform circuit-level noise
    (``probe_p``; the set of mechanisms and their symptoms does not depend
    on the rate) with stim's own decomposition, and for every mechanism
    that stays undecomposed the row with the widest tick span (then the
    most records, then the latest position) is converted; this repeats
    until stim decomposes everything.  Sharing a record three ways is thus
    only a trigger for the check, not a verdict: a relation stim can split
    off (because some two-symptom mechanism inside it exists) is kept, and
    a row is converted only when the decoder really cannot use it.  The
    rule reads only records, detectors and the error model, so it is
    protocol-agnostic; it is the identity on graph-decodable circuits, it
    never deletes a row, and conversions are printed -- never silently.
    (On qec_en_n5 under the paper configuration it converts exactly the two
    cross-window closures, 127 and 69 ticks wide, and keeps the split's
    own 2-tick closure; the paper's single-error audit then finds no
    miscorrected mechanism.)
    """
    flat, rec_tick, dets = _index_detectors(circuit)
    degree: Dict[int, int] = {}
    for _, recs in dets:
        for r in recs:
            degree[r] = degree.get(r, 0) + 1
    if not degree or max(degree.values()) <= 2:
        return circuit
    from circls.tools.evaluate import inject_uniform_noise     # evaluate imports this module
    span = {j: (max(rec_tick[r] for r in recs) - min(rec_tick[r] for r in recs)) if recs else 0
            for j, (_, recs) in enumerate(dets)}
    convert: set = set()                        # detector ordinals re-emitted as observables

    def rebuild() -> stim.Circuit:
        out = stim.Circuit()
        nobs = circuit.num_observables
        positions = {dets[j][0] for j in convert}
        for i, inst in enumerate(flat):
            if inst.name == "DETECTOR" and i in positions:
                out.append("OBSERVABLE_INCLUDE", inst.targets_copy(), [nobs])
                nobs += 1
            else:
                out.append(inst)
        return out

    out = circuit
    for _ in range(len(dets) + 1):
        dem = inject_uniform_noise(out, probe_p).detector_error_model(
            decompose_errors=True, ignore_decomposition_failures=True)
        bad = _undecomposable(dem)
        if not bad:
            break
        kept = [j for j in range(len(dets)) if j not in convert]   # DEM id -> ordinal
        for comp in bad:
            convert.add(max((kept[d] for d in comp),
                            key=lambda j: (span[j], len(dets[j][1]), j)))
        out = rebuild()
    else:                                        # every row converted and still not graphlike
        raise RuntimeError("graphlike_detectors_pass did not converge")
    if not convert:
        return circuit
    if verbose:
        converted = sorted((len(dets[j][1]), span[j]) for j in convert)
        print(f"graphlike_detectors: {len(converted)} detector(s) re-emitted "
              f"as observables (records, tick span): {converted}")
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
                 graphlike_detectors: bool = True,
                 graphlike_closures: Optional[bool] = None,
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
    ``graphlike_detectors`` (default ON): the graph-decodability post-pass
    (graphlike_detectors_pass): a banked cross-window logical relation
    that stim cannot split off the local checks' error mechanisms is
    emitted as an observable instead, so such relations never leave the
    DEM undecomposable; the identity on every graph-decodable circuit.
    The check is skipped (no error model built) whenever no measurement
    record belongs to three or more detectors, which is a filter for the
    shared-record mechanisms above, not a proof of decomposability: a
    hyperedge made of data-qubit faults alone would pass through, and the
    LER tooling then reports it.  ``graphlike_closures`` is the deprecated
    name of this switch.
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
    if graphlike_closures is not None:          # pre-2026-09-12 name of the switch
        warnings.warn("graphlike_closures= is now graphlike_detectors=",
                      DeprecationWarning, stacklevel=2)
        graphlike_detectors = graphlike_closures
    if graphlike_detectors:
        circuit = graphlike_detectors_pass(circuit)
    return CompiledProgram(circuit=circuit, observables=obs,
                           experiment=exp, program=prog,
                           reconstruction=recon, reorder_perm=perm,
                           m_load_trace=m_load_trace,
                           source_qasm=proxy_qasm if t_as_s else qasm,
                           t_as_s=t_as_s,
                           graphlike_detectors=graphlike_detectors)
