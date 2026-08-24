"""Customization hooks (docs/API_HOOKS.md): result/strategy injection.

Slice 1: reselector=, scheduler=, placement=.  Contract: an injected
decision reproduces the built-in path bit-exactly when it returns the
built-in's answer, is rejected loudly when it violates the stage
contract, and conflicts with its shorthand flag are errors.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import ghz

from circls.compiler.measure_reduce import (Reconstruction,
                                           reduce_measurements)
from circls.pipeline import compile_qasm
from circls.compiler.scheduling import schedule_ops

QASM = ghz(8)


def _identity_recon(raw):
    m = len(raw.ops)
    return raw, Reconstruction(
        support=tuple(frozenset({k}) for k in range(m)),
        inverse=tuple(frozenset({k}) for k in range(m)),
        const=tuple(0 for _ in range(m)))


def test_reselector_builtin_matches_flag():
    via_flag = compile_qasm(QASM, distance=3)
    via_hook = compile_qasm(QASM, distance=3,
                            reselector=reduce_measurements)
    assert str(via_flag.circuit) == str(via_hook.circuit)


def test_reselector_identity_matches_reduction_off():
    off = compile_qasm(QASM, distance=3, measure_reduction=False)
    hook = compile_qasm(QASM, distance=3, reselector=_identity_recon)
    assert str(off.circuit) == str(hook.circuit)


def test_reselector_dropping_a_measurement_rejected():
    def bad(raw):
        red, recon = _identity_recon(raw)
        red = type(red)(num_qubits=red.num_qubits, ops=red.ops[:-1])
        return red, recon
    with pytest.raises(ValueError):
        compile_qasm(QASM, distance=3, reselector=bad)


def test_reselector_conflicts_with_flag():
    with pytest.raises(ValueError, match="not both"):
        compile_qasm(QASM, distance=3, measure_reduction=True,
                     reselector=_identity_recon)


def test_scheduler_builtin_matches_flag():
    via_flag = compile_qasm(QASM, distance=3)
    via_hook = compile_qasm(
        QASM, distance=3,
        scheduler=lambda c: schedule_ops(c, parallel=True))
    assert str(via_flag.circuit) == str(via_hook.circuit)


def test_scheduler_identity_permutation():
    hook = compile_qasm(
        QASM, distance=3,
        scheduler=lambda c: (c, list(range(len(c.ops)))))
    off = compile_qasm(QASM, distance=3, step_scheduling=False)
    assert str(hook.circuit) == str(off.circuit)


def test_scheduler_non_permutation_rejected():
    with pytest.raises(ValueError, match="permutation"):
        compile_qasm(QASM, distance=3,
                     scheduler=lambda c: (c, [0] * len(c.ops)))


def test_scheduler_conflicts_with_flag():
    with pytest.raises(ValueError, match="not both"):
        compile_qasm(QASM, distance=3, step_scheduling=True,
                     scheduler=lambda c: (c, list(range(len(c.ops)))))


def test_placement_row_major_cells_match_default():
    from circls.compiler.mapping import _natural_key, square_sparse_cells
    base = compile_qasm(QASM, distance=3)
    names = sorted({sp.name for sp in base.experiment.patches},
                   key=_natural_key)
    cells = dict(zip(names, square_sparse_cells(len(names))))
    injected = compile_qasm(QASM, distance=3, placement=cells)
    assert str(base.circuit) == str(injected.circuit)


def test_placement_duplicate_cell_rejected():
    names = [f"q{i}" for i in range(8)]
    cells = {nm: (0, 0) for nm in names}
    with pytest.raises(ValueError, match="one cell"):
        compile_qasm(QASM, distance=3, placement=cells)


def test_placement_missing_patch_rejected():
    with pytest.raises(ValueError, match="missing"):
        compile_qasm(QASM, distance=3, placement={"q0": (0, 0)})


def test_placement_conflicts_with_optimized():
    names = [f"q{i}" for i in range(8)]
    from circls.compiler.mapping import square_sparse_cells
    cells = dict(zip(names, square_sparse_cells(len(names))))
    with pytest.raises(ValueError, match="together"):
        compile_qasm(QASM, distance=3, assignment="optimized",
                     placement=cells)


def test_lifetime_override_widening_accepted():
    # measure_reduction=False keeps the 8 joint steps, so the derived
    # table is non-empty; widening every lifetime to the full sequence
    # is legal and simply retires nothing early
    base = compile_qasm(QASM, distance=3, measure_reduction=False)
    assert base.experiment.lifetimes, "test needs joint steps"
    n_layers = len(base.experiment.ppm_sequence)
    widened = {nm: (0, n_layers - 1)
               for nm in base.experiment.lifetimes}
    out = compile_qasm(QASM, distance=3, measure_reduction=False,
                       lifetime=widened)
    assert out.circuit is not None


def test_lifetime_override_narrowing_rejected():
    base = compile_qasm(QASM, distance=3, measure_reduction=False)
    items = [(nm, fl) for nm, fl in sorted(base.experiment.lifetimes.items())
             if fl[1] > fl[0]]
    assert items, "test needs a multi-layer lifetime"
    nm, (fu, lu) = items[0]
    with pytest.raises(ValueError, match="contain its uses"):
        compile_qasm(QASM, distance=3, measure_reduction=False,
                     lifetime={nm: (fu, lu - 1)})


def test_lifetime_override_unknown_patch_rejected():
    with pytest.raises(ValueError, match="unknown patch"):
        compile_qasm(QASM, distance=3, lifetime={"nope": (0, 0)})


def test_router_none_falls_back_to_search():
    calls = []

    def declining(specs, step):
        calls.append(step)
        return None
    base = compile_qasm(QASM, distance=3, measure_reduction=False)
    hook = compile_qasm(QASM, distance=3, measure_reduction=False,
                        router=declining)
    assert str(base.circuit) == str(hook.circuit)
    assert calls, "router was never consulted"


def test_router_replay_of_verified_corridors():
    # serve back the corridors the internal search verified: the compile
    # must succeed and the circuit must stay deterministic at p=0
    base = compile_qasm(QASM, distance=3, measure_reduction=False)
    verified = {tuple(sorted(s.interaction_type)): sorted(r.tree)
                for s, r in zip(base.experiment.ppm_sequence,
                                base.experiment._routes)
                if r is not None and getattr(r, "tree", None)}
    assert verified, "test needs at least one routed step"

    def replay(specs, step):
        return verified.get(tuple(sorted(step.interaction_type)))

    hook = compile_qasm(QASM, distance=3, measure_reduction=False,
                        router=replay)
    det, obs = hook.circuit.compile_detector_sampler(seed=0).sample(
        32, separate_observables=True)
    assert not det.any(), "replayed corridors broke p=0 silence"


def test_router_illegal_corridor_rejected():
    def on_top_of_patches(specs, step):
        return [(0, 0)]        # cell (0,0) hosts patch q0 in row-major
    with pytest.raises(Exception):
        compile_qasm(QASM, distance=3, measure_reduction=False,
                     router=on_top_of_patches)


def test_compile_ppm_sequence_entry():
    from circls.pipeline import compile_ppm_sequence
    base = compile_qasm(QASM, distance=3, measure_reduction=False,
                        step_scheduling=False)
    out = compile_ppm_sequence(base.program, distance=3)
    det, _ = out.circuit.compile_detector_sampler(seed=0).sample(
        32, separate_observables=True)
    assert not det.any(), "PPM-sequence entry broke p=0 silence"
    assert len(out.observables) == len(base.program.out_bits)


def test_router_once_per_step_and_direct_seams_skip_it():
    # contract (docs/API_HOOKS.md precedence table): the hook is asked at
    # most once per sequence step, and never for a cell-adjacent pair's
    # zero-cell direct seam
    calls = []

    def declining(specs, step):
        calls.append(step)
        return None
    hook = compile_qasm(QASM, distance=3, measure_reduction=False,
                        router=declining)
    exp = hook.experiment
    assert calls, "router was never consulted"
    assert len(calls) == len({id(s) for s in calls}), \
        "a step consulted the router more than once"
    consulted = {i for i, st in enumerate(exp.ppm_sequence)
                 if any(st is c for c in calls)}
    assert len(consulted) == len(calls), \
        "the router saw a step object that is not in ppm_sequence"
    assert not (consulted & exp._adj_steps), \
        "a direct-seam step consulted the router"


def test_lifetime_intermediate_init_layer_takes_effect():
    # the window is a schedule: an init layer strictly between 0 and the
    # first use allocates the patch at that layer, changing the circuit
    base = compile_qasm(QASM, distance=3, measure_reduction=False)
    nm, (fu, lu) = next((n, w) for n, w in sorted(
        base.experiment.lifetimes.items()) if w[0] >= 2)
    early = compile_qasm(QASM, distance=3, measure_reduction=False,
                         lifetime={nm: (fu - 1, lu)})
    assert str(early.circuit) != str(base.circuit)


def test_orientation_explicit_derived_matches_builtin():
    base = compile_qasm(QASM, distance=3, measure_reduction=False)
    derived = {s.name: s.orientation for s in base.experiment.patches}
    hook = compile_qasm(QASM, distance=3, measure_reduction=False,
                        orientation=derived)
    assert str(hook.circuit) == str(base.circuit)


def test_orientation_flip_changes_circuit():
    base = compile_qasm(QASM, distance=3, measure_reduction=False)
    specs = base.experiment.patches
    nm = sorted(s.name for s in specs)[0]
    cur = next(s.orientation for s in specs if s.name == nm)
    flip = "X_vertical" if cur == "X_horizontal" else "X_horizontal"
    hook = compile_qasm(QASM, distance=3, measure_reduction=False,
                        orientation={nm: flip})
    assert next(s.orientation for s in hook.experiment.patches
                if s.name == nm) == flip
    assert str(hook.circuit) != str(base.circuit)


def test_orientation_composes_with_placement():
    base = compile_qasm(QASM, distance=3, measure_reduction=False)
    derived = {s.name: s.orientation for s in base.experiment.patches}
    hook = compile_qasm(QASM, distance=3, measure_reduction=False,
                        placement={k: tuple(v) for k, v
                                   in base.placement.items()},
                        orientation=derived)
    assert str(hook.circuit) == str(base.circuit)


def test_orientation_rejected_unknown_bad_value_and_y():
    with pytest.raises(ValueError, match="unknown"):
        compile_qasm(QASM, distance=3, orientation={"nope": "X_vertical"})
    with pytest.raises(ValueError, match="values"):
        compile_qasm(QASM, distance=3, orientation={"q0": "vertical"})
    from benchsuite import twisted_ghz
    tg = twisted_ghz(4)
    y_base = compile_qasm(tg, distance=3)
    ynm = next(s.name for s in y_base.experiment.patches
               if s.name.startswith("y"))
    with pytest.raises(ValueError, match="Y"):
        compile_qasm(tg, distance=3, orientation={ynm: "X_horizontal"})


def test_analyze_qasm_matches_compiled_steps():
    # the front-end-only view must agree with what compile_qasm compiles
    from circls.pipeline import analyze_qasm
    info = analyze_qasm(QASM, measure_reduction=False)
    out = compile_qasm(QASM, distance=3, measure_reduction=False)
    assert info.interactions == [list(s.interaction_type)
                                 for s in out.experiment.ppm_sequence]
    assert set(info.patch_names) == set(out.placement)
    assert set(info.first_letters) == set(info.patch_names)
