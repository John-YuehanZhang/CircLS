"""The graph-decodability post-pass (circls.pipeline.graphlike_detectors_pass): while stim reports an error
mechanism it cannot decompose into pieces of at most two symptoms, the least local DETECTOR of that mechanism (the
widest tick span, then the most records, then the latest position) is re-emitted as an OBSERVABLE_INCLUDE.  A record
shared by three rows only triggers the check (no error model is built otherwise); the verdict is stim's own.  Identity
on graph-decodable circuits; never deletes a row."""
import pytest
import stim

import circls.tools.evaluate as E
from circls.pipeline import graphlike_detectors_pass

pytestmark = pytest.mark.smoke


def _records(circ):
    """(number of detectors, number of observables, list of record sets of the OBSERVABLE_INCLUDEs)."""
    n, obs = 0, {}
    for inst in circ.flattened():
        if inst.num_measurements:
            n += inst.num_measurements
        elif inst.name == "OBSERVABLE_INCLUDE":
            j = int(inst.gate_args_copy()[0])
            obs.setdefault(j, set())
            for t in inst.targets_copy():
                obs[j] ^= {n + t.value}
    return circ.num_detectors, circ.num_observables, obs


def _kept(circ):
    return sorted(int(i.gate_args_copy()[0]) for i in circ.flattened() if i.name == "DETECTOR")


def _toy(order, dets):
    """Four measurements, one per tick (record r at tick r); ``dets`` maps a one-letter name to its records and
    ``order`` is the emission order, so a decision must not depend on where a row happens to be emitted."""
    c = stim.Circuit()
    for q in range(4):
        c.append("M", [q])
        if q < 3:
            c.append("TICK")
    for name in order:
        c.append("DETECTOR", [stim.target_rec(r - 4) for r in dets[name]], [ord(name)])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [0])
    return c


# record 0 is in A, B and C, and every other record pairs one of them with D: a flip of record 0 has symptoms
# {A, B, C} and no two-symptom mechanism inside that set exists for stim to subtract, so the model is not
# graphlike.  Spans: A 1, B 2, C 3, D 2 (D has the most records but is not part of the failing mechanism).
UNDECOMPOSABLE = {"A": [0, 1], "B": [0, 2], "C": [0, 3], "D": [1, 2, 3]}


@pytest.mark.parametrize("order", ["ABCD", "CABD", "DCBA"])
def test_widest_row_of_the_failing_mechanism_becomes_observable(order):
    c = _toy(order, UNDECOMPOSABLE)
    with pytest.raises(ValueError):
        E.inject_uniform_noise(c, 1e-3).detector_error_model(decompose_errors=True)
    out = graphlike_detectors_pass(c, verbose=False)
    nd, no, obs = _records(out)
    assert (nd, no) == (3, 2)                       # one row converted, nothing deleted
    assert _kept(out) == sorted(map(ord, "ABD"))    # C (span 3) goes; D (3 records, span 2) stays
    assert obs[1] == {0, 3}                         # the converted row keeps its records
    E.inject_uniform_noise(out, 1e-3).detector_error_model(decompose_errors=True)   # graphlike now


def test_every_measuring_instruction_kind_is_counted():
    """Records from MZZ, MPP and MYY precede the plain M records of the UNDECOMPOSABLE layout (as a
    fifth row E of degree-1 records).  Miscounting any of them shifts every later record index, so the pass
    would inspect the wrong rows; with stim's per-instruction count the verdict is unchanged: C goes, E stays."""
    c = stim.Circuit()
    c.append("RX", [12, 13])                        # deterministic outcomes for the three exotic rows
    c.append("RY", [14, 15])
    c.append("MZZ", [10, 11])                       # record 0
    c.append("MPP", [stim.target_x(12), stim.target_combiner(), stim.target_x(13)])   # record 1
    c.append("MYY", [14, 15])                       # record 2
    c.append("TICK")
    for q in range(4):                              # records 3..6 at ticks 1..4
        c.append("M", [q])
        c.append("TICK")
    rows = {"A": [3, 4], "B": [3, 5], "C": [3, 6], "D": [4, 5, 6], "E": [0, 1, 2]}
    for name, recs in rows.items():
        c.append("DETECTOR", [stim.target_rec(r - 7) for r in recs], [ord(name)])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [0])
    out = graphlike_detectors_pass(c, verbose=False)
    nd, no, obs = _records(out)
    assert (nd, no) == (4, 2) and _kept(out) == sorted(map(ord, "ABDE"))
    assert obs[1] == {3, 6}


def test_three_way_record_alone_is_not_a_verdict():
    """A = {0, 1}, B = {1, 2}, G = {0, 1, 2, 3, 4}: record 1 is in three rows, but stim splits the flip of
    record 1 as {A, G} + {B} (record 0 gives the {A, G} mechanism), so nothing is converted."""
    c = stim.Circuit()
    c.append("M", [0, 1, 2, 3, 4])
    for name, recs in (("A", (-5, -4)), ("B", (-4, -3)), ("G", (-5, -4, -3, -2, -1))):
        c.append("DETECTOR", [stim.target_rec(k) for k in recs], [ord(name)])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [0])
    E.inject_uniform_noise(c, 1e-3).detector_error_model(decompose_errors=True)
    out = graphlike_detectors_pass(c, verbose=False)
    assert str(out) == str(c.flattened())


def test_no_error_model_when_every_record_has_degree_at_most_two(monkeypatch):
    c = _toy("AB", {"A": [0, 1], "B": [1, 2]})     # no third owner anywhere
    def boom(*a, **k):
        raise AssertionError("error model built for a circuit with no shared record")
    monkeypatch.setattr(E, "inject_uniform_noise", boom)
    out = graphlike_detectors_pass(c, verbose=False)
    assert str(out) == str(c.flattened())


QEC_EN_N5 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
h q[2]; t q[2]; h q[2]; h q[0]; h q[1]; h q[2];
cx q[1], q[2]; cx q[0], q[2]; h q[0]; h q[1]; h q[3];
cx q[3], q[2]; h q[2]; h q[3]; cx q[3], q[2]; cx q[0], q[2]; cx q[1], q[2];
h q[2]; h q[4]; cx q[4], q[2]; h q[2]; h q[4]; cx q[4], q[2]; cx q[1], q[2]; cx q[3], q[2];
measure q -> c;
"""     # QASMBench qec_en_n5 as TopoLS transpiles it (the paper's Table 2 input), all five bits read out


# The layout on which the two cross-window closures were characterised: qec_en_n5 under the paper configuration as
# the optimizer placed it before its spectral start got a deterministic tie-break (2026-09-12).  Which relations the
# tracker banks, and how many of them stim cannot split off, is a property of the layout (the tie-break moves qec_en
# to a layout with three such closures), and this test is about the pass, not the optimizer, so the layout is pinned
# (placement= replaces the assignment search; the data patches' orientation follows from it, the |Y> patch's is
# protocol-fixed).
QEC_EN_PAPER_PLACEMENT = {"q0": (3, 3), "q1": (1, 3), "q2": (5, 3), "q3": (3, 1), "q4": (1, 1), "y0": (5, 1)}


def test_qec_en_paper_configuration_is_graph_decodable(monkeypatch):
    """The one Table 2 program whose paper-configuration compile banks cross-window relations: qec_en_n5 with
    measurement re-selection, step scheduling and parallel windows off, under the X-state proxy (gadget ancillas
    prepared in |+> instead of |i>, the Table 2 convention; with |i> ancillas the relations close within a window).
    On the pinned layout two of its banked closures (tick spans 127 and 69) share records with the local checks and
    stim cannot split them off; the pass re-emits exactly those two -- not the split's own 2-tick closure, which
    stays a detector -- and stim can then decompose the DEM."""
    pytest.importorskip("nwqec")
    import contextlib, io
    import circls.pipeline as P
    from circls.tools.evaluate import inject_uniform_noise
    orig = P.to_experiment_inputs
    def x_proxy(*a, **k):
        specs, steps, init, final, prog = orig(*a, **k)
        return specs, steps, {nm: ("X" if v == "Y" else v) for nm, v in init.items()}, final, prog
    monkeypatch.setattr(P, "to_experiment_inputs", x_proxy)
    kw = dict(placement=dict(QEC_EN_PAPER_PLACEMENT), measure_reduction=False, first_use_init=True,
              step_scheduling=False, parallel_steps=False, liveness=True, keep_patches=[])
    with contextlib.redirect_stdout(io.StringIO()):
        raw = P.compile_qasm(QEC_EN_N5, distance=3, t_as_s=True, graphlike_detectors=False, **kw)
        cp = P.compile_qasm(QEC_EN_N5, distance=3, t_as_s=True, **kw)
    with pytest.raises(ValueError):                 # without the pass: hyperedges, PyMatching cannot be built
        inject_uniform_noise(raw.circuit, 5e-4).detector_error_model(decompose_errors=True)
    assert cp.placement == QEC_EN_PAPER_PLACEMENT and raw.placement == QEC_EN_PAPER_PLACEMENT
    assert cp.circuit.num_detectors == raw.circuit.num_detectors - 2
    assert cp.circuit.num_observables == raw.circuit.num_observables + 2
    inject_uniform_noise(cp.circuit, 5e-4).detector_error_model(decompose_errors=True)   # graphlike now
    det, obs = cp.circuit.compile_detector_sampler(seed=0).sample(64, separate_observables=True)
    assert not det.any() and (obs == obs[0]).all()  # still deterministic at p = 0
    # the converted rows are the two cross-window relations, not the split's own closure
    spans = []
    n, tick, rec_tick = 0, 0, []
    for inst in cp.circuit.flattened():
        if inst.name == "TICK":
            tick += 1
        elif inst.num_measurements:
            rec_tick += [tick] * inst.num_measurements; n += inst.num_measurements
        elif inst.name == "OBSERVABLE_INCLUDE" and int(inst.gate_args_copy()[0]) >= raw.circuit.num_observables:
            ts = [rec_tick[n + t.value] for t in inst.targets_copy()]
            spans.append(max(ts) - min(ts))
    assert sorted(spans) == [69, 127]
