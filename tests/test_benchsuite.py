"""Benchmark suite generators + QASM normalizer: front-end oracles."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import (bbpssw_chain, bv, dj, ghz, graph_state_ring,  # noqa: E402
                        random_clifford, steane_encode,
                        teleport_chain, twisted_ghz)

from circls.interop.nwqec.frontend import load_clifford_mpauli  # noqa: E402
from circls.interop.ir.pauli_algebra import (drop_free_prefix,  # noqa: E402
                                          reorder_weight1_last, y_free_sweep)
from circls.interop.ir.gosc_gadgets import expand_gadgets  # noqa: E402
from circls.interop.nwqec.qasm_norm import normalize_qasm  # noqa: E402

pytestmark = pytest.mark.smoke


def _gadgets(qasm):
    ordered, _ = reorder_weight1_last(load_clifford_mpauli(qasm))
    return expand_gadgets(drop_free_prefix(y_free_sweep(ordered))).num_ancilla


@pytest.mark.parametrize("gen, n", [(ghz, 8), (bv, 8), (dj, 8),
                                    (graph_state_ring, 8)])
def test_family_generators_load_clifford(gen, n):
    pc = load_clifford_mpauli(gen(n))
    assert pc.num_qubits == n
    assert len(pc.ops) == n


def test_twisted_ghz_produces_consumables_and_observables():
    # alternate-link twists: gadgets AND deterministic parity bits coexist
    assert _gadgets(twisted_ghz(6)) >= 2
    assert _gadgets(ghz(6)) == 0            # named chains stay gadget-free


def test_teleport_chain_loads():
    pc = load_clifford_mpauli(teleport_chain(3))
    assert pc.num_qubits == 7


def test_steane_encode_stabilizer_oracle():
    # correct-by-construction check: the synthesized circuit's output state
    # must be stabilized by all six Steane generators plus logical Z-bar
    import stim
    qasm = steane_encode()
    body = [ln for ln in qasm.splitlines()
            if ln.startswith(("h ", "cx "))]
    circ = stim.Circuit()
    import re
    for ln in body:
        args = [int(a) for a in re.findall(r"q\[(\d+)\]", ln)]
        circ.append("H" if ln.startswith("h ") else "CX", args)
    sim = stim.TableauSimulator()
    sim.do_circuit(circ)
    gens = ["___XXXX", "_XX__XX", "X_X_X_X",
            "___ZZZZ", "_ZZ__ZZ", "Z_Z_Z_Z", "ZZZZZZZ"]
    for g in gens:
        assert sim.peek_observable_expectation(stim.PauliString(g)) == 1
    assert load_clifford_mpauli(qasm).num_qubits == 7


def test_bbpssw_chain_loads():
    pc = load_clifford_mpauli(bbpssw_chain(2))
    assert pc.num_qubits == 6


def test_random_clifford_reproducible_and_loads():
    a, b = random_clifford(5, 30, seed=7), random_clifford(5, 30, seed=7)
    assert a == b
    assert a != random_clifford(5, 30, seed=8)
    assert load_clifford_mpauli(a).num_qubits == 5


# ── qasm_norm unit oracles ────────────────────────────────────────────────────

_HDR = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def test_normalizer_rescues_partial_permuted_measure():
    raw = (f"{_HDR}qreg qr[3];\ncreg c[2];\nh qr[0];\ncx qr[0],qr[1];\n"
           "measure qr[1] -> c[0];\nmeasure qr[0] -> c[1];\n")
    with pytest.raises(Exception):
        load_clifford_mpauli(raw)           # partial + permuted + renamed
    fixed, notes = normalize_qasm(raw)
    pc = load_clifford_mpauli(fixed)
    assert pc.num_qubits == 3
    assert "renamed-register:qr->q" in notes
    assert "full-terminal-measurement" in notes


def test_normalizer_rejects_mid_circuit_measurement():
    # a non-Z-diagonal gate ON the measured qubit is a true mid-circuit
    # measurement — not rescuable
    raw = (f"{_HDR}qreg q[2];\ncreg c[2];\nh q[0];\n"
           "measure q[0] -> c[0];\nh q[0];\nmeasure q[1] -> c[1];\n")
    with pytest.raises(ValueError, match="mid-circuit"):
        normalize_qasm(raw)


def test_normalizer_defers_commuting_measures():
    # gates on OTHER qubits, or Z-diagonal positions on the measured one
    # (cx control, cz), commute with the measurement: exact deferral
    raw = (f"{_HDR}qreg q[2];\ncreg c[2];\nh q[0];\n"
           "measure q[0] -> c[0];\nh q[1];\ncx q[0], q[1];\n"
           "measure q[1] -> c[1];\n")
    text, notes = normalize_qasm(raw)
    assert "deferred-terminal-measurement" in notes
    assert text.rstrip().endswith("measure q -> c;")


def test_normalizer_flattens_multiple_qregs():
    raw = (f"{_HDR}qreg a[2];\nqreg b[1];\ncreg ans[1];\n"
           "h a[0];\ncx a[1], b[0];\nmeasure b[0] -> ans[0];\n")
    text, notes = normalize_qasm(raw)
    assert any(n.startswith("flattened-qregs:a[2]+b[1]") for n in notes)
    assert "qreg q[3];" in text and "cx q[1], q[2];" in text


def test_normalizer_rejects_reset():
    raw = f"{_HDR}qreg q[1];\nreset q[0];\nh q[0];\n"
    with pytest.raises(ValueError, match="reset"):
        normalize_qasm(raw)


def test_tclass_suite_roster():
    from benchsuite import tclass_suite, QASMBENCH_ROOT
    if not QASMBENCH_ROOT.exists():
        pytest.skip("QASMBench checkout not present")
    from circls.interop.nwqec.frontend import load_clifford_t_as_s
    cases = tclass_suite()
    assert len(cases) == 19
    core = [c for c in cases if c.tags == ("tclass",)]
    assert len(core) == 9
    # every case must pass the t_as_s front end as stored
    for c in cases:
        pc = load_clifford_t_as_s(c.qasm)
        assert sum(1 for op in pc.ops if op.kind == "s") >= 1
        assert c.n_qubits == pc.num_qubits


# ── adversarial-review regressions (2026-08-20) ─────────────────────────

def test_normalizer_comment_cannot_hide_statements():
    raw = (f"{_HDR}qreg q[2];\ncreg c[2];\nh q[0];\n"
           "measure q[0] -> c[0]; // read out\nx q[0];\ncx q[0], q[1];\n"
           "measure q[1] -> c[1];\n")
    with pytest.raises(ValueError, match="mid-circuit"):
        normalize_qasm(raw)


def test_normalizer_register_form_measure_then_gate_rejected():
    raw = (f"{_HDR}qreg q[2];\ncreg c[2];\nh q[0];\nmeasure q -> c;\n"
           "x q[0];\n")
    with pytest.raises(ValueError, match="mid-circuit|register-form"):
        normalize_qasm(raw)


def test_normalizer_rejects_register_form_on_q_named_register():
    raw = (f"{_HDR}qreg q[2];\nqreg a[2];\ncreg m[2];\nt q;\n"
           "cx q[0],a[0];\nmeasure q[0] -> m[0];\nmeasure q[1] -> m[1];\n")
    with pytest.raises(ValueError, match="register-form"):
        normalize_qasm(raw)
