"""GoSC-transcribed phase algebra + Y elimination, cross-validated against
independent oracles (stim PauliString products, exact numpy matrix identities,
stim inverse-tableau conjugation for the nwqec m_pauli reader).

The Fig. 10 test is the acceptance criterion requested by the user: the
(Y x 1 x Y x Z x Y x Y)_pi/8 rotation must decompose into the exact masks the
figure draws (Z_1 and Z_3 Z_5 Z_6, 1-indexed) with the honest transformed sign
(MINUS — the figure omits signs; LSC's yfree gets exactly this wrong).
"""
import random

import numpy as np
import pytest
import stim

from circls.interop.ir.pauli_algebra import (
    PhasedPauli, commute_pi4_past, eliminate_y, zmask,
)
from circls.interop.ir.pauli_ir import PauliOp

# ---------------------------------------------------------------- numpy oracle
_M2 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_PHASE = {0: 1, 1: 1j, 2: -1, 3: -1j}


def mat(op: PhasedPauli, n: int) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for q in range(n):
        out = np.kron(out, _M2[op.paulis.get(q, "I")])
    return _PHASE[op.phase_exp] * out


def rot(op: PhasedPauli, phi: float, n: int) -> np.ndarray:
    """P_phi = exp(-i*phi*P) (GoSC convention: S = Z_pi/4, T = Z_pi/8)."""
    m = mat(op, n)
    return np.cos(phi) * np.eye(2 ** n) - 1j * np.sin(phi) * m


def _random_phased(rng, n) -> PhasedPauli:
    paulis = {q: rng.choice("XYZ") for q in range(n) if rng.random() < 0.7}
    return PhasedPauli(rng.randrange(4), paulis)


def _to_stim(op: PhasedPauli, n: int) -> stim.PauliString:
    prefix = {0: "+", 1: "i", 2: "-", 3: "-i"}[op.phase_exp]
    return stim.PauliString(
        prefix + "".join(op.paulis.get(q, "_") for q in range(n)))


# ------------------------------------------------------------------- products
def test_mul_matches_stim():
    rng = random.Random(7)
    n = 5
    for _ in range(300):
        a, b = _random_phased(rng, n), _random_phased(rng, n)
        ours = a.mul(b)
        theirs = _to_stim(a, n) * _to_stim(b, n)
        assert _to_stim(ours, n) == theirs, f"{a} * {b}: {ours} != {theirs}"


def test_commutes_matches_stim():
    rng = random.Random(11)
    n = 5
    for _ in range(200):
        a, b = _random_phased(rng, n), _random_phased(rng, n)
        assert a.commutes_with(b) == _to_stim(a, n).commutes(_to_stim(b, n))


# ---------------------------------------------------- RULE 1: pi/4 commutation
def test_commute_pi4_past_matrix_identity():
    """Circuit [(sA)_pi/4, P_phi] == [P'_phi, (sA)_pi/4] exactly (as matrices,
    rightmost-acts-first: R(P,phi)R(sA,pi/4) == R(sA,pi/4)R(P',phi))."""
    rng = random.Random(23)
    n = 3
    for _ in range(60):
        mask = PhasedPauli(0, {q: rng.choice("XYZ") for q in range(n)
                               if rng.random() < 0.8})
        op = PhasedPauli(0 if rng.random() < 0.5 else 2,
                         {q: rng.choice("XYZ") for q in range(n)
                          if rng.random() < 0.8})
        sign = rng.choice([1, -1])
        phi = rng.uniform(0, 2 * np.pi)
        out = commute_pi4_past(mask, sign, op)
        smask = PhasedPauli(0 if sign > 0 else 2, dict(mask.paulis))
        lhs = rot(op, phi, n) @ rot(smask, np.pi / 4, n)
        rhs = rot(smask, np.pi / 4, n) @ rot(out, phi, n)
        assert np.allclose(lhs, rhs, atol=1e-12)
        if mask.commutes_with(op):
            assert out is op    # "can simply be moved past"


# ------------------------------------------- RULE 3: Y elimination (Fig. 10)
def test_gosc_identity_single_y():
    """The paper's own pinned identity: Y_pi/4 = Z_pi/4 X_pi/4 Z_-pi/4
    (matrix order), i.e. eliminate_y(Y) -> mask Z, op' = +X."""
    masks, out = eliminate_y(PhasedPauli(0, {0: "Y"}))
    assert [m.paulis for m in masks] == [{0: "Z"}]
    assert out.paulis == {0: "X"} and out.to_sign() == 1
    z, x, y = (PhasedPauli(0, {0: p}) for p in "ZXY")
    lhs = rot(y, np.pi / 4, 1)
    rhs = rot(z, np.pi / 4, 1) @ rot(x, np.pi / 4, 1) @ rot(z, -np.pi / 4, 1)
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_fig10_masks_and_sign():
    """(Y x 1 x Y x Z x Y x Y): masks exactly {Z_1} and {Z_3 Z_5 Z_6}
    (1-indexed as in the figure), transformed operator -(X 1 X Z X X)."""
    p = PhasedPauli(0, {0: "Y", 2: "Y", 3: "Z", 4: "Y", 5: "Y"})
    masks, out = eliminate_y(p)
    assert [m.paulis for m in masks] == [{0: "Z"}, {2: "Z", 4: "Z", 5: "Z"}]
    assert out.paulis == {0: "X", 2: "X", 3: "Z", 4: "X", 5: "X"}
    assert out.to_sign() == -1, (
        "honest sign is MINUS; the figure draws no sign and LSC's yfree "
        "returns +1 here (its known bug)")


@pytest.mark.parametrize("phi", [np.pi / 8, np.pi / 4, 0.371])
def test_fig10_matrix_identity(phi):
    """Exact 64x64 identity for the full sandwich, any angle:
    circuit [A1_-pi/4, A2_-pi/4, op'_phi, A2_pi/4, A1_pi/4] == [P_phi]."""
    n = 6
    p = PhasedPauli(0, {0: "Y", 2: "Y", 3: "Z", 4: "Y", 5: "Y"})
    masks, out = eliminate_y(p)
    a1, a2 = masks
    m = (rot(a1, np.pi / 4, n) @ rot(a2, np.pi / 4, n) @ rot(out, phi, n)
         @ rot(a2, -np.pi / 4, n) @ rot(a1, -np.pi / 4, n))
    assert np.allclose(m, rot(p, phi, n), atol=1e-12)


def test_odd_y_matrix_identity():
    """Odd number of Y's: a single mask suffices (GoSC Sec. 2.1)."""
    n = 4
    p = PhasedPauli(0, {0: "Y", 1: "X", 2: "Y", 3: "Y"})
    masks, out = eliminate_y(p)
    assert len(masks) == 1 and masks[0].paulis == {0: "Z", 2: "Z", 3: "Z"}
    assert not out.y_qubits()
    phi = np.pi / 8
    m = (rot(masks[0], np.pi / 4, n) @ rot(out, phi, n)
         @ rot(masks[0], -np.pi / 4, n))
    assert np.allclose(m, rot(p, phi, n), atol=1e-12)


def test_eliminate_y_random_matrix_identity():
    rng = random.Random(41)
    n = 4
    for _ in range(40):
        paulis = {q: rng.choice("XYZ") for q in range(n) if rng.random() < 0.8}
        if not paulis:
            continue
        p = PhasedPauli(0 if rng.random() < 0.5 else 2, paulis)
        masks, out = eliminate_y(p)
        assert not out.y_qubits()
        phi = rng.uniform(0, 2 * np.pi)
        m = rot(out, phi, n)
        for a in reversed(masks):
            m = rot(a, np.pi / 4, n) @ m @ rot(a, -np.pi / 4, n)
        assert np.allclose(m, rot(p, phi, n), atol=1e-12)
        if not p.y_qubits():
            assert masks == [] and out is p


def test_pauli_op_bridge():
    op = PauliOp("m", {0: "Y", 1: "Z"}, -1)
    pp = PhasedPauli.from_pauli_op(op)
    assert pp.phase_exp == 2 and pp.paulis == op.paulis
    assert pp.to_pauli_op("m") == op
    with pytest.raises(ValueError):
        PhasedPauli(1, {0: "X"}).to_sign()


# ------------------------------------------------------- m_pauli reader (nwqec)
_GATE_QASM = {"h": "h q[{}];", "s": "s q[{}];", "sdg": "sdg q[{}];",
              "x": "x q[{}];", "z": "z q[{}];"}
_GATE2_QASM = {"cx": "cx q[{}],q[{}];", "cz": "cz q[{}],q[{}];",
               "swap": "swap q[{}],q[{}];"}
_GATE_STIM = {"h": "H", "s": "S", "sdg": "S_DAG", "x": "X", "z": "Z",
              "cx": "CX", "cz": "CZ", "swap": "SWAP"}


def _random_clifford(rng, n, depth):
    qasm = [f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];']
    circ = stim.Circuit()
    for _ in range(depth):
        if rng.random() < 0.5:
            g = rng.choice(list(_GATE_QASM))
            q = rng.randrange(n)
            qasm.append(_GATE_QASM[g].format(q))
            circ.append(_GATE_STIM[g], [q])
        else:
            g = rng.choice(list(_GATE2_QASM))
            a, b = rng.sample(range(n), 2)
            qasm.append(_GATE2_QASM[g].format(a, b))
            circ.append(_GATE_STIM[g], [a, b])
    return "\n".join(qasm) + "\n", circ


def test_reader_probe_example():
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    c = load_clifford_mpauli(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
        "x q[0];\nh q[1];\ns q[1];\n")
    assert c.num_qubits == 2
    assert [(op.kind, op.paulis, op.sign) for op in c.ops] == [
        ("m", {0: "Z"}, -1),       # X0 conj: measure -Z0
        ("m", {1: "X"}, 1),        # C^dag Z1 C = X1  (NOT the forward +Y1)
    ]


def test_reader_matches_stim_inverse_tableau():
    """m_pauli_i must equal C^dagger Z_i C — stim inverse-tableau oracle,
    letters AND sign, random Clifford circuits."""
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    rng = random.Random(97)
    for _ in range(12):
        n = rng.choice([2, 3, 4])
        qasm, circ = _random_clifford(rng, n, 25)
        got = load_clifford_mpauli(qasm)
        tinv = circ.to_tableau().inverse()
        assert len(got.ops) == n
        for i, op in enumerate(got.ops):
            z = stim.PauliString(n)
            z[i] = 3
            want = tinv(z)
            assert op.kind == "m"
            assert op.paulis == {q: "_XYZ"[want[q]] for q in range(n)
                                 if want[q]}, f"qubit {i}"
            assert want.sign in (1, -1)
            assert op.sign == int(want.sign.real), f"sign of qubit {i}"


def test_nwqec_mpauli_sign_bug_documented():
    """UPSTREAM BUG PIN (nwqec pbc_pass.hpp): reverse sweep + forward-gate
    conjugation flips m_pauli signs on S-family gates.  Minimal case
    ``s q0; h q0``: true C^dag Z C = S^dag X S = -Y, nwqec emits +Y.

    If this test FAILS, nwqec fixed the bug — remove the stim-sign workaround
    in load_clifford_mpauli and take signs from nwqec again."""
    nwqec = pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import _load_qasm_str, from_nwqec
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ns q[0];\nh q[0];\n'
    raw = from_nwqec(nwqec.to_pbc(_load_qasm_str(nwqec, qasm), keep_cx=False,
                                  optimize_t_count=False))
    assert [(op.paulis, op.sign) for op in raw.ops] == [({0: "Y"}, 1)], \
        "nwqec now returns something else here — did upstream fix the sign bug?"
    # our reader returns the truth
    fixed = load_clifford_mpauli(qasm)
    assert [(op.paulis, op.sign) for op in fixed.ops] == [({0: "Y"}, -1)]


def test_reader_rejects_non_clifford():
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    with pytest.raises(ValueError, match="Clifford-only"):
        load_clifford_mpauli(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nt q[0];\n')


def test_reader_rejects_mid_circuit_measure_and_reset():
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    head = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    with pytest.raises(ValueError, match="after a 'measure'"):
        load_clifford_mpauli(head + "measure q[0] -> c[0];\nh q[1];\n")
    with pytest.raises(ValueError, match="reset"):
        load_clifford_mpauli(head + "h q[0];\nreset q[0];\nh q[0];\n")
    # terminal measurement of all qubits is the assumed semantic: accepted
    c = load_clifford_mpauli(
        head + "h q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    assert len(c.ops) == 2


# ------------------------------------------- C3/M6/M7: comment & operand parsing
_HEAD1 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
_HEAD2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'


def test_comment_before_gate_does_not_eat_it():
    """C3 repro: '// flip it' used to delete the following x gate from the
    sign-authoritative stim tableau -> m0 = +X instead of the true -X, with
    the letter cross-check blind to it."""
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    c = load_clifford_mpauli(_HEAD1 + "h q[0];\n// flip it\nx q[0];\n")
    assert [(op.paulis, op.sign) for op in c.ops] == [({0: "X"}, -1)]


def test_semicolon_inside_comment_is_not_a_phantom_gate():
    """C3 repro 2: a ';' inside a comment used to resurrect the commented-out
    text after it as a REAL gate."""
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    c = load_clifford_mpauli(_HEAD1 + "h q[0]; // disabled; x q[0]\n")
    assert [(op.paulis, op.sign) for op in c.ops] == [({0: "X"}, 1)]


def test_comment_before_reset_still_rejected():
    """M6 repro: '// prep' used to swallow the following reset from the
    faithfulness guard's view."""
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    with pytest.raises(ValueError, match="reset"):
        load_clifford_mpauli(
            _HEAD2 + "h q[0]; // prep\nreset q[0];\nh q[0];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")


def test_measure_operands_enforced():
    """M7: permuted / duplicate / partial terminal measurements are silently
    reinterpreted by nwqec as 'measure all in order' — must fail loudly."""
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    body = "h q[0];\ncx q[0],q[1];\n"
    with pytest.raises(ValueError, match="non-identity"):
        load_clifford_mpauli(_HEAD2 + body +
                             "measure q[0] -> c[1];\nmeasure q[1] -> c[0];\n")
    with pytest.raises(ValueError, match="twice"):
        load_clifford_mpauli(_HEAD2 + body +
                             "measure q[0] -> c[0];\nmeasure q[0] -> c[0];\n")
    with pytest.raises(ValueError, match="partial"):
        load_clifford_mpauli(_HEAD2 + body + "measure q[0] -> c[0];\n")
    # register form and identity indexed form are both fine
    ok1 = load_clifford_mpauli(_HEAD2 + body + "measure q -> c;\n")
    ok2 = load_clifford_mpauli(_HEAD2 + body +
                               "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    assert [(o.paulis, o.sign) for o in ok1.ops] == \
           [(o.paulis, o.sign) for o in ok2.ops]


def test_no_tempfile_leak_on_string_load(tmp_path, monkeypatch):
    """m2: string loads must not leave .qasm files behind in tempdir.
    Runs against a PRIVATE tempdir: counting the shared system tempdir
    races against other xdist workers' legitimate temp files."""
    pytest.importorskip("nwqec")
    import glob, os, tempfile
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)  # re-resolve from env
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    for _ in range(3):
        load_clifford_mpauli(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n')
    leaked = glob.glob(os.path.join(str(tmp_path), "*.qasm"))
    assert leaked == []
