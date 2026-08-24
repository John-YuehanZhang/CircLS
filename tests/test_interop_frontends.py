"""Interop front-ends (nwqec_frontend, yfree) + the isolation contract.

The external tools are OPTIONAL: nwqec tests importorskip 'nwqec'; yfree tests
skip unless the lattice-surgery-compiler source is reachable.  The isolation
test asserts that importing LightStim (or circls.interop) never pulls either
third-party package in.
"""
import subprocess
import sys

import pytest

from circls.interop import PauliCircuit, PauliOp


def _lsqecc_available() -> bool:
    try:
        from circls.interop.ir.yfree import _import_lsqecc
        _import_lsqecc()
        return True
    except ImportError:
        return False


needs_lsqecc = pytest.mark.skipif(not _lsqecc_available(), reason="lsqecc source not reachable")


# ----------------------------------------------------------------- isolation
def test_core_import_does_not_pull_external_deps():
    # A fresh interpreter importing LightStim + interop must not import nwqec/lsqecc.
    code = (
        "import lightstim, circls.interop, sys;"
        "assert 'nwqec' not in sys.modules, 'lightstim import pulled in nwqec';"
        "assert 'lsqecc' not in sys.modules, 'lightstim import pulled in lsqecc';"
        "print('ok')"
    )
    # the fresh interpreter inherits neither conftest's sys.path bootstrap
    # nor the repo cwd — hand it both roots explicitly
    import os
    from pathlib import Path
    _root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_root), str(_root.parent / "LightStim"),
         env.get("PYTHONPATH", "")])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# ----------------------------------------------------------------- yfree
@needs_lsqecc
def test_yfree_removes_y_and_keeps_mixed_structure():
    from circls.interop.ir.yfree import to_y_free
    # Rot about X0 Y1 Z2  ->  Zrot(pi/4) . Rot(X0 X1 Z2) . Zrot(-pi/4), all X/Z.
    circ = PauliCircuit(3, [PauliOp("t", {0: "X", 1: "Y", 2: "Z"})])
    yf = to_y_free(circ)
    assert not yf.has_y()
    # the main T rotation survives with Y->X, still a genuine mixed X/Z joint
    t_ops = [op for op in yf.ops if op.kind == "t"]
    assert len(t_ops) == 1
    assert t_ops[0].paulis == {0: "X", 1: "X", 2: "Z"}
    assert t_ops[0].is_mixed()
    # flanked by pure-Z compensating rotations
    assert all(set(op.paulis.values()) <= {"Z"} for op in yf.ops if op.kind != "t")


@needs_lsqecc
def test_yfree_noop_on_xz_circuit():
    from circls.interop.ir.yfree import to_y_free
    circ = PauliCircuit(2, [PauliOp("t", {0: "X", 1: "Z"}), PauliOp("m", {0: "X"})])
    yf = to_y_free(circ)
    assert not yf.has_y()
    assert [op.paulis for op in yf.ops] == [{0: "X", 1: "Z"}, {0: "X"}]


# ----------------------------------------------------------------- nwqec
def test_nwqec_parse_and_pipeline():
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_pbc

    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
        "h q[0];\ncx q[0],q[1];\nt q[1];\ncx q[1],q[2];\nt q[2];\n"
    )
    pc = load_pbc(qasm, keep_cx=False)
    assert pc.num_qubits == 3
    assert pc.counts().get("t", 0) >= 1
    # parsed Pauli strings are well-formed
    assert all(set(op.paulis.values()) <= {"X", "Y", "Z"} for op in pc.ops)


@needs_lsqecc
def test_yfree_preserves_unitary():
    # to_y_free is a logical identity: the X/Z-only rotation sequence computes the
    # SAME unitary as the original Y-containing one (lsqecc's rotation convention is
    # exp(+i * pi * frac * P)).  Checked up to global phase, incl. an even-Y op.
    import numpy as np
    from fractions import Fraction
    from circls.interop.ir.yfree import to_y_free

    PM = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], complex),
          "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]).astype(complex)}
    FR = {"t": Fraction(1, 8), "s": Fraction(1, 4), "z": Fraction(1, 2)}

    def unitary(ops, n):
        U = np.eye(2 ** n, dtype=complex)
        for op in ops:
            if op.kind == "m":
                continue
            P = np.array([[1]], complex)
            for q in range(n):
                P = np.kron(P, PM[op.paulis.get(q, "I")])
            a = op.sign * float(FR[op.kind]) * np.pi
            U = (np.cos(a) * np.eye(2 ** n) + 1j * np.sin(a) * P) @ U
        return U

    n = 3
    circ = PauliCircuit(n, [
        PauliOp("t", {0: "X", 1: "Y", 2: "Z"}),
        PauliOp("s", {0: "Y", 1: "Z"}),          # even total Y across ops exercises the extra-pair branch
        PauliOp("z", {1: "Y"}),
    ])
    yf = to_y_free(circ)
    assert not yf.has_y()
    U0, U1 = unitary(circ.ops, n), unitary(yf.ops, n)
    overlap = abs(np.trace(U0.conj().T @ U1)) / 2 ** n
    assert abs(overlap - 1.0) < 1e-9      # equal up to global phase


@needs_lsqecc
def test_nwqec_to_yfree_to_check_end_to_end():
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_pbc
    from circls.interop.ir.yfree import to_y_free

    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
        "h q[0];\ncx q[0],q[1];\ns q[1];\nt q[1];\ncx q[1],q[0];\nt q[0];\n"
    )
    yf = to_y_free(load_pbc(qasm, keep_cx=True))
    assert not yf.has_y()


def test_normalize_qasm_drops_by_statement_not_by_line():
    """Regression (fatal silent-miscompilation): measure/creg/barrier are
    dropped per-statement, not by whole line — a gate sharing a line with a
    dropped keyword must survive (over-drop), and a keyword not at line-start
    must still be dropped (under-drop)."""
    from circls.interop.nwqec.qasm_norm import normalize_qasm, _statements
    # over-drop: 'barrier q; x q[0];' on one line -- the x must survive
    text, _ = normalize_qasm(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[1];\n'
        'barrier q; x q[0];\nmeasure q[0] -> c[0];\n')
    stmts = [s.strip() for s in _statements(text)]
    assert "x q[0]" in stmts
    assert not any(s.lower().startswith("barrier") for s in stmts)
    assert not any(s.lower().startswith("measure q[0]") for s in stmts)
    assert "measure q -> c" in stmts
    # under-drop: a measure not at line-start must still be dropped
    text2, _ = normalize_qasm(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[1];\n'
        'x q[0]; measure q[0] -> c[0];\n')
    stmts2 = [s.strip() for s in _statements(text2)]
    assert "x q[0]" in stmts2
    assert not any(s.lower().startswith("measure q[0]") for s in stmts2)


def test_normalize_qasm_ignores_commented_qreg():
    """Regression (fatal silent-miscompilation): a commented-out '// qreg'
    must not be counted in the register census, else it triggers a spurious
    multi-register flatten that shifts operands and inflates the qubit count."""
    from circls.interop.nwqec.qasm_norm import normalize_qasm, _statements
    text, notes = normalize_qasm(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n// qreg anc[2];\n'
        'qreg q[3];\nx q[0];\nmeasure q[0] -> c[0];\n')
    stmts = [s.strip() for s in _statements(text)]
    assert "qreg q[3]" in stmts
    assert "x q[0]" in stmts
    assert not any("flatten" in n for n in notes)


def test_normalize_qasm_refuses_gate_definition_blocks():
    # A gate body's braces are not ';'-statement-aligned: the statement
    # after the closing '}' arrives glued to it ("}\nqreg cin[1]"), so the
    # head-keyed passes (qreg drop, reset refusal, measure census) all
    # miss it — the flatten emitted a bogus leftover qreg (measured on
    # QASMBench adder_n10/bigadder_n18), a glued measure survived
    # alongside the appended full measurement, and a glued reset slipped
    # past the not-rescuable refusal.  The Clifford front-end rejects
    # every `gate` statement anyway, so the normalizer must refuse these
    # files loudly instead of scanning them wrong.
    from circls.interop.nwqec.qasm_norm import normalize_qasm

    gate_then_qreg = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        "gate majority a,b,c {\n  cx c,b;\n  cx c,a;\n  ccx a,b,c;\n}\n"
        "qreg cin[1];\nqreg a[4];\n"
        "cx a[0],a[1];\nmeasure a[0] -> c[0];\n"
    )
    with pytest.raises(ValueError, match="gate-definition"):
        normalize_qasm(gate_then_qreg)

    gate_then_reset = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
        "gate g a {\n  x a;\n}\n"
        "reset q[0];\nmeasure q[0] -> c[0];\n"
    )
    with pytest.raises(ValueError, match="gate-definition"):
        normalize_qasm(gate_then_reset)
