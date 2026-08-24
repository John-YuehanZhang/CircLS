"""The S-state proxy front end (load_clifford_t_as_s).

Covers: the upstream t_pauli sign defect (pnnl/nwqec#5) stays
documented, the in-house signs match dense simulation of the
substituted (T->S) circuit, composite gates expand, and the guards
hold.  Skipped wholesale when nwqec is not installed.
"""
import math

import numpy as np
import pytest

pytest.importorskip("nwqec")

from circls.interop.nwqec.frontend import load_clifford_t_as_s

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1]).astype(complex)
H = (X + Z) / math.sqrt(2)
S = np.diag([1, 1j])
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
G1 = {"h": H, "s": S, "sdg": S.conj().T, "x": X, "y": Y, "z": Z,
      "sx": SX, "sxdg": SX.conj().T,
      # the PROXY ground truth substitutes S for T at the gate level
      "t": S, "tdg": S.conj().T}


def _qasm(n, lines):
    out = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[%d];\ncreg c[%d];\n' % (n, n)
    for g, qs in lines:
        out += f"{g} q[{qs[0]}]" + (f", q[{qs[1]}]" if len(qs) > 1 else "") + ";\n"
    for i in range(n):
        out += f"measure q[{i}] -> c[{i}];\n"
    return out


def _u_gate(n, q, g):
    m = np.array([[1]], dtype=complex)
    for i in range(n):
        m = np.kron(m, g if i == q else I2)
    return m


def _u_cx(n, c, t):
    dim = 2 ** n
    m = np.zeros((dim, dim), dtype=complex)
    for b in range(dim):
        bits = [(b >> (n - 1 - i)) & 1 for i in range(n)]
        if bits[c]:
            bits[t] ^= 1
        m[sum(x << (n - 1 - i) for i, x in enumerate(bits)), b] = 1
    return m


def _dense_proxy_dist(n, lines):
    """Z-basis distribution of the T->S substituted gate circuit."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1
    for g, qs in lines:
        u = _u_cx(n, *qs) if g == "cx" else _u_gate(n, qs[0], G1[g])
        psi = u @ psi
    return np.abs(psi) ** 2


def _pmat(n, letters, sign):
    P = {"X": X, "Y": Y, "Z": Z}
    m = np.array([[1]], dtype=complex)
    for i in range(n):
        m = np.kron(m, P[letters[i]] if i in letters else I2)
    return sign * m


def _loader_dist(pc):
    """Joint distribution of the loader's proxy program, densely."""
    import itertools
    n = pc.num_qubits
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1
    ms = []
    for op in pc.ops:
        Pm = _pmat(n, op.paulis, op.sign)
        if op.kind == "s":
            psi = (math.cos(math.pi / 4) * np.eye(2 ** n)
                   - 1j * math.sin(math.pi / 4) * Pm) @ psi
        else:
            ms.append(Pm)
    out = np.zeros(2 ** len(ms))
    for idx, bits in enumerate(itertools.product([0, 1], repeat=len(ms))):
        proj = np.eye(2 ** n, dtype=complex)
        for Pm, b in zip(ms, bits):
            proj = proj @ (np.eye(2 ** n) + ((-1) ** b) * Pm) / 2
        out[idx] = np.real(np.vdot(psi, proj @ psi))
    return out


CASES = [
    ("sx_t", 1, [("sx", [0]), ("t", [0])]),
    ("sx_t_sx_h", 1, [("sx", [0]), ("t", [0]), ("sx", [0]), ("h", [0])]),
    ("mixed3", 3, [("h", [0]), ("t", [0]), ("cx", [0, 1]), ("s", [1]),
                   ("tdg", [1]), ("sx", [2]), ("t", [2]), ("cx", [1, 2]),
                   ("h", [2])]),
    ("t_between_cx", 2, [("h", [0]), ("cx", [0, 1]), ("t", [1]), ("s", [1]),
                          ("cx", [0, 1]), ("h", [1])]),
]


@pytest.mark.parametrize("name,n,lines", CASES, ids=[c[0] for c in CASES])
def test_proxy_matches_dense_substitution(name, n, lines):
    pc = load_clifford_t_as_s(_qasm(n, lines))
    d_loader = _loader_dist(pc)
    d_truth = _dense_proxy_dist(n, lines)
    assert np.abs(d_loader - d_truth).sum() < 1e-9


def test_upstream_t_sign_bug_stays_documented():
    # pnnl/nwqec#5 minimal case: sx;t — raw nwqec sign is -1 (wrong),
    # the loader's in-house sign is +1.  If this assertion on the RAW
    # output ever fails, upstream fixed the bug: drop the workaround.
    import nwqec
    import os, tempfile
    from circls.interop.nwqec.frontend import from_nwqec
    fd, path = tempfile.mkstemp(suffix=".qasm")
    os.write(fd, _qasm(1, [("sx", [0]), ("t", [0])]).encode())
    os.close(fd)
    try:
        raw = from_nwqec(nwqec.to_pbc(nwqec.load_qasm(path), keep_cx=False,
                                      optimize_t_count=False))
    finally:
        os.unlink(path)
    nw_t = [op for op in raw.ops if op.kind == "t"][0]
    assert nw_t.paulis == {0: "Y"}
    assert nw_t.sign == -1, "upstream fixed pnnl/nwqec#5 — drop the workaround"
    pc = load_clifford_t_as_s(_qasm(1, [("sx", [0]), ("t", [0])]))
    ours = [op for op in pc.ops if op.kind == "s"][0]
    assert (ours.paulis, ours.sign) == ({0: "Y"}, 1)


def test_ccx_expands_to_seven_rotations():
    qasm = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'
            "ccx q[0], q[1], q[2];\n"
            + "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(3)))
    pc = load_clifford_t_as_s(qasm)
    assert sum(1 for op in pc.ops if op.kind == "s") == 7
    assert sum(1 for op in pc.ops if op.kind == "m") == 3


def test_mid_circuit_measure_still_rejected():
    qasm = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "t q[0];\nmeasure q[0] -> c[0];\nh q[1];\n"
            "measure q[1] -> c[1];\n")
    with pytest.raises(ValueError):
        load_clifford_t_as_s(qasm)


# ── pipeline level: gadget execution + symbolic-tracker reconstruction ──────

def _affine_truth(proxy_qasm, shots=256):
    import stim
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    from circls.tools.reporting import affine_structure
    raw = load_clifford_mpauli(proxy_qasm)
    letter = {"X": 1, "Y": 2, "Z": 3}
    mops = []
    for op in raw.ops:
        ps = stim.PauliString(raw.num_qubits)
        for q, l in op.paulis.items():
            ps[q] = letter[l]
        if op.sign == -1:
            ps.sign = -1
        mops.append(ps)
    rows = []
    for shot in range(shots):
        ts = stim.TableauSimulator(seed=1000 + shot)
        rows.append([ts.measure_observable(p) for p in mops])
    return affine_structure(np.array(rows, dtype=bool))


@pytest.mark.parametrize("name,body", [
    # two commuting X-axis pi/4 heads on ONE qubit: the raw-post
    # reconstruction reads a random bit here (split-byproduct
    # contamination) — the symbolic tracker must recover determinism
    ("t_t_x", "h q[0];\nt q[0];\nt q[0];\nh q[0];\n"),
    # X head then Z head on one qubit: anticommuting pair, exercises the
    # Gadget.conds chain AND the drop_free_prefix non-Z guard
    ("two_t", "h q[0];\nt q[0];\nh q[0];\nt q[0];\nh q[0];\n"),
])
def test_pipeline_proxy_single_qubit(name, body):
    from circls.pipeline import compile_qasm
    from circls.tools.evaluate import verify
    from circls.tools.reporting import affine_structure, sample_program_bits
    qasm = ('OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[1];\ncreg c[1];\n" + body + "measure q[0] -> c[0];\n")
    cp = compile_qasm(qasm, distance=3, t_as_s=True)
    r = verify(cp)
    assert r.ok and r.logical is True and not r.skipped, r
    assert _affine_truth(cp.source_qasm) == affine_structure(
        sample_program_bits(cp, shots=256, seed=29))


def test_pipeline_proxy_toffoli():
    """|110> through ccx under the proxy: 7 mixed-axis gadgets; the
    affine structure (all three bits deterministic) must match the dense
    truth of the substituted program."""
    from circls.pipeline import compile_qasm
    from circls.tools.evaluate import verify
    from circls.tools.reporting import affine_structure, sample_program_bits
    qasm = ('OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[3];\ncreg c[3];\n"
            "x q[0];\nx q[1];\nccx q[0], q[1], q[2];\n"
            + "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(3)))
    cp = compile_qasm(qasm, distance=3, t_as_s=True)
    assert len(cp.program.gadgets) >= 5
    assert any(l == "X" for g in cp.program.gadgets
               for l in g.mask.values())
    r = verify(cp)
    assert r.ok and r.logical is True, r
    assert _affine_truth(cp.source_qasm) == affine_structure(
        sample_program_bits(cp, shots=256, seed=29))


def test_drop_free_prefix_keeps_non_z_heads():
    from circls.interop.ir.pauli_algebra import drop_free_prefix
    from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp
    c = PauliCircuit(1, [PauliOp("s", {0: "X"}, 1),
                         PauliOp("s", {0: "Z"}, -1),
                         PauliOp("m", {0: "Z"}, 1)])
    out = drop_free_prefix(c)
    # X head not droppable; the Z head AFTER it is not droppable either
    # (the state is no longer |0...0> once the X rotation acted)
    assert [op.kind for op in out.ops] == ["s", "s", "m"]
    z_only = PauliCircuit(1, [PauliOp("s", {0: "Z"}, -1),
                              PauliOp("m", {0: "Z"}, 1)])
    assert [op.kind for op in drop_free_prefix(z_only).ops] == ["m"]


def test_expand_gadgets_kappa_and_conds():
    from circls.interop.ir.gosc_gadgets import expand_gadgets
    from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp
    prog = expand_gadgets(PauliCircuit(2, [
        PauliOp("s", {0: "X"}, 1),          # sigma=+1 -> kappa 0
        PauliOp("s", {0: "Z"}, -1),         # sigma=-1 -> kappa 1, chains
        PauliOp("s", {1: "Z"}, -1),         # disjoint -> no chain
        PauliOp("m", {0: "Z", 1: "Z"}, 1),
    ]))
    assert [(g.kappa, g.conds) for g in prog.gadgets] == \
        [(0, []), (1, [0]), (1, [])]
    # the m op anticommutes with the X mask only
    assert prog.out_bits[0].gadgets == [0]


# ── adversarial-review regressions (2026-08-20) ─────────────────────────

def test_envelope_rejects_arbitrary_angles():
    from circls.interop.nwqec.frontend import load_clifford_t_as_s
    H = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
    with pytest.raises(ValueError, match="not a multiple of pi/4"):
        load_clifford_t_as_s(H + "rz(0.3) q[0];\nmeasure q[0] -> c[0];\n")
    with pytest.raises(ValueError, match="outside the T-class envelope"):
        load_clifford_t_as_s(H + "rccx q[0];\nmeasure q[0] -> c[0];\n")
    # pi/4 multiples stay accepted (exp(-i k pi/8 P) is exact Clifford+T)
    pc = load_clifford_t_as_s(
        H + "rx(pi*0.25) q[0];\nmeasure q[0] -> c[0];\n")
    assert sum(1 for op in pc.ops if op.kind == "s") == 1


def test_scheduler_cannot_float_head_past_weight1():
    """review: a commuting proxy head scheduled behind a weight-1
    terminal measurement used to hard-fail folded_out_bits; the pipeline
    now falls back to the unscheduled order."""
    from circls.pipeline import compile_qasm
    from circls.tools.evaluate import verify
    qasm = ('OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[3];\ncreg c[3];\nx q[1];\nsx q[1];\ntdg q[1];\n"
            "tdg q[0];\nh q[1];\ncx q[2],q[1];\n"
            + "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(3)))
    cp = compile_qasm(qasm, distance=3, t_as_s=True)
    assert verify(cp).logical is True


def test_pure_z_masks_still_take_the_tracker():
    """review: the dispatch used to test gadget-mask letters, so an
    all-Z-mask proxy with an X-bearing m frame read a deterministic
    parity as a free coin on the legacy raw-post path."""
    from circls.pipeline import compile_qasm
    from circls.tools.evaluate import verify
    qasm = ('OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\ntdg q[1];\nh q[0];\ncx q[0],q[1];\n"
            "sx q[1];\nsdg q[0];\ny q[1];\nh q[0];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    cp = compile_qasm(qasm, distance=3, t_as_s=True)
    assert cp.t_as_s
    assert all(set(g.mask.values()) == {"Z"} for g in cp.program.gadgets)
    assert verify(cp).logical is True
