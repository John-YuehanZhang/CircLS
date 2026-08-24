"""End-to-end validation of the GoSC pass-1 sweep (y_free_sweep).

The decisive oracle is exact Kraus-level joint-outcome-distribution equality:
for every outcome bitstring b, the probability of b under the ORIGINAL op
sequence must equal the probability under the swept sequence (gadget pi/4
rotations as unitaries, measurements as signed projectors), simulated with
dense complex matrices on <= 3 qubits.  This catches any mask ordering, sign
or conjugation-direction error — the exact bug class that sank LSC's yfree.
"""
import random

import numpy as np
import pytest

from circls.interop.ir.pauli_algebra import y_free_sweep
from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp

_M2 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _mat(paulis, n):
    out = np.array([[1]], dtype=complex)
    for q in range(n):
        out = np.kron(out, _M2[paulis.get(q, "I")])
    return out


def _joint_distribution(circuit: PauliCircuit):
    """{bitstring over the 'm' ops: probability}, exact dense simulation.

    's' ops are unitaries exp(-i*(pi/4)*sign*A); 'm' ops are projective
    measurements of sign*P whose outcome bit is 0 for +1 and 1 for -1.
    """
    n = circuit.num_qubits
    n_m = sum(1 for op in circuit.ops if op.kind == "m")
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0
    dist = {}

    def run(i, psi, bits, prob):
        if prob < 1e-15:
            return
        if i == len(circuit.ops):
            assert len(bits) == n_m
            dist[tuple(bits)] = dist.get(tuple(bits), 0.0) + prob
            return
        op = circuit.ops[i]
        m = op.sign * _mat(op.paulis, n)
        if op.kind == "m":
            for bit in (0, 1):
                proj = (np.eye(2 ** n) + (1 - 2 * bit) * m) / 2
                phi = proj @ psi
                p = float(np.vdot(phi, phi).real)
                if p > 1e-15:
                    run(i + 1, phi / np.sqrt(p), bits + [bit], prob * p)
        else:   # 's' (pi/4) — 't'/'z' would need pi/8/pi/2; not used here
            assert op.kind == "s"
            u = (np.cos(np.pi / 4) * np.eye(2 ** n)
                 - 1j * np.sin(np.pi / 4) * m)
            run(i + 1, u @ psi, bits, prob)

    run(0, psi0, [], 1.0)
    return dist


def _assert_equivalent(before: PauliCircuit, after: PauliCircuit):
    d1, d2 = _joint_distribution(before), _joint_distribution(after)
    keys = set(d1) | set(d2)
    for k in keys:
        assert abs(d1.get(k, 0.0) - d2.get(k, 0.0)) < 1e-10, (
            f"joint distribution differs at {k}: {d1.get(k, 0)} vs {d2.get(k, 0)}")


# -------------------------------------------------------------------- shapes
def test_sweep_passthrough_without_y():
    c = PauliCircuit(2, [PauliOp("m", {0: "X", 1: "Z"}, -1),
                         PauliOp("m", {1: "X"}, 1)])
    out = y_free_sweep(c)
    assert out.ops == c.ops


def test_sweep_fig10_single_measurement():
    p = {0: "Y", 2: "Y", 3: "Z", 4: "Y", 5: "Y"}
    out = y_free_sweep(PauliCircuit(6, [PauliOp("m", p, 1)]))
    assert [(o.kind, o.paulis, o.sign) for o in out.ops] == [
        ("s", {0: "Z"}, -1),
        ("s", {2: "Z", 4: "Z", 5: "Z"}, -1),
        ("m", {0: "X", 2: "X", 3: "Z", 4: "X", 5: "X"}, -1),
    ]


def test_sweep_cascade_reintroduced_y():
    """The pushed +pi/4 Z-mask turns a downstream X into Y, which must then be
    eliminated in its own right (second gadget pair appears)."""
    c = PauliCircuit(2, [
        PauliOp("m", {0: "Y"}, 1),          # -> mask Z0, pending Z0_{+pi/4}
        PauliOp("m", {0: "X", 1: "Z"}, 1),  # Z0 anticommutes: X0 -> Y0 -> cascade
    ])
    out = y_free_sweep(c)
    kinds = [o.kind for o in out.ops]
    assert kinds == ["s", "m", "s", "m"], out.ops
    assert all("Y" not in o.paulis.values() for o in out.ops)
    _assert_equivalent(c, out)


def test_sweep_positional_m_mapping_and_yfree():
    rng = random.Random(5)
    for _ in range(20):
        n = 3
        ops = [PauliOp("m",
                       {q: rng.choice("XYZ") for q in range(n)
                        if rng.random() < 0.8} or {0: rng.choice("XYZ")},
                       rng.choice([1, -1]))
               for _ in range(rng.randrange(1, 5))]
        c = PauliCircuit(n, ops)
        out = y_free_sweep(c)
        m_ops = [o for o in out.ops if o.kind == "m"]
        assert len(m_ops) == len(ops)
        assert not out.has_y()
        # idempotent: a Y-free circuit sweeps to itself
        again = y_free_sweep(out)
        assert again.ops == out.ops


# ------------------------------------------------------- the decisive oracle
def test_sweep_joint_distribution_random():
    rng = random.Random(31)
    for _ in range(30):
        n = 3
        ops = [PauliOp("m",
                       {q: rng.choice("XYZ") for q in range(n)
                        if rng.random() < 0.75} or {0: rng.choice("XYZ")},
                       rng.choice([1, -1]))
               for _ in range(rng.randrange(1, 5))]
        c = PauliCircuit(n, ops)
        _assert_equivalent(c, y_free_sweep(c))


def test_sweep_end_to_end_from_nwqec():
    """Full Clifford-only pipeline so far: QASM -> load_clifford_mpauli ->
    y_free_sweep, joint distribution preserved."""
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    from tests.test_gosc_pauli_algebra import _random_clifford
    rng = random.Random(59)
    for _ in range(6):
        qasm, _circ = _random_clifford(rng, 3, 20)
        loaded = load_clifford_mpauli(qasm)
        swept = y_free_sweep(loaded)
        assert not swept.has_y()
        _assert_equivalent(loaded, swept)
