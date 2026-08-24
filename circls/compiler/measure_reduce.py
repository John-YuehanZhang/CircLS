"""Terminal measurement-set re-selection.

The Clifford front-end emits n pairwise-commuting terminal measurement
operators m_1..m_n.  Physically executing a high-weight m_k means a long
multi-patch corridor merge — but any GENERATING SET of the abelian group
<m_1..m_n> is the same projective measurement (same common eigenspaces),
and every original outcome is a classical XOR of generator outcomes.  So:
greedily row-reduce the set over GF(2) to minimise operator weights, execute
the light generators, and reconstruct the original bits offline.

Motivating case: GHZ-n's terminal family is the nested prefix chain
{X0 Z1, X0 Z1 Z2, ..., X0 Z1..Z(n-1)} — adjacent products collapse it to
ONE weight-2 operator plus (n-2) single-qubit measurements, which the
existing weight-1 folding machinery absorbs with no corridors at all.

Everything here is compile-time classical; the reconstruction is output-side
XOR (terminal measurements — no runtime feedback).  Sign bookkeeping rides
stim's Pauli-string product (commuting Hermitian products carry a real +-1
sign, never i); the completeness oracle is the exact operator identity
m_k == (+-1) * prod(executed[j] for j in support), checked in tests.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

import stim

from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp

_LETTER = {"X": 1, "Y": 2, "Z": 3}
_UNLETTER = {1: "X", 2: "Y", 3: "Z"}


def _to_stim(op: PauliOp, n: int) -> stim.PauliString:
    ps = stim.PauliString(n)
    for q, letter in op.paulis.items():
        ps[q] = _LETTER[letter]
    if op.sign == -1:
        ps.sign = -1
    return ps


def _to_op(ps: stim.PauliString) -> PauliOp:
    sign = complex(ps.sign)
    assert sign in (1, -1), f"non-Hermitian sign {sign} in measurement product"
    paulis = {q: _UNLETTER[ps[q]] for q in range(len(ps)) if ps[q]}
    return PauliOp("m", paulis, int(sign.real))


def verify_reconstruction(raw: "PauliCircuit", red: "PauliCircuit",
                          recon: "Reconstruction") -> None:
    """Exact operator identity for a (possibly external) re-selection:
    every original m_k must equal (-1)^const[k] times the stim product
    of its reconstruction support over the executed set.  Hook
    validation (docs/API_HOOKS.md): reject loudly, never repair.
    Same identity as the test-suite oracle."""
    n = raw.num_qubits
    if red.num_qubits != n:
        raise ValueError("re-selected circuit changed the qubit count")
    if any(op.kind != "m" for op in red.ops):
        raise ValueError("re-selected circuit must be terminal m ops only")
    if len(recon.inverse) != len(raw.ops):
        raise ValueError(
            "reconstruction must cover every original measurement")
    execs = [_to_stim(o, n) for o in red.ops]
    for k, orig in enumerate(raw.ops):
        prod = stim.PauliString(n)
        for j in sorted(recon.inverse[k]):
            if not 0 <= j < len(execs):
                raise ValueError(
                    f"reconstruction of measurement {k} references "
                    f"executed op {j} out of range")
            prod = prod * execs[j]
        want = _to_stim(orig, n)
        if prod != (want if recon.const[k] == 0 else -want):
            raise ValueError(
                f"reconstruction identity fails on original "
                f"measurement {k}")


@dataclasses.dataclass(frozen=True)
class Reconstruction:
    """``executed[j] = prod(original[k] for k in support[j])`` and
    ``bit(original[k]) = const[k] XOR XOR(bit(executed[j]) for j in inverse[k])``."""
    support: Tuple[frozenset, ...]
    inverse: Tuple[frozenset, ...]
    const: Tuple[int, ...]


def reduce_measurements(circuit: PauliCircuit,
                        ) -> Tuple[PauliCircuit, Reconstruction]:
    """Greedy weight-reduction sweeps over the terminal measurement set."""
    n = circuit.num_qubits
    if any(op.kind != "m" for op in circuit.ops):
        raise ValueError("reduce_measurements expects a terminal m-only circuit")
    reps: List[stim.PauliString] = [_to_stim(op, n) for op in circuit.ops]
    m = len(reps)
    for i in range(m):
        for j in range(i + 1, m):
            if not reps[i].commutes(reps[j]):
                raise ValueError(
                    f"terminal measurements {i} and {j} anticommute — not a "
                    f"legal simultaneous measurement set")
    support: List[set] = [{k} for k in range(m)]

    def weight(ps: stim.PauliString) -> int:
        return len(ps.pauli_indices())

    improved = True
    while improved:
        improved = False
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                cand = reps[i] * reps[j]
                if weight(cand) < weight(reps[j]):
                    reps[j] = cand
                    support[j] ^= support[i]
                    improved = True

    # invert the support matrix over GF(2): original k = prod of executed rows
    mat = [[1 if k in support[j] else 0 for k in range(m)] for j in range(m)]
    aug = [row[:] + [1 if r == j else 0 for r in range(m)]
           for j, row in enumerate(mat)]
    for col in range(m):
        piv = next(r for r in range(col, m) if aug[r][col])
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(m):
            if r != col and aug[r][col]:
                aug[r] = [a ^ b for a, b in zip(aug[r], aug[col])]
    inverse = [frozenset(j for j in range(m) if aug[k][m + j]) for k in range(m)]

    # sign constants: m_k == (-1)^c * prod executed[j], letters must match exactly
    const: List[int] = []
    originals = [_to_stim(op, n) for op in circuit.ops]
    for k in range(m):
        prod = stim.PauliString(n)
        for j in sorted(inverse[k]):
            prod = prod * reps[j]
        if prod == originals[k]:
            const.append(0)
        elif prod == -originals[k]:
            const.append(1)
        else:
            raise AssertionError(
                f"reconstruction mismatch on measurement {k}: the letters of "
                f"the generator product differ from the original operator")

    executed = PauliCircuit(n, [_to_op(ps) for ps in reps])
    return executed, Reconstruction(tuple(frozenset(s) for s in support),
                                    tuple(inverse), tuple(const))
