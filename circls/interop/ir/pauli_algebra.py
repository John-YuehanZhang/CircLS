"""Phase-tracked Pauli algebra + GoSC Y-elimination (the yfree replacement core).

Every rule here is a line-for-line transcription from Litinski, "A Game of
Surface Codes" (GoSC, arXiv:1808.02892, Quantum 3, 128 (2019)); no invented
intermediate forms.  The transcribed statements:

RULE 1 (GoSC Sec. 1, Fig. 4a — moving a pi/4 rotation to the right):
    "If P and P' commute, P_pi/4 can simply be moved past P'_phi.  If they
    anticommute, P'_phi turns into (iPP')_phi when P_pi/4 is moved to the
    right."

RULE 2 (GoSC Sec. 1, Fig. 4c — same rule absorbs Cliffords into measurements):
    "the Clifford gates can be absorbed by the final measurements, turning Z
    measurements into Pauli product measurements.  The commutation rules of
    this final step are shown in Fig. 4c and are similar to the commutation of
    Clifford gates past rotations."

RULE 3 (GoSC Sec. 2.1, Fig. 10 — Y elimination by Z-mask pi/4 conjugation):
    "One possibility to replace Y operators by X or Z operators is via pi/4
    rotations, since Y_pi/4 = Z_pi/4 X_pi/4 Z_-pi/4.  Rotations with an even
    number of Y's require two pi/4 rotations, while an odd number of Y's can
    be handled by one rotation.  Only the left two pi/4 rotations in Fig. 10
    need to be performed explicitly.  The right two rotations can be commuted
    to the end of the circuit, changing the subsequent pi/8 rotations."

Sign conventions pinned by the paper text (P_phi = exp(-i*phi*P), so
S = Z_pi/4, T = Z_pi/8):  the identity Y_pi/4 = Z_pi/4 X_pi/4 Z_-pi/4 is in
MATRIX order (rightmost acts first), i.e. the sandwich in circuit-time order is

    [A_-pi/4]  [P'_phi]  [A_pi/4]        with  P' = i * A * P

per Z-mask A.  For an even number of Y's the two masks are "first Y alone" +
"all remaining Y's" — exactly the split drawn in Fig. 10 (Z_1 and
Z_3 Z_5 Z_6 for (Y x 1 x Y x Z x Y x Y)).  NOTE the honest transformed sign for
that example is MINUS (X x 1 x X x Z x X x X): the figure omits signs, and
getting this wrong is precisely the LSC ``to_y_free_equivalent`` bug class
(direction/sign of the conjugation).  ``tests/test_gosc_pauli_algebra.py``
pins it with an exact 64x64 matrix identity.

This module is pure (stdlib only) — the independent cross-validation oracles
(stim PauliString products, numpy matrix identities) live in the tests, so the
transcription and its checker share no code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from circls.interop.ir.pauli_ir import PauliOp

# single-qubit products: (left, right) -> (phase exponent k with i^k, letter)
# XY = iZ, YZ = iX, ZX = iY; reversed order picks up -i (exponent 3).
_MUL: Dict[Tuple[str, str], Tuple[int, str]] = {
    ("I", "I"): (0, "I"),
    ("I", "X"): (0, "X"), ("X", "I"): (0, "X"), ("X", "X"): (0, "I"),
    ("I", "Y"): (0, "Y"), ("Y", "I"): (0, "Y"), ("Y", "Y"): (0, "I"),
    ("I", "Z"): (0, "Z"), ("Z", "I"): (0, "Z"), ("Z", "Z"): (0, "I"),
    ("X", "Y"): (1, "Z"), ("Y", "X"): (3, "Z"),
    ("Y", "Z"): (1, "X"), ("Z", "Y"): (3, "X"),
    ("Z", "X"): (1, "Y"), ("X", "Z"): (3, "Y"),
}


@dataclass
class PhasedPauli:
    """A Pauli product with an exact global phase i^k, k in {0,1,2,3}.

    ``paulis`` maps qubit -> 'X'|'Y'|'Z' (identity omitted), like
    :class:`~circls.interop.ir.pauli_ir.PauliOp`.  Hermitian operators (the only
    ones that may label a rotation or measurement) have k in {0, 2}.
    """
    phase_exp: int
    paulis: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.phase_exp %= 4
        for q, p in self.paulis.items():
            if p not in ("X", "Y", "Z"):
                raise ValueError(f"qubit {q}: letter must be X/Y/Z, got {p!r}")

    # ------------------------------------------------------------- bridges
    @classmethod
    def from_pauli_op(cls, op: PauliOp) -> "PhasedPauli":
        return cls(0 if op.sign > 0 else 2, dict(op.paulis))

    def to_sign(self) -> int:
        """+1/-1 for a Hermitian operator; raises if the phase is imaginary."""
        if self.phase_exp == 0:
            return 1
        if self.phase_exp == 2:
            return -1
        raise ValueError(
            f"operator has imaginary phase i^{self.phase_exp} — not Hermitian; "
            f"a rotation/measurement axis must have phase +1 or -1")

    def to_pauli_op(self, kind: str) -> PauliOp:
        return PauliOp(kind, dict(self.paulis), self.to_sign())

    # ------------------------------------------------------------- algebra
    def y_qubits(self) -> List[int]:
        return sorted(q for q, p in self.paulis.items() if p == "Y")

    def commutes_with(self, other: "PhasedPauli") -> bool:
        anti = sum(1 for q, p in self.paulis.items()
                   if q in other.paulis and other.paulis[q] != p)
        return anti % 2 == 0

    def mul(self, other: "PhasedPauli") -> "PhasedPauli":
        """self * other (operator product, exact phase)."""
        k = self.phase_exp + other.phase_exp
        out: Dict[int, str] = dict(self.paulis)
        for q, p in other.paulis.items():
            dk, letter = _MUL[(out.get(q, "I"), p)]
            k += dk
            if letter == "I":
                out.pop(q, None)
            else:
                out[q] = letter
        return PhasedPauli(k % 4, out)

    def __str__(self) -> str:
        pre = {0: "+", 1: "+i", 2: "-", 3: "-i"}[self.phase_exp]
        body = " ".join(f"{p}{q}" for q, p in sorted(self.paulis.items())) or "I"
        return f"{pre}{body}"


def zmask(qubits) -> PhasedPauli:
    """The Pauli product Z on the given qubits (a Fig. 10 conjugation mask)."""
    return PhasedPauli(0, {q: "Z" for q in qubits})


def commute_pi4_past(mask: PhasedPauli, mask_sign: int,
                     op: PhasedPauli) -> PhasedPauli:
    """RULE 1/2: move ``(mask_sign * mask)_pi/4`` to the RIGHT past ``op``.

    Commuting: ``op`` unchanged.  Anticommuting: ``op -> i * (sign*mask) * op``
    ("P'_phi turns into (iPP')_phi when P_pi/4 is moved to the right").  The
    same transformation applies when ``op`` labels a measurement (Fig. 4c).
    """
    if mask_sign not in (1, -1):
        raise ValueError(f"mask_sign must be +1/-1, got {mask_sign!r}")
    if mask.commutes_with(op):
        return op
    out = PhasedPauli((1 + (0 if mask_sign > 0 else 2)) % 4, {}).mul(mask).mul(op)
    out.to_sign()   # Hermiticity assertion: phase must land on +/-1
    return out


def y_free_sweep(circuit) -> "PauliCircuit":
    """GoSC pass-1 sweep: rewrite a Pauli-op sequence into an equivalent one
    whose axes are X/Z-only (Fig. 10 applied along the whole circuit).

    Single left-to-right pass.  For each op: (1) fold every pending trailing
    ``+pi/4`` mask through it, newest (time-adjacent) first — RULE 1/2; this
    may INTRODUCE Y into the op; (2) if the op now contains Y, apply RULE 3:
    the leading masks are emitted as explicit rotations, the trailing partners
    join ``pending``.  After the last op the pending masks act on a state
    nothing ever uses again and are dropped (GoSC: "The right two rotations
    can be commuted to the end of the circuit") — sound only for circuits that
    END in terminal measurements, which is the m_pauli stream's contract.

    Output ops (order preserved; the i-th 'm' corresponds to the i-th input
    'm', so the classical bit mapping is positional; a sign of -1 on an 'm'
    means "flip that classical bit"):

    * ``PauliOp('s', A, -1)`` — an EXPLICIT pi/4 rotation to execute
      (exp(-i*(pi/4)*sign*A) = A_{-pi/4}, the left/leading rotations of
      Fig. 10; pass 2 turns each into a |Y>-consuming gadget).  Masks are
      always pure-Z products.
    * ``PauliOp(kind, P, sign)`` — the original op, now X/Z-only.
    """
    from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp as _PauliOp
    pending: List[PhasedPauli] = []     # trailing +pi/4 masks, in time order
    out = []
    for op in circuit.ops:
        cur = PhasedPauli.from_pauli_op(op)
        for mask in reversed(pending):          # adjacent (newest) mask first
            cur = commute_pi4_past(mask, 1, cur)
        masks, cur = eliminate_y(cur)
        for a in masks:                          # leading A_{-pi/4}, Fig. 10 order
            out.append(_PauliOp("s", dict(a.paulis), -1))
        # trailing partners run in reverse order right after the op
        pending = list(reversed(masks)) + pending
        out.append(cur.to_pauli_op(op.kind))
    for o in out:
        if any(p == "Y" for p in o.paulis.values()):
            raise AssertionError(f"Y survived the sweep: {o}")
    return PauliCircuit(circuit.num_qubits, out)


def eliminate_y(op: PhasedPauli) -> Tuple[List[PhasedPauli], PhasedPauli]:
    """RULE 3 (Fig. 10): rewrite an op containing Y as Z-mask pi/4 conjugations.

    Returns ``(masks, op')`` with every mask a Z-product on Y-positions of
    ``op`` and ``op'`` Y-free.  Placement in circuit-time order (per
    Y_pi/4 = Z_pi/4 X_pi/4 Z_-pi/4, matrix order):

        masks[0]_-pi/4, masks[1]_-pi/4, ..., op', ..., masks[1]_pi/4, masks[0]_pi/4

    i.e. the LEADING rotations are -pi/4 about the masks (outermost first) and
    the TRAILING +pi/4 rotations are the ones "commuted to the end of the
    circuit".  ``op'`` folds one factor ``i * mask`` per mask:  op' = i*A2*(i*A1*op).

    Odd number of Y's: one mask on all Y positions.  Even: two masks — first Y
    alone, remaining Y's together (the exact Fig. 10 split: Z_1 and Z_3 Z_5 Z_6).
    """
    ys = op.y_qubits()
    if not ys:
        return [], op
    if len(ys) % 2 == 1:
        masks = [zmask(ys)]
    else:
        masks = [zmask(ys[:1]), zmask(ys[1:])]
    out = op
    for m in masks:
        if m.commutes_with(out):    # cannot happen for odd-size Z-masks on Y's
            raise AssertionError(f"mask {m} unexpectedly commutes with {out}")
        out = PhasedPauli(1, {}).mul(m).mul(out)    # op' = i * A * op
    if out.y_qubits():
        raise AssertionError(f"Y left after elimination: {out}")
    out.to_sign()   # Hermiticity assertion
    return masks, out


def reorder_weight1_last(circuit) -> Tuple["PauliCircuit", List[int]]:
    """M1 pre-pass: move every weight-1 measurement to the tail so it lands
    after all wider measurements touching its qubit (each qubit has at most
    one weight-1 op among an independent commuting set).  Legal because the
    m_pauli frame is mutually commuting — guaranteed algebraically
    (m_i = C^dag Z_i C, conjugation preserves commutators) and re-checked
    here pairwise as a fail-loud guard against upstream corruption.

    Returns ``(reordered_circuit, perm)`` with ``perm[new_pos] = old_index``
    (the classical bit relabelling the caller must carry)."""
    from circls.interop.ir.pauli_ir import PauliCircuit
    ops = circuit.ops
    bad = [op.kind for op in ops if op.kind != "m"]
    if bad:
        raise ValueError(f"reorder_weight1_last expects a pure 'm' stream, "
                         f"got kinds {sorted(set(bad))}")
    pp = [PhasedPauli.from_pauli_op(op) for op in ops]
    for i in range(len(pp)):
        for j in range(i + 1, len(pp)):
            if not pp[i].commutes_with(pp[j]):
                raise ValueError(
                    f"m ops {i} and {j} anticommute ({ops[i]} vs {ops[j]}) — "
                    f"a valid terminal-measurement frame must commute; "
                    f"refusing to reorder")
    perm = ([i for i, op in enumerate(ops) if op.weight != 1]
            + [i for i, op in enumerate(ops) if op.weight == 1])
    return PauliCircuit(circuit.num_qubits, [ops[i] for i in perm]), perm


def drop_free_prefix(swept) -> "PauliCircuit":
    """m1 peephole: delete the LEADING run of 's' ops from a swept stream.

    Sound ONLY when the data qubits start in the Z basis product state
    |0...0> — the Clifford pipeline's contract (m_i = C^dag Z_i C measured on
    |0...0>; to_experiment_inputs initialises every data patch 'Z').  A
    leading pure-Z pi/4 rotation acts on a Z-basis eigenstate: a global
    phase, physically nothing — yet each would otherwise cost a full
    |Y>-resource gadget (measured: 51% of gadgets in random streams; the
    cx;h;sx benchmark halves to 142 qubits / 62 ticks).  Z-purity is
    CHECKED per op, not assumed: the Clifford front end only ever emits
    pure-Z heads here, but the t_as_s proxy front end injects arbitrary-
    axis rotation heads, and once a non-Z rotation has acted the state is
    no longer |0...0>, so the run must stop at the first non-Z op.  Do NOT
    bury this inside expand_gadgets: that function is deliberately
    initial-state agnostic."""
    from circls.interop.ir.pauli_ir import PauliCircuit
    i = 0
    while (i < len(swept.ops) and swept.ops[i].kind == "s"
           and all(l == "Z" for l in swept.ops[i].paulis.values())):
        i += 1
    return PauliCircuit(swept.num_qubits, list(swept.ops[i:]))
