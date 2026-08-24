"""Repository-agnostic intermediate representation (IR) for a Pauli-based circuit.

This is the single contract every front-end translates to/from, so no two
external repos (nwqec, lattice-surgery-compiler, LightStim) ever touch each
other's format directly — each adapter converts only between its own repo and
this IR.

A :class:`PauliOp` is one Pauli-product operation: a rotation (``t`` = pi/8 / T,
``s`` = pi/4 / S, ``z`` = pi / Pauli, in the Litinski angle convention) or a
projective measurement (``m``), about a signed multi-qubit Pauli string.  Only
the axis (per-qubit X/Y/Z) and the kind matter for the downstream 2-colouring
check; the sign is carried for faithfulness but is irrelevant to bus assignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

_PAULIS = ("X", "Y", "Z")
_KINDS = ("t", "s", "z", "m")   # pi/8 rotation, pi/4 rotation, pi rotation, measurement


@dataclass
class PauliOp:
    """One Pauli-product operation on a subset of qubits.

    kind:   't' | 's' | 'z' | 'm'
    paulis: {qubit_index: 'X'|'Y'|'Z'} — identity qubits are omitted.
    sign:   +1 or -1 (rotation direction / measurement sign; unused by 2-colouring).
    """
    kind: str
    paulis: Dict[int, str]
    sign: int = 1

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {self.sign!r}")
        for q, p in self.paulis.items():
            if p not in _PAULIS:
                raise ValueError(f"pauli on qubit {q} must be one of {_PAULIS}, got {p!r}")

    def has_y(self) -> bool:
        return any(p == "Y" for p in self.paulis.values())

    @property
    def qubits(self) -> List[int]:
        """Sorted list of non-identity qubit indices."""
        return sorted(self.paulis)

    @property
    def weight(self) -> int:
        return len(self.paulis)

    def is_mixed(self) -> bool:
        """True if this op measures/rotates about both an X and a Z on different qubits."""
        kinds = set(self.paulis.values())
        return ("X" in kinds) and ("Z" in kinds)

    def __str__(self) -> str:
        sgn = "+" if self.sign > 0 else "-"
        body = " ".join(f"{p}{q}" for q, p in sorted(self.paulis.items())) or "I"
        return f"{self.kind}({sgn}{body})"


@dataclass
class PauliCircuit:
    """An ordered list of :class:`PauliOp` on ``num_qubits`` qubits."""
    num_qubits: int
    ops: List[PauliOp] = field(default_factory=list)

    def has_y(self) -> bool:
        return any(op.has_y() for op in self.ops)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for op in self.ops:
            out[op.kind] = out.get(op.kind, 0) + 1
        return out

    def y_fraction(self) -> float:
        if not self.ops:
            return 0.0
        return sum(1 for op in self.ops if op.has_y()) / len(self.ops)

    def mixed_fraction(self) -> float:
        if not self.ops:
            return 0.0
        return sum(1 for op in self.ops if op.is_mixed()) / len(self.ops)
