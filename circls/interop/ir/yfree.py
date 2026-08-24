"""Front-end: remove Y from a :class:`PauliCircuit` using the lattice-surgery
compiler's Litinski Y-operator decomposition.

Wraps ``lsqecc.pauli_rotations.circuit.PauliOpCircuit.to_y_free_equivalent``
(latticesurgery-com/lattice-surgery-compiler, arXiv:2302.02459): every Pauli
rotation/measurement whose axis contains Y is rewritten as an equivalent
X/Z-only sequence, ``Rot(...Y...) = Zrot(pi/4) . Rot(...X...) . Zrot(-pi/4)``,
using pi/4 Z-rotations to conjugate X into Y.  The result is Y-free, at the cost
of extra pure-Z rotations.

``lsqecc`` is imported LAZILY.  It is NOT pip-installable here (its QASM parser
needs an old qiskit) but the Y-free core is pure Python; we only import the
``pauli_rotations`` classes, adding the compiler's ``src`` to ``sys.path``.  Set
env ``LSQECC_SRC`` to override the default location.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp

# IR kind  <->  lsqecc rotation_amount (Litinski angle convention)
_KIND_TO_FRACTION = {"t": Fraction(1, 8), "s": Fraction(1, 4), "z": Fraction(1, 2)}
_FRACTION_TO_KIND = {Fraction(1, 8): "t", Fraction(1, 4): "s", Fraction(1, 2): "z"}

_DEFAULT_LSQECC_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                 "lattice-surgery-compiler", "src"))


def _import_lsqecc():
    src = os.environ.get("LSQECC_SRC", _DEFAULT_LSQECC_SRC)
    if not os.path.isdir(os.path.join(src, "lsqecc")):
        raise ImportError(
            f"lsqecc source not found under {src!r}. Clone latticesurgery-com/"
            "lattice-surgery-compiler and set env LSQECC_SRC to its 'src' dir.")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from lsqecc.pauli_rotations.circuit import PauliOpCircuit
        from lsqecc.pauli_rotations.rotation import (
            Measurement, PauliOperator, PauliRotation,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"found lsqecc at {src!r} but could not import its pauli_rotations "
            f"module: {e}") from e
    return PauliOpCircuit, PauliRotation, Measurement, PauliOperator


def _to_lsqecc(circuit: PauliCircuit, cls):
    PauliOpCircuit, PauliRotation, Measurement, PauliOperator = cls
    n = circuit.num_qubits
    out = PauliOpCircuit(n, "interop")
    for op in circuit.ops:
        ops_list = [
            PauliOperator(op.paulis.get(q, "I")) for q in range(n)
        ]
        if op.kind == "m":
            block = Measurement.from_list(ops_list, isNegative=(op.sign < 0))
        else:
            block = PauliRotation.from_list(ops_list, _KIND_TO_FRACTION[op.kind])
        out.add_pauli_block(block)
    return out


def _from_lsqecc(ls_circuit, cls) -> PauliCircuit:
    _, PauliRotation, Measurement, PauliOperator = cls
    ops = []
    n = ls_circuit.qubit_num
    for block in ls_circuit.ops:
        paulis = {
            q: block.ops_list[q].value
            for q in range(n)
            if block.ops_list[q] != PauliOperator.I
        }
        if isinstance(block, Measurement):
            sign = -1 if getattr(block, "isNegative", False) else 1
            ops.append(PauliOp("m", paulis, sign))
        else:
            # A negative rotation angle R_P(-t) == R_{-P}(t): fold its sign into the
            # Pauli-string sign, map |angle| to the kind (2-colouring ignores sign).
            amt = block.rotation_amount
            kind = _FRACTION_TO_KIND.get(abs(amt))
            if kind is None:
                raise ValueError(
                    f"unexpected rotation amount {amt} from lsqecc; expected "
                    f"+/- one of {sorted(_FRACTION_TO_KIND)}")
            ops.append(PauliOp(kind, paulis, 1 if amt > 0 else -1))
    return PauliCircuit(n, ops)


def to_y_free(circuit: PauliCircuit) -> PauliCircuit:
    """Return a Y-operator-free equivalent of ``circuit`` (X/Z axes only), via
    lsqecc's Litinski decomposition.  Adds extra pure-Z rotations per Y removed.
    A circuit already free of Y round-trips unchanged (aside from that)."""
    cls = _import_lsqecc()
    ls_in = _to_lsqecc(circuit, cls)
    ls_out = ls_in.to_y_free_equivalent()
    result = _from_lsqecc(ls_out, cls)
    if result.has_y():  # pragma: no cover - defensive
        raise RuntimeError("lsqecc to_y_free_equivalent left residual Y operators")
    return result
