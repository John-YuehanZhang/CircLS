"""Front-end: NWQEC Pauli-Based Circuit  -->  interop :class:`PauliCircuit`.

NWQEC (pnnl/nwqec) compiles a QASM circuit to a PBC: a sequence of Pauli
rotations ``t_pauli`` (pi/8 / T), ``s_pauli`` (pi/4 / S), ``z_pauli`` (pi) and
terminal ``m_pauli`` measurements, each about a signed Pauli string.  This
adapter reads that schedule out of an NWQEC ``Circuit`` (via its QASM dump) into
the repo-agnostic IR.  The PBC generally CONTAINS Y — run
:func:`circls.interop.ir.yfree.to_y_free` afterwards to get an X/Z-only circuit.

``nwqec`` is imported LAZILY (only :func:`load_pbc` needs it); :func:`from_nwqec`
works on an already-built Circuit object and imports nothing.
"""
from __future__ import annotations

import re
from typing import Optional

from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp

_KIND = {"t_pauli": "t", "s_pauli": "s", "z_pauli": "z", "m_pauli": "m"}
_QREG_RE = re.compile(r"qreg\s+\w+\s*\[\s*(\d+)\s*\]")
_OP_RE = re.compile(r"^\s*([tszm]_pauli)\s+([+-])([XYZI]+)\s*;?\s*$")


def _parse_qasm_str(qasm: str) -> PauliCircuit:
    num_qubits: Optional[int] = None
    ops = []
    for line in qasm.splitlines():
        m = _QREG_RE.search(line)
        if m and num_qubits is None:
            num_qubits = int(m.group(1))
            continue
        m = _OP_RE.match(line)
        if not m:
            continue
        kind = _KIND[m.group(1)]
        sign = 1 if m.group(2) == "+" else -1
        paulis = {q: ch for q, ch in enumerate(m.group(3)) if ch != "I"}
        ops.append(PauliOp(kind, paulis, sign))
    if num_qubits is None:
        # fall back to the widest Pauli string seen
        num_qubits = 1 + max((max(op.paulis, default=-1) for op in ops), default=-1)
    return PauliCircuit(num_qubits, ops)


def from_nwqec(pbc_circuit) -> PauliCircuit:
    """Read an already-PBC-converted NWQEC ``Circuit`` into a :class:`PauliCircuit`.

    ``pbc_circuit`` must be the output of ``nwqec.to_pbc(...)``.  Uses only the
    object's ``to_qasm_str()`` method, so this function imports nothing.
    """
    if not hasattr(pbc_circuit, "to_qasm_str"):
        raise TypeError("expected an NWQEC PBC Circuit with .to_qasm_str(); "
                        f"got {type(pbc_circuit).__name__}")
    return _parse_qasm_str(pbc_circuit.to_qasm_str())


def load_pbc(qasm: str, *, keep_cx: bool = True, optimize_t_count: bool = False,
             **to_pbc_kw) -> PauliCircuit:
    """Compile a QASM string (or a path to a .qasm file) to a PBC and read it into
    a :class:`PauliCircuit`.  Lazily imports ``nwqec``.

    ``keep_cx=True`` keeps CX gates in place (mostly weight-1 rotations, fewer Y);
    ``keep_cx=False`` folds Cliffords into the axes (high-weight, mostly Y).  Both
    generally still contain Y — remove it with :func:`yfree.to_y_free`.
    """
    try:
        import nwqec  # noqa: WPS433 (lazy optional dependency)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "load_pbc needs the 'nwqec' package (pip install nwqec, or build from "
            "the cloned source). from_nwqec() works on an already-built Circuit "
            "without importing nwqec.") from e
    import os

    circuit = nwqec.load_qasm(qasm) if os.path.exists(qasm) else _load_qasm_str(nwqec, qasm)
    pbc = nwqec.to_pbc(circuit, keep_cx=keep_cx, optimize_t_count=optimize_t_count, **to_pbc_kw)
    return from_nwqec(pbc)


def _load_qasm_str(nwqec, qasm: str):
    if hasattr(nwqec, "load_qasm_str"):
        return nwqec.load_qasm_str(qasm)
    import os
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".qasm", delete=False) as fh:
        fh.write(qasm)
        path = fh.name
    try:
        return nwqec.load_qasm(path)
    finally:
        os.unlink(path)      # m2: every string load used to leak one file


def load_clifford_mpauli(qasm: str) -> PauliCircuit:
    """Clifford-only entry: QASM -> exactly the ``n`` terminal measurement
    operators ``m_i = C^dagger Z_i C`` (sign included, index order = qubit
    order 0..n-1), as ``PauliOp('m', ...)``.

    Two independent tableaus compute the answer and must agree on the LETTERS:
    nwqec's ``to_pbc(keep_cx=False)`` m_pauli stream, and a stim inverse
    tableau built from the same QASM (whitelisted Clifford gates only).  The
    SIGNS are taken from stim alone:

    * KNOWN UPSTREAM BUG (nwqec, found 2026-08-02): ``pbc_pass.hpp`` sweeps the
      operations in REVERSE order but ``vtab.hpp`` conjugates each row by the
      gate itself (U P U^dag) instead of its inverse (U^dag P U).  Self-inverse
      gates are unaffected, so all LETTERS and the signs of H/X/Z/CX/CZ/SWAP
      paths are right, but every S/SDG/SX/SXDG acting on an X/Y row flips the
      m_pauli sign the wrong way (minimal case ``s q0; h q0``: nwqec says +Y,
      the true C^dag Z C is -Y).  ``tests/test_gosc_pauli_algebra.py::
      test_nwqec_mpauli_sign_bug_documented`` pins the upstream behaviour — if
      nwqec ever fixes it, that test fires and this workaround can go.

    Faithfulness guards (each violation is a loud error, never silent):

    * nwqec's PBC pass SKIPS ``measure``/``reset`` statements and
      unconditionally emits one terminal ``m_pauli`` per qubit
      (``pbc_pass.hpp``), so the PBC is only faithful to "unitary Clifford
      circuit with every qubit measured in Z at the very end".  ``reset`` is
      rejected anywhere; ``measure`` only in trailing position.
    * ``keep_cx`` is forced False (``keep_cx=True`` emits a mixed ``s_pauli``
      stream and has a separate double-reversal bug at pbc_pass.hpp:161);
      ``optimize_t_count`` is forced False (``fuse_t`` reverses the m_pauli
      emission order, breaking the qubit mapping).
    * Any ``t_pauli``/``s_pauli``/``z_pauli`` in the nwqec output means the
      input contained T/Tdg/CCX — this pipeline version is Clifford-only.
    """
    import os
    is_path = os.path.exists(qasm)
    text = open(qasm).read() if is_path else qasm
    _check_terminal_measurement_only(text)

    try:
        import nwqec  # noqa: WPS433 (lazy optional dependency)
    except ImportError as e:  # pragma: no cover
        raise ImportError("load_clifford_mpauli needs the 'nwqec' package") from e
    loaded = nwqec.load_qasm(qasm) if is_path else _load_qasm_str(nwqec, text)
    pbc = nwqec.to_pbc(loaded, keep_cx=False, optimize_t_count=False)
    circuit = from_nwqec(pbc)

    non_m = sorted({op.kind for op in circuit.ops if op.kind != "m"})
    if non_m:
        raise ValueError(
            f"Clifford-only pipeline: to_pbc emitted {non_m} rotations, so the "
            f"input contained non-Clifford gates (T/Tdg/CCX). pi/8 handling is "
            f"out of scope for this version.")
    if len(circuit.ops) != circuit.num_qubits:
        raise ValueError(
            f"expected exactly num_qubits={circuit.num_qubits} m_pauli ops, "
            f"got {len(circuit.ops)} — nwqec output not in the assumed form")

    # independent stim tableau: signs, plus letter cross-validation vs nwqec
    import stim
    n, stim_circ = _clifford_qasm_to_stim(text)
    if n != circuit.num_qubits:
        raise ValueError(f"qreg width {n} != nwqec num_qubits {circuit.num_qubits}")
    tinv = stim_circ.to_tableau().inverse()
    ops = []
    for i, nw_op in enumerate(circuit.ops):
        want = tinv(_z_i(stim, i, n))
        letters = {q: "_XYZ"[want[q]] for q in range(n) if want[q]}
        if letters != nw_op.paulis:
            raise RuntimeError(
                f"m_pauli letter mismatch on qubit {i}: nwqec {nw_op.paulis} vs "
                f"stim {letters} — two independent tableaus disagree, refusing "
                f"to guess")
        sign = complex(want.sign)
        if sign not in (1, -1):
            raise AssertionError(f"non-Hermitian sign {sign} from stim on qubit {i}")
        ops.append(PauliOp("m", letters, int(sign.real)))
    return PauliCircuit(n, ops)


def _z_i(stim, i: int, n: int):
    z = stim.PauliString(n)
    z[i] = 3
    return z


# Clifford-only QASM gate whitelist -> stim gate names.
_STIM_1Q = {"h": "H", "s": "S", "sdg": "S_DAG", "sx": "SQRT_X",
            "sxdg": "SQRT_X_DAG", "x": "X", "y": "Y", "z": "Z", "id": "I"}
_STIM_2Q = {"cx": "CX", "cy": "CY", "cz": "CZ", "swap": "SWAP"}
_ARG_RE = re.compile(r"q\[(\d+)\]")


def _clifford_qasm_to_stim(text: str):
    """Parse the whitelisted Clifford-QASM subset into a stim Circuit.

    Loud failure on anything outside the whitelist — this parser exists only to
    provide the second, independent tableau for :func:`load_clifford_mpauli`.
    """
    import stim
    n = None
    circ = stim.Circuit()
    for stmt in _qasm_statements(text):
        head = stmt.split()[0].lower()
        if head in ("openqasm", "include", "creg", "barrier", "measure"):
            continue
        if head == "qreg":
            if n is not None:
                raise ValueError("multiple qreg declarations are not supported")
            m = _QREG_RE.search(stmt + ";")
            if not m:
                raise ValueError(f"unparseable qreg statement: {stmt!r}")
            n = int(m.group(1))
            circ.append("I", range(n))     # pin the tableau width
            continue
        args = [int(a) for a in _ARG_RE.findall(stmt)]
        if head in _STIM_1Q and len(args) == 1:
            circ.append(_STIM_1Q[head], args)
        elif head in _STIM_2Q and len(args) == 2:
            circ.append(_STIM_2Q[head], args)
        else:
            raise ValueError(
                f"statement {stmt!r} is outside the Clifford-only whitelist "
                f"({sorted(_STIM_1Q)} + {sorted(_STIM_2Q)})")
    if n is None:
        raise ValueError("no qreg declaration found")
    return n, circ


def _qasm_statements(text: str):
    """QASM statements with LINE-scoped ``//`` comments stripped FIRST, then
    ``;``-split.  The reverse order (split-then-strip, the C3 bug) glues a
    comment to the following statement and silently deletes it — or
    resurrects code written INSIDE a comment as a phantom gate whenever the
    comment contains a ``;``."""
    no_comments = "\n".join(line.split("//")[0] for line in text.splitlines())
    for raw in no_comments.split(";"):
        stmt = raw.strip()
        if stmt:
            yield stmt


_TCLASS_FIXED = {"openqasm", "include", "qreg", "creg", "measure",
                 "barrier", "id", "h", "s", "sdg", "x", "y", "z", "sx",
                 "sxdg", "cx", "cy", "cz", "swap", "ccx", "cswap",
                 "t", "tdg"}
_TCLASS_PARAM = {"rx", "ry", "rz", "u1", "u2", "u3", "u", "p",
                 "crz", "cp", "cu1"}


def _check_tclass_envelope(text: str) -> None:
    """Refuse input outside the exact Clifford+T envelope BEFORE nwqec
    sees it: ``to_clifford_t`` gridsynth-APPROXIMATES arbitrary-angle
    rotations instead of failing, and the post-synthesis PBC contains
    nothing but t/m ops, so no downstream guard can tell an exact
    program from an approximated one (review 2026-08-20).  A rotation
    exp(-i*k*pi/8*P) is exactly Clifford+T for integer k, so every
    angle argument must be a multiple of pi/4."""
    import math
    for stmt in _qasm_statements(text):
        head = stmt.split()[0]
        name = head.split("(")[0].lower()
        if name in _TCLASS_FIXED:
            continue
        if name not in _TCLASS_PARAM:
            raise ValueError(
                f"gate {name!r} is outside the T-class envelope "
                f"(statement {stmt!r})")
        m = re.match(r"\w+\s*\(([^)]*)\)", stmt)
        args = m.group(1) if m else ""
        for arg in args.split(","):
            expr = arg.strip().lower().replace("pi", repr(math.pi))
            if not re.fullmatch(r"[0-9eE().*/+\- ]+", expr):
                raise ValueError(
                    f"unparseable angle {arg!r} in {stmt!r}")
            val = eval(expr, {"__builtins__": {}}, {})
            if abs((val / (math.pi / 4)) - round(val / (math.pi / 4))) > 1e-9:
                raise ValueError(
                    f"angle {arg.strip()!r} in {stmt!r} is not a multiple "
                    f"of pi/4 — exact Clifford+T synthesis is impossible "
                    f"and gridsynth approximation is refused")


def _check_terminal_measurement_only(text: str) -> None:
    """Reject ``reset`` anywhere; allow ``measure`` only in trailing position
    (nothing but further measures/barriers after it) and only in the one form
    that matches nwqec's semantics (one m_pauli per qubit, in qubit order):
    every qubit measured exactly once with the IDENTITY classical mapping
    (``measure q[i] -> c[i]`` or the register form ``measure q -> c``).
    Partial or permuted measurement would be silently reinterpreted by
    nwqec's PBC pass as "measure all qubits in order" — mis-compiled
    classical bits — so it fails loudly here instead."""
    seen_measure = False
    n_qubits = None
    measured = set()
    for stmt in _qasm_statements(text):
        head = stmt.split()[0].lower()
        if head == "qreg":
            m = _QREG_RE.search(stmt + ";")
            if m and n_qubits is None:
                n_qubits = int(m.group(1))
            continue
        if head == "reset":
            raise ValueError(
                "input contains 'reset' — nwqec's PBC pass silently ignores "
                "it; resets are not supported in the Clifford-only pipeline")
        if head == "measure":
            seen_measure = True
            m = _MEASURE_RE.match(stmt)
            if not m:
                raise ValueError(f"unparseable measure statement: {stmt!r}")
            qi, ci = m.group("qi"), m.group("ci")
            if (qi is None) != (ci is None):
                raise ValueError(
                    f"measure statement mixes register and indexed operands: "
                    f"{stmt!r}")
            if qi is None:      # register form: identity over all qubits
                if measured:
                    raise ValueError(
                        f"register-form measure after indexed measures: {stmt!r}")
                if n_qubits is None:
                    raise ValueError("measure before qreg declaration")
                measured.update(range(n_qubits))
            else:
                qi, ci = int(qi), int(ci)
                if qi != ci:
                    raise ValueError(
                        f"non-identity classical mapping {stmt!r}: nwqec "
                        f"emits one m_pauli per qubit IN QUBIT ORDER — a "
                        f"permuted -> c[j] target would be silently ignored")
                if qi in measured:
                    raise ValueError(f"qubit {qi} measured twice: {stmt!r}")
                measured.add(qi)
            continue
        if head == "barrier":
            continue
        if seen_measure:
            raise ValueError(
                f"statement {stmt!r} appears after a 'measure' — nwqec's PBC "
                f"pass silently drops measurements, so only terminal "
                f"measurement (all qubits, end of circuit) is faithful")
    if measured and n_qubits is not None and measured != set(range(n_qubits)):
        missing = sorted(set(range(n_qubits)) - measured)
        raise ValueError(
            f"partial terminal measurement (qubits {missing} unmeasured): "
            f"nwqec measures ALL qubits unconditionally, so the classical "
            f"record would not match the source circuit's intent")


_MEASURE_RE = re.compile(
    r"^measure\s+\w+(?:\[(?P<qi>\d+)\])?\s*->\s*\w+(?:\[(?P<ci>\d+)\])?$")


_SIM_METH = {"h": "h", "s": "s", "sdg": "s_dag", "sx": "sqrt_x",
             "sxdg": "sqrt_x_dag", "x": "x", "y": "y", "z": "z",
             "cx": "cnot", "cy": "cy", "cz": "cz", "swap": "swap"}


def load_clifford_t_as_s(qasm: str, return_proxy_qasm: bool = False):
    """Clifford+T entry with the S-state proxy: QASM -> PauliCircuit
    whose T rotations are replaced by S rotations (each t_pauli
    ``exp(-i pi/8 s P)`` becomes ``PauliOp('s', P, s)``, i.e.
    ``exp(-i pi/4 s P)``), followed by the terminal measurement frame.
    The returned program IS the proxy program; every downstream oracle
    must be run against it, not against the original.

    Sign policy: upstream signs are not trusted on EITHER stream.
    nwqec's PBC pass has a conjugation-direction defect (pnnl/nwqec#5,
    the t_pauli extension of the m_pauli bug documented in
    load_clifford_mpauli) that flips signs whenever S/SDG/SX/SXDG
    conjugates an X/Y row.  Letters and op structure are
    cross-validated against nwqec and must agree loudly; every sign
    comes from a stim tableau walk over nwqec's own FLATTENED gate
    list (to_qasm_str, so ccx/cswap are expanded exactly as the PBC
    pass saw them):

    * rotation j: axis+sign = (Clifford-prefix tableau)^-1 applied to
      Z_q, times +1 for t / -1 for tdg (validated against dense
      simulation on 21 adversarial+random circuits, 2026-08-19);
    * terminal frame: (full Clifford tableau)^-1 applied to Z_i (the
      existing m_pauli defense, same direction convention).
    """
    import os
    import stim
    is_path = os.path.exists(qasm)
    text = open(qasm).read() if is_path else qasm
    _check_terminal_measurement_only(text)
    _check_tclass_envelope(text)
    try:
        import nwqec  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        raise ImportError("load_clifford_t_as_s needs the 'nwqec' package") from e
    loaded = nwqec.load_qasm(qasm) if is_path else _load_qasm_str(nwqec, text)
    # to_clifford_t expands composite gates (ccx, cswap) and synthesizes
    # pi/4-multiple rotations into basic Clifford+T; the flattened walk
    # and the PBC are taken from the SAME transformed circuit so the
    # rotation order is aligned by construction (the letter cross-check
    # below still enforces it loudly).
    loaded = nwqec.to_clifford_t(loaded)
    flat = loaded.to_qasm_str()
    _check_terminal_measurement_only(flat)
    pbc = nwqec.to_pbc(loaded, keep_cx=False, optimize_t_count=False)
    circuit = from_nwqec(pbc)

    bad = sorted({op.kind for op in circuit.ops} - {"t", "m"})
    if bad:
        raise ValueError(
            f"t_as_s proxy handles t_pauli rotations only; to_pbc emitted "
            f"{bad} — the input is outside the Clifford+T class")
    n_m = sum(1 for op in circuit.ops if op.kind == "m")
    if n_m != circuit.num_qubits:
        raise ValueError(
            f"expected num_qubits={circuit.num_qubits} m_pauli ops, got {n_m}")

    # in-house axes+signs from the flattened gate walk
    n = None
    sim = None
    refs = []
    for stmt in _qasm_statements(flat):
        head = stmt.split()[0].lower()
        if head in ("openqasm", "include", "creg", "barrier", "measure"):
            continue
        if head == "qreg":
            if n is not None:
                raise ValueError("multiple qreg declarations are not supported")
            m = _QREG_RE.search(stmt + ";")
            n = int(m.group(1))
            if n != circuit.num_qubits:
                raise ValueError(
                    f"flattened qreg width {n} != nwqec num_qubits "
                    f"{circuit.num_qubits}")
            sim = stim.TableauSimulator()
            sim.set_num_qubits(n)
            continue
        args = [int(a) for a in _ARG_RE.findall(stmt)]
        if head in ("t", "tdg") and len(args) == 1:
            axis = sim.current_inverse_tableau()(_z_i(stim, args[0], n))
            letters = {q: "_XYZ"[axis[q]] for q in range(n) if axis[q]}
            sgn = complex(axis.sign)
            if sgn not in (1, -1):
                raise AssertionError(f"non-Hermitian axis sign {sgn}")
            refs.append((letters, (1 if head == "t" else -1) * int(sgn.real)))
        elif head in _SIM_METH and len(args) in (1, 2):
            getattr(sim, _SIM_METH[head])(*args)
        elif head == "id" and len(args) == 1:
            pass
        else:
            raise ValueError(
                f"statement {stmt!r} is outside the Clifford+T whitelist")

    nw_t = [op for op in circuit.ops if op.kind == "t"]
    if len(nw_t) != len(refs):
        raise RuntimeError(
            f"rotation count mismatch: nwqec emitted {len(nw_t)} t_pauli, "
            f"the flattened gate walk found {len(refs)} T gates")
    ops = []
    for j, ((letters, sign), nw_op) in enumerate(zip(refs, nw_t)):
        if letters != nw_op.paulis:
            raise RuntimeError(
                f"t_pauli letter mismatch at rotation {j}: nwqec "
                f"{nw_op.paulis} vs stim {letters} — refusing to guess")
        ops.append(PauliOp("s", letters, sign))
    tinv = sim.current_inverse_tableau()
    nw_m = [op for op in circuit.ops if op.kind == "m"]
    for i, nw_op in enumerate(nw_m):
        want = tinv(_z_i(stim, i, n))
        letters = {q: "_XYZ"[want[q]] for q in range(n) if want[q]}
        if letters != nw_op.paulis:
            raise RuntimeError(
                f"m_pauli letter mismatch on qubit {i}: nwqec {nw_op.paulis} "
                f"vs stim {letters} — refusing to guess")
        sgn = complex(want.sign)
        ops.append(PauliOp("m", letters, int(sgn.real)))
    pc = PauliCircuit(n, ops)
    if not return_proxy_qasm:
        return pc
    # the T->S substituted flat circuit is pure Clifford: it is the
    # proxy program's gate-level form, usable as verify()'s logical
    # oracle (tdg before t so the substitution never bites a prefix)
    proxy = re.sub(r"(?m)^(\s*)tdg\b", r"\1sdg", flat)
    proxy = re.sub(r"(?m)^(\s*)t\b", r"\1s", proxy)
    return pc, proxy
