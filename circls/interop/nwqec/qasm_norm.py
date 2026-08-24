"""Disclosed QASM normalizations that rescue benchmark files for the
Clifford-only front-end (``load_clifford_mpauli``).

The front-end's faithfulness guard demands "every qubit measured exactly
once, terminally, with the identity classical mapping" (that is the only
semantics nwqec's PBC pass implements).  Many suite files (QASMBench) fail
only on the surface form: they measure a SUBSET of qubits, or permute the
classical bits, or name the register something other than ``q``.  Those are
rescuable without touching the unitary part:

* rename the (single) quantum register to ``q``, or flatten MULTIPLE
  quantum registers into one ``q[N]`` in declaration order (nwqec's own
  global index order, so the rewrite is the identity on the flattened
  circuit);
* verify every ``measure`` is terminal (only measures/barriers after the
  first one) — mid-circuit measurement is NOT rescuable and raises, and
  neither is ``reset``;
* drop all ``measure``/``creg`` statements and append the canonical full
  measurement ``creg c[n]; measure q -> c;``.

Faithfulness: extra terminal Z measurements on previously-unmeasured qubits
commute with the original terminal measurements, so the distribution of the
original bits is unchanged — the normalized circuit measures a SUPERSET of
the original outputs, in qubit order.  Every applied rewrite is returned so
benchmark tables can disclose it (``QASMBench + declared normalization``).
"""
from __future__ import annotations

import re
from typing import List, Tuple

_QREG_RE = re.compile(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]")


def _statements(text: str) -> List[str]:
    # strip line comments BEFORE splitting on ';' — splitting first let a
    # trailing comment swallow the next line's statement, hiding it from
    # the terminal-measurement scan (review 2026-08-20)
    text = "\n".join(line.split("//")[0] for line in text.splitlines())
    out = []
    for raw in text.split(";"):
        stmt = raw.strip()
        if not stmt:
            continue
        out.append(stmt)
    return out


def normalize_qasm(text: str) -> Tuple[str, List[str]]:
    """Return ``(normalized_text, applied_notes)``; raises ``ValueError``
    when the file is not rescuable (mid-circuit measurement, resets).

    Multiple quantum registers flatten into one ``q[N]`` in DECLARATION
    order — the same order nwqec's loader assigns global indices, so the
    rewrite is the identity on the flattened circuit.  Register-form gate
    arguments (``h var;``) are not rewritten; they fail loudly downstream."""
    notes: List[str] = []
    # strip line comments before the register census: a commented-out
    # 'qreg' must not be counted as a real register (the review-2026-08-20
    # comment fix landed in _statements but missed this census, so a
    # '// qreg ...' line spuriously triggered the multi-register flatten).
    _nocomment_text = "\n".join(l.split("//")[0] for l in text.splitlines())
    # gate-definition blocks are not rescuable and defeat every
    # ';'-statement pass below: a gate body's braces are not ';'-aligned,
    # so the statement after the closing '}' arrives GLUED to it
    # ("}\nqreg cin[1]") and the head-keyed scans (qreg drop, reset
    # refusal, measure census) all miss it — measured on
    # QASMBench adder_n10/bigadder_n18, where the flatten emitted a bogus
    # leftover qreg.  In OPENQASM 2.0 braces occur only in gate bodies,
    # and the Clifford front-end rejects every `gate` statement anyway,
    # so refusing here loses nothing and keeps the safety gates sound.
    if "{" in _nocomment_text:
        raise ValueError(
            "contains a gate-definition block — its braces are not "
            "';'-statement-aligned, so the normalization passes cannot "
            "scan it faithfully; not rescuable")
    qregs = _QREG_RE.findall(_nocomment_text)
    if not qregs:
        raise ValueError("no qreg declaration found")
    if len(qregs) > 1:
        offs, total = {}, 0
        for nm, sz in qregs:
            offs[nm] = total
            total += int(sz)
        # drop the qreg declarations on the ';'-statement stream (comments
        # stripped), like every other pass: the old whole-line regex missed
        # a declaration sharing a line or carrying a trailing comment, and
        # _flat then rewrote its brackets into a bogus leftover declaration
        kept = []
        for stmt in _statements(text):
            if re.fullmatch(r"qreg\s+\w+\s*\[\s*\d+\s*\]", stmt):
                continue
            kept.append(stmt + ";")

        def _flat(m):
            nm, i = m.group(1), int(m.group(2))
            if nm not in offs:
                return m.group(0)
            return f"q[{offs[nm] + i}]"

        lines = [re.sub(r"\b(\w+)\s*\[\s*(\d+)\s*\]", _flat, s)
                 for s in kept]
        k = next((i for i, l in enumerate(lines) if "include" in l), 0) + 1
        lines.insert(k, f"qreg q[{total}];")
        text = "\n".join(lines)
        if "q" in offs:
            # a source register literally named 'q' used in register form
            # (gate or measure operand) would silently widen to the whole
            # flattened register (review 2026-08-20) — refuse instead.
            # Checked per statement with comments stripped; barriers are
            # dropped later and stay harmless.
            for stmt in _statements(text):
                head = stmt.split()[0].lower()
                if head in ("barrier", "qreg", "creg", "openqasm",
                            "include"):
                    continue
                if re.search(r"\bq\b(?!\s*\[)", stmt):
                    raise ValueError(
                        f"register-form operand on a source register "
                        f"named 'q' in {stmt!r} — flattening cannot "
                        f"preserve it; not rescuable")
        notes.append("flattened-qregs:"
                     + "+".join(f"{nm}[{sz}]" for nm, sz in qregs))
        qregs = [("q", str(total))]
    reg, n = qregs[0][0], int(qregs[0][1])

    # statement-based like every other pass: the old line-anchored form
    # missed a reset that was not line-leading ('x q[0]; reset q[1];')
    if any(stmt.split()[0].lower() == "reset" for stmt in _statements(text)):
        raise ValueError("contains 'reset' — not rescuable")

    # terminal-measurement check on the statement stream.  A measure may
    # be DEFERRED to the end exactly when every later gate acting on the
    # measured qubit is Z-diagonal ON THAT OPERAND POSITION (commutes with
    # the measurement): measure-then-diagonal and diagonal-then-measure
    # are the same channel, so stripping the measure and re-measuring
    # terminally is the identity.  cx/cy/ccx are diagonal on their
    # CONTROLS but not their target.
    _Z_DIAG_POS = {"z": {0}, "s": {0}, "sdg": {0}, "t": {0}, "tdg": {0},
                   "rz": {0}, "u1": {0}, "p": {0}, "cz": {0, 1},
                   "crz": {0, 1}, "cp": {0, 1}, "cu1": {0, 1},
                   "cx": {0}, "cy": {0}, "ccx": {0, 1}}
    measured_qubits: set = set()
    deferred = False
    for stmt in _statements(text):
        head = stmt.split()[0].lower()
        if head == "measure":
            m = re.search(rf"\b\w+\s*\[\s*(\d+)\s*\]", stmt)
            measured_qubits.add(int(m.group(1)) if m else -1)
        elif measured_qubits and head not in ("barrier",):
            operands = [int(i) for i in re.findall(r"\[\s*(\d+)\s*\]", stmt)]
            measured_all = -1 in measured_qubits
            # a register-form measure covered EVERY qubit, so every
            # operand of a later gate counts as already measured — and a
            # register-form gate operand (no [i]) cannot be checked at
            # all (review 2026-08-20: the old sentinel test was
            # unreachable behind the empty-intersection continue)
            if measured_all and not operands:
                raise ValueError(
                    f"statement {stmt!r} after a register-form measure — "
                    f"not rescuable")
            hit = (set(operands) if measured_all
                   else set(operands) & measured_qubits)
            if not hit:
                continue
            gate = head.split("(")[0]
            diag = _Z_DIAG_POS.get(gate, set())
            bad = [q for pos, q in enumerate(operands)
                   if (measured_all or q in measured_qubits)
                   and pos not in diag]
            if bad:
                raise ValueError(
                    f"mid-circuit measurement (statement {stmt!r} touches "
                    f"already-measured qubit(s) {sorted(hit)} at a "
                    f"non-Z-diagonal position) — not rescuable")
            deferred = True
    if deferred:
        notes.append("deferred-terminal-measurement")

    # Drop measure/creg/barrier at the STATEMENT level, not by line: QASM
    # statements are ';'-separated and need not be line-aligned, so a
    # line-based drop would delete a gate sharing a line with a dropped
    # keyword (over-drop) or miss a keyword that is not line-leading
    # (under-drop).  _statements() strips comments and splits on ';'.
    lines = []
    for stmt in _statements(text):
        head = stmt.split()[0].lower()
        if head in ("measure", "creg", "barrier"):
            if head == "measure":
                notes.append("dropped-measure")
            continue
        lines.append(stmt + ";")
    if reg != "q":
        renamed = []
        for ln in lines:
            renamed.append(re.sub(rf"\b{re.escape(reg)}\b", "q", ln))
        lines = renamed
        notes.append(f"renamed-register:{reg}->q")
    if any(nt == "dropped-measure" for nt in notes):
        notes = [nt for nt in notes if nt != "dropped-measure"]
        notes.append("full-terminal-measurement")
    out = "\n".join(lines).rstrip() + f"\ncreg c[{n}];\nmeasure q -> c;\n"
    return out, notes


def load_with_normalization(qasm_text: str):
    """Try the raw front-end first; on a rescuable rejection, normalize and
    retry.  Returns ``(PauliCircuit, notes)`` where ``notes`` is empty for a
    raw pass."""
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    try:
        return load_clifford_mpauli(qasm_text), []
    except (ValueError, RuntimeError) as first:
        msg = str(first)
        rescuable = ("partial terminal measurement" in msg
                     or "non-identity classical mapping" in msg
                     or "Unknown quantum register" in msg
                     or "outside the Clifford-only whitelist" in msg)
        if not rescuable:
            raise
        normalized, notes = normalize_qasm(qasm_text)
        return load_clifford_mpauli(normalized), notes
