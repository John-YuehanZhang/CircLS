"""QASM-level magic-state gadget rewrite: the form of a non-Clifford circuit that
topologiq + tqec can take end to end.

tqec cannot compile T gates (no magic-state cube, tqec issue #571) and cannot parse
the Y cubes topologiq emits for S gates (issue #548).  So every ``t q[i]`` (and, with
``s_proxy=True``, every ``s q[i]``) is replaced by the textbook injection gadget on a
fresh wire ``a``::

    a = magic input port      cx q[i], a      a = readout port

In tqec the magic input port is filled with an X preparation (a |+> state stands in for
|T>, the paper's X-state proxy) and the readout port with a Z measurement.  With |+> on
``a`` the gadget acts as the identity on q[i] (see ``gadget_flow_check``), so the
circuit stays Clifford and stim can sample it, while the block graph keeps the full
gadget structure.  The same proxy is used on the CircLS side of the paper's Table 2.
"""
import re

import stim

_KEEP = ("h", "x", "z", "s", "cx", "cz")


def rewrite(qasm_text, s_proxy=True):
    """Return (rewritten_qasm_text, meta).  ``creg`` / ``measure`` / ``barrier`` lines
    are dropped (the ports are what topologiq lays out), ``tdg`` counts as ``t`` and
    ``sdg`` as ``s`` (a proxy state does not see the sign)."""
    n = reg = None
    header, gates, magic = [], [], []
    for raw in qasm_text.splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        m = re.match(r"qreg\s+(\w+)\[(\d+)\];", line)
        if m:
            if reg is not None:
                raise ValueError("QASM with more than one qreg is not handled")
            reg, n = m.group(1), int(m.group(2))
            continue
        if line.startswith(("OPENQASM", "include")):
            header.append(line)
            continue
        if line.startswith(("creg", "measure", "barrier")):
            continue
        g = line.split()[0].lower()
        qs = [int(v) for v in re.findall(r"\[(\d+)\]", line)]
        g = {"tdg": "t", "sdg": "s"}.get(g, g)
        if g == "t" or (s_proxy and g == "s"):
            a = n + len(magic)
            magic.append({"wire": a, "target": qs[0], "gate": g})
            gates.append(("cx", [qs[0], a]))
        elif g in _KEEP:
            gates.append((g, qs))
        else:
            raise ValueError(f"gate {g!r} is not handled by the rewrite")
    total = n + len(magic)
    body = [f"qreg {reg}[{total}];"] + [
        f"{g} " + ",".join(f"{reg}[{q}]" for q in qs) + ";" for g, qs in gates]
    meta = {"n_data": n, "n_magic": len(magic), "n_total": total, "magic": magic,
            "gates": len(gates), "s_proxy": s_proxy}
    return "\n".join(header + body) + "\n", meta


def skeleton(qasm_text, drop_s=True):
    """The Clifford reference the block graph is checked against: the circuit with
    every T (and S, when proxied) deleted.  Returns (gates, n)."""
    n = None
    gs = []
    for raw in qasm_text.splitlines():
        line = raw.split("//")[0].strip().rstrip(";")
        if not line:
            continue
        m = re.match(r"qreg\s+\w+\[(\d+)\]", line)
        if m:
            if n is not None:
                raise ValueError("QASM with more than one qreg is not handled")
            n = int(m.group(1))
            continue
        if line.startswith(("OPENQASM", "include", "creg", "measure", "barrier")):
            continue
        g = line.split()[0].lower()
        qs = [int(v) for v in re.findall(r"\[(\d+)\]", line)]
        g = {"tdg": "t", "sdg": "s"}.get(g, g)
        if g == "t" or (drop_s and g == "s"):
            continue
        if g not in _KEEP:
            raise ValueError(f"gate {g!r} is not handled by the skeleton")
        gs.append((g, qs))
    return gs, n


def gadget_flow_check():
    """stim flow check of the proxy gadget (magic wire prepared |+>, CNOT, Z readout):
    it must carry X and Z of the data qubit through unchanged."""
    c = stim.Circuit("RX 1\nCX 0 1\nM 1")
    return {"X0 -> X0": c.has_flow(stim.Flow("X0 -> X0")),
            "Z0 -> Z0": c.has_flow(stim.Flow("Z0 -> Z0"))}
