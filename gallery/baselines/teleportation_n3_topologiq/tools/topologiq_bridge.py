"""topologiq .bgraph -> tqec BlockGraph -> stim, the bridge used for the paper's
TQEC_2 column (topologiq + tqec).  Steps, in the order the notebook calls them:

  build(bgraph_path, n_data)      parse the .bgraph text, drop positionless rows,
                                  run the Hadamard shim (hadamard_rewrite.py: a spatial
                                  Hadamard pipe next to a junction cube is not compilable
                                  by tqec, so the graph is stretched by one unit and a
                                  plain cube inserted; padding is counted), relabel ports
                                  (in_q/out_q -> input_q/output_q for data wires,
                                  magic{k}_in/_out for the gadget wires), build and
                                  validate the tqec BlockGraph.
  equivalent(g, qasm, drop_s)     equivalence gate: the open graph's correlation surfaces
                                  must contain a Heisenberg flow of every X_j / Z_j of the
                                  Clifford skeleton (T and S deleted), with magic inputs
                                  restricted to I/X and magic readouts to I/Z, and the data
                                  ports spanning rank 2n.  Leading/trailing Hadamards may
                                  be absorbed into a port basis.
  fill_rule(g, lead, trail)       port fill: data ports Z (X where an H was absorbed),
                                  magic inputs X (the |+> proxy), magic readouts Z;
                                  kind[pipe axis] = basis.
  compile_closed(g, surfaces, k)  tqec compile (fixed_bulk first, fixed_boundary when
                                  fixed_bulk is not implemented for the graph), detector
                                  database off, followed by an exact p=0 check.

tqec opens multiprocessing pools sized to every core; TQEC_WORKERS (default 8) bounds
them before tqec is imported.
"""
import functools
import itertools
import multiprocessing as _mp
import os
import pathlib
import re
import time

_NW = int(os.environ.get("TQEC_WORKERS", "8"))
_mp.cpu_count = lambda: _NW
_mp.Pool = functools.partial(_mp.Pool, processes=_NW)

import numpy as np  # noqa: E402
import stim  # noqa: E402
from tqec import BlockGraph  # noqa: E402
from tqec.compile.compile import compile_block_graph  # noqa: E402
from tqec.compile.convention import FIXED_BOUNDARY_CONVENTION, FIXED_BULK_CONVENTION  # noqa: E402
from tqec.computation.cube import Port, ZXCube  # noqa: E402
from tqec.computation.pipe import PipeKind  # noqa: E402
from tqec.utils.position import Position3D  # noqa: E402

import hadamard_rewrite as hr  # noqa: E402
from gadget_rewrite import skeleton  # noqa: E402


# ---------- GF(2) helpers ----------
def vec(s, m):
    v = 0
    for i, ch in enumerate(s):
        if ch in "XY":
            v |= 1 << i
        if ch in "ZY":
            v |= 1 << (m + i)
    return v


def basis_of(vs):
    piv = {}
    for v in vs:
        while v:
            p = v.bit_length() - 1
            if p in piv:
                v ^= piv[p]
            else:
                piv[p] = v
                break
    return piv


def in_span(v, piv):
    while v:
        p = v.bit_length() - 1
        if p in piv:
            v ^= piv[p]
        else:
            return False
    return True


def components(g):
    cubes = list(g.cubes)
    pos = {c.position: c for c in cubes}
    adj = {c.position: set() for c in cubes}
    for pp in g.pipes:
        adj[pp.u.position].add(pp.v.position)
        adj[pp.v.position].add(pp.u.position)
    seen, comps = set(), []
    for c in cubes:
        if c.position in seen:
            continue
        stack, comp = [c.position], set()
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.add(v)
            stack.extend(adj[v] - seen)
        sub = BlockGraph("c")
        for v in comp:
            cc = pos[v]
            if cc.is_port:
                sub.add_cube(v, Port(), cc.label)
            else:
                sub.add_cube(v, cc.kind)
        for pp in g.pipes:
            if pp.u.position in comp and pp.v.position in comp and not sub.has_pipe_between(pp.u.position, pp.v.position):
                sub.add_pipe(pp.u.position, pp.v.position, pp.kind)
        comps.append(sub)
    return comps


# ---------- bgraph parsing ----------
def parse_bgraph(path):
    lines = pathlib.Path(path).read_text().splitlines()
    sec = None
    cubes_by_idx, pipes, none_rows, bad_kinds = {}, [], 0, {}
    for line in lines:
        s = line.strip()
        if s.startswith("CUBES:"):
            sec = "c"; continue
        if s.startswith("PIPES:"):
            sec = "p"; continue
        if not s or sec is None:
            continue
        f = [x.strip() for x in s.split(";")]
        if sec == "c":
            idx, x, y, z, kind, label = f[:6]
            if "None" in (x, y, z):          # topologiq's positionless (unvisited) entries
                none_rows += 1
                continue
            if kind != "OOO":
                try:
                    ZXCube.from_str(kind)
                except Exception:
                    bad_kinds[kind] = bad_kinds.get(kind, 0) + 1
            cubes_by_idx[int(idx)] = (Position3D(int(x), int(y), int(z)), kind, label)
        else:
            u, v, kind = f[:3]
            if int(u) in cubes_by_idx and int(v) in cubes_by_idx:
                pipes.append((int(u), int(v), kind))
    return cubes_by_idx, pipes, none_rows, bad_kinds


def relabel(label, n_data):
    m = re.match(r"(in|out)_(\d+)$", label)
    if not m:
        raise ValueError(f"port label without qubit index: {label!r}")
    side, q = m.group(1), int(m.group(2))
    if q < n_data:
        return ("input" if side == "in" else "output") + f"_{q}"
    return f"magic{q - n_data}_{side}"


def build(path, n_data, name="topologiq"):
    """-> (open tqec BlockGraph, info) after the None-row filter and the Hadamard shim."""
    t0 = time.perf_counter()
    cb, pl, none_rows, bad = parse_bgraph(path)
    if bad:
        raise ValueError(f"tqec cannot parse cube kinds {bad}")
    cubes = {}
    for idx, (pos, kind, label) in cb.items():
        cubes[pos] = ("PORT", relabel(label, n_data)) if kind == "OOO" else (kind, label)
    pipes = [(cb[u][0], cb[v][0], k) for u, v, k in pl]
    n0 = sum(1 for k, _ in cubes.values() if k != "PORT")
    rounds = 0
    while rounds < 64:
        step = hr.stretch_and_fix(cubes, pipes)
        if step is None:
            break
        cubes, pipes = step
        rounds += 1
    n1 = sum(1 for k, _ in cubes.values() if k != "PORT")
    info = {"none_rows": none_rows, "shim_rounds": rounds, "padding_blocks": n1 - n0,
            "cubes_open_raw": n0, "cubes_open": n1,
            "ports": sum(1 for k, _ in cubes.values() if k == "PORT"), "pipes": len(pipes),
            "h_pipes": sum(1 for _, _, k in pipes if k.endswith("H"))}
    g = BlockGraph(name)
    for p, (kind, label) in cubes.items():
        g.add_cube(p, Port() if kind == "PORT" else ZXCube.from_str(kind), label)
    for a, b, k in pipes:
        g.add_pipe(a, b, PipeKind.from_str(k))
    g.validate()
    zs = [c.position.z for c in g.cubes if not c.is_port]
    info["layers_open"] = max(zs) - min(zs) + 1
    info["bridge_s"] = round(time.perf_counter() - t0, 2)
    return g, info


# ---------- reference skeleton ----------
def tableau(gs, n):
    c = stim.Circuit()
    for g, qs in gs:
        c.append({"h": "H", "x": "X", "z": "Z", "s": "S", "cx": "CX", "cz": "CZ"}[g], qs)
    c.append("I", [n - 1])
    return stim.Tableau.from_circuit(c)


def strip_runs(gs, n, lead_wires, trail_wires):
    seen = [False] * n
    out = []
    for g, qs in gs:
        if g == "h" and qs[0] in lead_wires and not seen[qs[0]]:
            continue
        for q in qs:
            seen[q] = True
        out.append((g, qs))
    last2 = [-1] * n
    for i, (g, qs) in enumerate(out):
        if len(qs) == 2:
            for q in qs:
                last2[q] = i
    return [(g, qs) for i, (g, qs) in enumerate(out)
            if not (g == "h" and qs[0] in trail_wires and i > last2[qs[0]])]


# ---------- equivalence gate ----------
def surface_cache(g):
    out = []
    for sub in components(g):
        ordered = list(sub.ordered_ports)
        if not ordered:
            continue
        exts = [str(s.external_stabilizer_on_graph(sub)) for s in sub.find_correlation_surfaces()]
        out.append((sub, ordered, exts))
    return out


def check(T, n, cache, idle=frozenset()):
    total = 0
    for sub, ordered, exts in cache:
        m = len(ordered)
        pin = {int(l.split("_")[1]): i for i, l in enumerate(ordered) if l.startswith("input")}
        pout = {int(l.split("_")[1]): i for i, l in enumerate(ordered) if l.startswith("output")}
        mag_in = [i for i, l in enumerate(ordered) if l.startswith("magic") and l.endswith("_in")]
        mag_out = [i for i, l in enumerate(ordered) if l.startswith("magic") and l.endswith("_out")]
        if not pin and not pout:
            continue
        if set(pin) != set(pout):
            return False, f"component wires mismatch in={sorted(pin)} out={sorted(pout)}"
        vecs = [vec(e, m) for e in exts]
        bads = [sum(1 << i for i in mag_in if e[i] not in ("I", "X"))
                | sum(1 << i for i in mag_out if e[i] not in ("I", "Z")) for e in exts]
        pivots, kernel = {}, []
        for j in range(len(exts)):
            c, combo = bads[j], 1 << j
            while c:
                pbit = c.bit_length() - 1
                if pbit in pivots:
                    pc, pcombo = pivots[pbit]
                    c ^= pc
                    combo ^= pcombo
                else:
                    pivots[pbit] = (c, combo)
                    break
            if not c:
                kernel.append(combo)
        data = []
        for combo in kernel:
            v = 0
            for j in range(len(exts)):
                if (combo >> j) & 1:
                    v ^= vecs[j]
            for i in mag_in + mag_out:
                v &= ~((1 << i) | (1 << (m + i)))
            data.append(v)
        piv = basis_of(data)
        total += len(piv)
        for j in pin:
            for L in "XZ":
                p = stim.PauliString("".join(L if k == j else "_" for k in range(n)))
                img = str(T(p)).lstrip("+-")
                s = ["I"] * m
                s[pin[j]] = L
                for k, ch in enumerate(img):
                    if ch == "_":
                        continue
                    if k not in pout:
                        return False, f"generator {L}{j} leaves its component (q{k})"
                    s[pout[k]] = ch
                if not in_span(vec("".join(s), m), piv):
                    return False, f"generator {L}{j} not in span"
    if total != 2 * (n - len(idle)):
        return False, f"data rank {total} != 2n (n={n}, idle wires without ports {sorted(idle)})"
    return True, "EQUIV"


def equivalent(g, qasm_text, drop_s=True):
    """-> (ok, why, (lead_absorbed, trail_absorbed, idle_wires))"""
    gs, n = skeleton(qasm_text, drop_s)
    touched = {q for _, qs in gs for q in qs}
    ported = {int(l.split("_")[1]) for l in g.ordered_ports if l.startswith(("input", "output"))}
    idle = frozenset(q for q in range(n) if q not in touched and q not in ported)
    first, last = {}, {}
    for gname, qs in gs:
        for q in qs:
            first.setdefault(q, gname)
            last[q] = gname
    lead_c = [q for q in range(n) if first.get(q) == "h"]
    trail_c = [q for q in range(n) if last.get(q) == "h"]
    if 2 ** (len(lead_c) + len(trail_c)) > 1 << 16:
        combos = [(set(), set()), (set(lead_c), set()), (set(), set(trail_c)), (set(lead_c), set(trail_c))]
    else:
        combos = [(set(a), set(b))
                  for a in (set(x) for r in range(len(lead_c) + 1) for x in itertools.combinations(lead_c, r))
                  for b in (set(y) for r in range(len(trail_c) + 1) for y in itertools.combinations(trail_c, r))]
    cache = surface_cache(g)
    last_why = "?"
    for a, b in sorted(combos, key=lambda ab: (len(ab[0]) + len(ab[1]))):
        ok, why = check(tableau(strip_runs(gs, n, a, b), n), n, cache, idle)
        if ok:
            note = "EQUIV" if not (a or b) else f"EQUIV (H absorbed into ports: in={sorted(a)} out={sorted(b)})"
            if idle:
                note += f" [idle wires dropped by topologiq: {sorted(idle)}]"
            return True, note, (set(a), set(b), sorted(idle))
        last_why = why
    return False, last_why, (set(), set(), sorted(idle))


# ---------- port fill ----------
def fill_rule(g, lead_abs=(), trail_abs=()):
    fill = {}
    for lb, pos in dict(g.ports).items():
        pipe = g.pipes_at(pos)[0]
        pk = str(pipe.kind)
        if pipe.kind.has_hadamard and not pipe.at_head(pos):   # tail of a Hadamard pipe: flipped wall colours
            pk = "".join({"X": "Z", "Z": "X", "O": "O"}[c] for c in pk[:3])
        if lb.startswith("magic"):
            want = "X" if lb.endswith("_in") else "Z"
        else:
            m = re.match(r"(input|output)_(\d+)$", lb)
            want = "Z"
            if (int(m.group(2)) in lead_abs) if m.group(1) == "input" else (int(m.group(2)) in trail_abs):
                want = "X"
        kind = pk[:3].replace("O", want)
        if len(set(kind)) == 1:
            kind = pk[:3].replace("O", "X" if want == "Z" else "Z")
        fill[lb] = kind
    return fill


# ---------- exact p=0 check ----------
def exact_p0(circ, trials=8):
    out = {}
    try:
        circ.detector_error_model(allow_gauge_detectors=False)
        out["dem_deterministic"] = True
    except Exception as e:
        out["dem_deterministic"] = False
        out["dem_error"] = str(e).splitlines()[0][:120]
    nm, dets, obs = 0, [], {}
    for ins in circ.flattened():
        n = ins.name
        if n in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"):
            nm += len(ins.targets_copy())
        elif n == "MPP":
            nm += sum(1 for t in ins.targets_copy() if not t.is_combiner)
        elif n == "DETECTOR":
            dets.append([nm + t.value for t in ins.targets_copy()])
        elif n == "OBSERVABLE_INCLUDE":
            k = int(ins.gate_args_copy()[0])
            obs.setdefault(k, []).extend(nm + t.value for t in ins.targets_copy())
    dv, ov = [], []
    for t in range(trials):
        sim = stim.TableauSimulator(seed=100 + t)
        sim.do(circ)
        rec = np.array(sim.current_measurement_record(), dtype=np.uint8)
        dv.append(np.array([int(rec[d].sum() % 2) for d in dets], dtype=np.uint8) if dets else np.zeros(0, np.uint8))
        ov.append({k: int(rec[v].sum() % 2) for k, v in obs.items()})
    out["det_stable"] = all((dv[0] == d).all() for d in dv)
    out["det_fired"] = int(dv[0].sum()) if dets else 0
    out["obs_values"] = {k: sorted({o[k] for o in ov}) for k in sorted(obs)}
    out["obs_nondet"] = [k for k, v in out["obs_values"].items() if len(v) > 1]
    out["p0_deterministic"] = out["dem_deterministic"] and out["det_stable"] and not out["obs_nondet"]
    return out


def compile_closed(g, surfaces, k):
    """-> (circuit, convention_name, p0_check, errors).  fixed_bulk first; fixed_boundary
    when fixed_bulk raises or leaves something non-deterministic."""
    errs = []
    for cname, conv in (("fixed_bulk", FIXED_BULK_CONVENTION), ("fixed_boundary", FIXED_BOUNDARY_CONVENTION)):
        try:
            circ = compile_block_graph(g, observables=surfaces, convention=conv).generate_stim_circuit(
                k=k, do_not_use_database=True)
        except Exception as e:
            errs.append(f"{cname}: {type(e).__name__}: {str(e)[:140]}")
            continue
        chk = exact_p0(circ)
        if chk["p0_deterministic"]:
            return circ, cname, chk, errs
        errs.append(f"{cname}: compiled but not deterministic at p=0 ({chk})")
    return None, None, None, errs
