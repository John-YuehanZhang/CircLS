"""Bridge: DASCOT's mapping+routing decisions -> the CircLS lowering.

Reads a (program, seed) row of dascot_layout.jsonl (the wisq scmr
output recorded by dascot_sweep.py: qubit->cell map, per-step parallel
gate sets, each CX with its explicit corridor path) and replays the
SAME plan on the CircLS backend, reusing the lsqecc bridge's entire
translation and verification machinery.

Replay contract (every CircLS optimization off):

  - placement: their map, verbatim; corridors: their per-CX paths,
    injected as explicit routes (their router's cells, not ours);
  - CX realization: DASCOT drops 1q gates and schedules bare CXs; a CX
    is replayed as this paper's verified CNOT gadget (Fig. 3): a fresh
    |0> ancilla on the path cell next to the target, XX(anc, target)
    across the adjacent seam, ZZ(control, anc) through the remaining
    path cells, ancilla read out in X-bar (its scripted death);
  - gate order: THEIR step order; each qubit's original 1q gates are
    replayed (frame-conjugated) immediately before that qubit's next
    CX, preserving every per-qubit dependency while honoring their
    reordering of disjoint CXs;
  - execution: serial (parallel_steps=False) — the appendix table
    reports BOTH their claimed step count and our executed rounds, raw;
  - data patches static, ancilla lives scripted, tracker-native
    observables, and the same four verification gates as the lsqecc
    replay (silence, determinism, obs >= plan dims, output-bit affine
    equality against a logical reference of the same gadget stream).
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

from lsqecc_bridge import LLIOp   # noqa: E402  (shared op format)


def cell_rc(cell: int, width: int):
    return divmod(cell, width)     # (row, col)


def parse_program_gates(qasm: str):
    """Original program: per-qubit-ordered 1q gates, CX list (in program
    order), measured qubits.  Mirrors wisq's front end (which keeps only
    the CXs) plus everything it drops."""
    import re
    gates_1q = []            # (op_index, gate, qubit)
    cxs = []                 # (op_index, control, target)
    measured = set()
    n = 0
    idx = 0
    for raw in qasm.splitlines():
        l = raw.strip()
        if not l or l.startswith(("//", "OPENQASM", "include", "creg",
                                  "barrier")):
            continue
        m = re.match(r"qreg\s+\w+\[(\d+)\]", l)
        if m:
            n = int(m.group(1))
            continue
        if l.startswith("measure"):
            qs = [int(x) for x in re.findall(r"q\[(\d+)\]", l)]
            if qs:
                measured.update(qs)
            elif re.search(r"measure\s+\w+\s*->", l):
                measured.update(range(n))
            continue
        m = re.match(r"(\w+)\s+(.*?);?$", l)
        g = m.group(1).lower()
        qs = [int(x) for x in re.findall(r"q\[(\d+)\]", m.group(2))]
        if g == "cx":
            cxs.append((idx, qs[0], qs[1]))
        elif g in ("h", "x", "y", "z", "s", "sdg"):
            gates_1q.append((idx, g, qs[0]))
        else:
            raise NotImplementedError(f"gate {g} in {l!r}")
        idx += 1
    return n, gates_1q, cxs, sorted(measured)


_G1Q = {"h": "H", "x": "X", "y": "Y", "z": "Z", "s": "S", "sdg": "SD"}


def build_llops(row, qasm):
    """DASCOT schedule + original QASM -> (ops in LLIOp form, placement,
    routes, num_data).  Ancilla ids are allocated per CX gadget; routes
    maps the mbm's SEQUENTIAL index among joint steps to corridor cells."""
    n, gates_1q, cxs, measured = parse_program_gates(qasm)
    width = row["arch"]["width"] if isinstance(row.get("arch"), dict) \
        else None
    if width is None:
        # arch may be a list [height, width] or dict; probe: dict with
        # 'width'/'height'; fall back to max cell + 1 heuristics
        arch = row.get("arch")
        if isinstance(arch, (list, tuple)) and len(arch) == 2:
            width = arch[1]
        else:
            raise ValueError(f"cannot read arch width: {arch!r}")

    placement = {q: cell_rc(c, width) for q, c in
                 (row["map"].items() if isinstance(row["map"], dict)
                  else row["map"])}
    placement = {int(q): rc for q, rc in placement.items()}
    # idle qubits (no cx) are absent from their map but the program
    # still measures them: place each on an unused cell of THEIR
    # reserved algorithm-qubit set, deterministically
    used_cells = set(placement.values())
    for step in row["steps"]:
        for g in step:
            used_cells.update(cell_rc(x, width) for x in g["path"])
    height = row["arch"]["height"]
    msf = {cell_rc(c, width) for c in row["arch"].get("magic_states", [])}
    spare = [cell_rc(c, width) for c in sorted(row["arch"]["alg_qubits"])
             if cell_rc(c, width) not in used_cells]
    spare += [(r_, c_) for r_ in range(height) for c_ in range(width)
              if (r_, c_) not in used_cells and (r_, c_) not in msf
              and (r_, c_) not in spare]
    for q in range(n):
        if q not in placement:
            placement[q] = spare.pop(0)

    # their steps: schedule order of CX executions.  wisq gate ids index
    # into row["gates"]; each scheduled gate carries its path.
    sched = []               # (control, target, path_cells) in THEIR order
    for step in row["steps"]:
        for g in step:
            gid, path = g["id"], g.get("path", [])
            gate = row["gates"][gid]
            c, t = gate[0], gate[1]
            sched.append((c, t, [cell_rc(x, width) for x in path]))

    # per-qubit queues of original 1q gates before each CX occurrence
    from collections import defaultdict, deque
    per_qubit = defaultdict(deque)
    for idx, g, q in gates_1q:
        per_qubit[q].append((idx, g))
    cx_pos = defaultdict(deque)
    for idx, c, t in cxs:
        cx_pos[(c, t)].append(idx)

    ops = []
    routes = {}
    unstitches = {}
    next_anc = n
    joint_i = 0

    def flush_1q(q, before_idx):
        while per_qubit[q] and per_qubit[q][0][0] < before_idx:
            _, g = per_qubit[q].popleft()
            ops.append(LLIOp("gate1q", [(q, "")], state=_G1Q[g]))

    for (c, t, path) in sched:
        idx = cx_pos[(c, t)].popleft()
        flush_1q(c, idx)
        flush_1q(t, idx)
        anc = next_anc
        next_anc += 1
        if not path:
            raise ValueError(f"CX {c}->{t} scheduled with empty path")
        anc_cell = path[-1]           # adjacent to the target's X edge
        placement[anc] = anc_cell
        ops.append(LLIOp("init", [(anc, "")], state="|0>"))
        ops.append(LLIOp("mbm", [(anc, "X"), (t, "X")]))
        joint_i += 1                  # adjacent seam: no corridor
        corridor = path[:-1]          # control Z edge ... up to anc
        ops.append(LLIOp("mbm", [(c, "Z"), (anc, "Z")]))
        if corridor:
            # engine route cells are (col, row); placement dicts are
            # (row, col) and build_compiled flips those itself
            routes[joint_i] = [(cc, rr) for (rr, cc) in corridor]
            crc, arc = placement[c], anc_cell
            if abs(crc[0] - arc[0]) + abs(crc[1] - arc[1]) == 1:
                # the anc cell happens to border the control too; without
                # this the engine classifies the pair as a direct-seam
                # step and IGNORES the corridor (seam-rule crash when the
                # accidental seam runs perpendicular to Z-bar)
                unstitches[joint_i] = [(f"q{c}", f"q{anc}")]
        joint_i += 1
        ops.append(LLIOp("msp", [(anc, "X")]))
    for q in range(n):
        flush_1q(q, float("inf"))
    return ops, placement, routes, unstitches, n, measured


def main():
    import argparse
    import lsqecc_bridge as LB
    import numpy as np
    from circls.tools.reporting import (affine_structure,
                                          sample_program_bits)
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="default: the seed with the median step count")
    ap.add_argument("--rows", default=str(
        _ROOT / "experiments/results/best5/dascot_layout.jsonl"))
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--p", type=float, default=5e-4)
    ap.add_argument("--max-shots", type=int, default=1_000_000)
    ap.add_argument("--max-errors", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ler-budget", type=int, default=3600)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    prov = None
    if args.measure:
        from provenance import provenance
        prov = provenance(args.allow_dirty)

    cand = {}
    for line in Path(args.rows).read_text().splitlines():
        r = json.loads(line)
        if (r.get("record") == "dascot_layout"
                and r["program"] == args.program
                and r["status"] == "ok"):
            cand[r["seed"]] = r
    if not cand:
        raise SystemExit(f"no ok rows for {args.program}")
    if args.seed is not None:
        row = cand[args.seed]
    else:
        by_steps = sorted(cand.values(), key=lambda r: (r["n_steps"],
                                                        r["seed"]))
        row = by_steps[len(by_steps) // 2]
    print(f"replaying {args.program} seed={row['seed']} "
          f"({row['n_steps']} DASCOT steps)")

    from benchsuite import suite
    case = {c.name: c for c in suite()}[args.program]
    (ops, placement, routes, unstitches, num_data,
     measured) = build_llops(row, case.qasm)
    ops2, meta = LB.conjugate_frame(ops)
    program, init_letters, remap = LB.to_program(ops2, meta, num_data,
                                                 measured)
    print(f"gadget stream: {len(ops)} ops -> {len(program.ops)} mpps, "
          f"{len(routes)} routed merges, qubits {program.num_data}")
    # remap route keys (joint index) and ancilla placements to LB names
    placement_lb = {}
    for pid, rc in placement.items():
        placement_lb[pid] = rc
    last_err = None
    for orient in ("by_first_letter", "X_vertical", "X_horizontal"):
        try:
            cp, out_bit_ids, recipes = LB.build_compiled(
                program, init_letters, remap, placement_lb, d=args.d,
                orientation=orient, routes=routes,
                unstitches=unstitches)
        except Exception as e:
            print(f"orientation={orient}: {type(e).__name__}: "
                  f"{str(e)[:180]}")
            last_err = e
            continue
        det, obs = cp.circuit.compile_detector_sampler(seed=0).sample(
            args.shots, separate_observables=True)
        silent = not det.any()
        deterministic = (obs.shape[1] == 0) or bool((obs == obs[0]).all())
        S = max(args.shots, len(program.out_bits) + 128)
        raw = LB.logical_reference_bits(ops, remap, num_data, measured,
                                        program.num_data, S, 7)
        raw_masks, _ = affine_structure(raw)
        n_obs = cp.circuit.num_observables
        dim_ok = n_obs >= len(raw_masks)
        got_out = sample_program_bits(cp, shots=S, seed=29)[:, out_bit_ids]
        out_ok = (affine_structure(got_out)
                  == affine_structure(raw[:, out_bit_ids]))
        print(f"orientation={orient}: build OK, qubits="
              f"{cp.circuit.num_qubits}, p=0 detectors "
              f"{'SILENT' if silent else 'FIRED'}, observables "
              f"{'DETERMINISTIC' if deterministic else 'RANDOM'}; "
              f"observables={n_obs} vs plan dims {len(raw_masks)} "
              f"(>=: {'OK' if dim_ok else 'MISMATCH'}); "
              f"output-bit structure {'OK' if out_ok else 'MISMATCH'}; "
              f"rotations={len(cp.experiment.rotations)}")
        if not (silent and deterministic and dim_ok and out_ok):
            sys.exit(2)
        if args.measure:
            import time
            from circls.metrics.circuit_stats import circuit_stats
            cs = circuit_stats(cp.circuit)
            t0 = time.time()
            note = None
            if cp.circuit.num_observables == 0:
                stats = None
            else:
                d_ = LB._ler_with_budget(cp.circuit, args)
                stats = LB._Stats(d_) if d_ else None
                if stats is None:
                    note = "ler_budget_exceeded"
            rowo = {"record": "dascot_replay", "note": note,
                    "program": args.program, "seed": row["seed"],
                    "dascot_steps": row["n_steps"], "d": args.d,
                    "p": args.p, "orientation": orient,
                    "qubit_rounds": cs.qubit_rounds,
                    "qubits_active": cs.qubits_active,
                    "measurement_layers": cs.measurement_layers,
                    "num_observables": cs.num_observables,
                    "rotations": len(cp.experiment.rotations),
                    "decoder": stats.decoder if stats else None,
                    "shots": stats.shots if stats else 0,
                    "errors": stats.errors if stats else 0,
                    "ler": stats.logical_error_rate if stats else None,
                    "ler_bar": stats.ler_error_bar() if stats else None,
                    "seconds": round(time.time() - t0, 1)}
            out = (_ROOT / "experiments" / "results" / "best5"
                   / "dascot_replay.jsonl")
            with out.open("a") as fh:
                fh.write(json.dumps({**prov, "record": "provenance"})
                         + "\n")
                fh.write(json.dumps(rowo) + "\n")
            print("measured:", json.dumps(rowo))
        return
    raise SystemExit(f"all orientations failed: {last_err}")


if __name__ == "__main__":
    main()
