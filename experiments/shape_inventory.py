"""Shape-coverage inventory for the distance-evidence audit.

For every suite benchmark (full config, d=3): compile, then read the
construction shapes out of the compiled artifacts themselves -- no
instrumentation, so the inventory describes exactly what ships in the
circuit:

  - per-step check classes from the joint layout: (weight, kf?, mixed?)
    counters cover plain seams (no kf), uniform-domino walls (kf, one
    letter), mixed-domino walls (kf, two letters), weight-3 corner
    plaquettes and the weight-4 bulk;
  - recoloured regions (layout.retyped) and orientation mixing
    (layout.domains);
  - corridor topology from SubsetRoute.tree: straight cells, bends,
    branches, leaves; convex corner cuts (route.cut / how);
  - freed-tile reuse: corridor cells that overlap cells of patches
    already retired at that step;
  - program level: |Y> gadget count, parallel windows (batches with
    more than one step).

Output: one JSONL row per benchmark plus a stdout summary.

Usage:
    python experiments/shape_inventory.py [--d 3] [--workers 8] \
        [--out experiments/results/best5/shape_inventory.jsonl]
"""
import argparse
import collections
import contextlib
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))


def _tree_topology(tree):
    """(straight, bend, leaf, branch) counts over the corridor cells."""
    cells = set(tree)
    straight = bend = leaf = branch = 0
    for c in cells:
        nbs = [n for n in ((c[0] + 2, c[1]), (c[0] - 2, c[1]),
                           (c[0], c[1] + 2), (c[0], c[1] - 2)) if n in cells]
        if len(nbs) <= 1:
            leaf += 1
        elif len(nbs) >= 3:
            branch += 1
        elif nbs[0][0] == nbs[1][0] or nbs[0][1] == nbs[1][1]:
            straight += 1
        else:
            bend += 1
    return straight, bend, leaf, branch


def _one_case(task):
    name, d = task
    import ablation
    from benchsuite import suite
    from circls.pipeline import compile_qasm

    case = {c.name: c for c in suite()}[name]
    row = {"record": "shapes", "name": name, "n": case.n_qubits, "d": d}
    t0 = time.perf_counter()
    try:
        kw = ablation.compile_kwargs("full", case)
        kw["distance"] = d
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, **kw)
        exp = cp.experiment
        routes = getattr(exp, "_routes", []) or []
        checks = collections.Counter()
        topo = collections.Counter()
        retyped_cells = 0
        mixed_orient_steps = 0
        cuts = walls = reuse_cells = reuse_steps = 0
        patch_cells = {}
        for r in routes:
            if r is None or not getattr(r, "ok", False):
                continue
            for nm, cells in r.placed.items():
                patch_cells.setdefault(nm, frozenset(cells))
        lifetimes = getattr(exp, "lifetimes", {})
        keep = getattr(exp, "keep_patches", set())
        routed_steps = 0
        for i, r in enumerate(routes):
            if r is None or not getattr(r, "ok", False):
                continue
            routed_steps += 1
            lay = r.layout
            for c in lay.checks:
                pauli = c.get("pauli", {})
                letters = set(pauli.values())
                checks[(len(pauli), bool(c.get("kf")),
                        "mixed" if len(letters) > 1 else "pure")] += 1
            retyped_cells += len(getattr(lay, "retyped", ()) or ())
            if len(set(getattr(lay, "domains", {}).values())) > 1:
                mixed_orient_steps += 1
            s, b, le, br = _tree_topology(r.tree)
            topo["straight"] += s
            topo["bend"] += b
            topo["leaf"] += le
            topo["branch"] += br
            cuts += len(r.cut or ())
            walls += r.n_walls or 0
            dead = {nm for nm, (f, l) in lifetimes.items()
                    if l < i and nm not in keep}
            hit = r.tree & set().union(*(patch_cells.get(nm, frozenset())
                                         for nm in dead)) if dead else set()
            if hit:
                reuse_steps += 1
                reuse_cells += len(hit)
        row.update({
            "status": "OK",
            "steps": len(routes),
            "routed_steps": routed_steps,
            "checks": {f"w{w}_{'kf' if kf else 'plain'}_{m}": v
                       for (w, kf, m), v in sorted(checks.items())},
            "topology": dict(topo),
            "convex_cuts": cuts,
            "stretched_walls": walls,
            "retyped_cells": retyped_cells,
            "mixed_orientation_steps": mixed_orient_steps,
            "reuse_steps": reuse_steps,
            "reuse_cells": reuse_cells,
            "gadgets": len(cp.program.gadgets),
            "parallel_windows": sum(
                1 for b in getattr(exp, "_step_batches", []) if len(b) > 1),
            "searched": case.n_qubits <= 10,
        })
    except Exception as e:
        row.update({"status": type(e).__name__, "error": str(e)[:200]})
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default=str(
        _ROOT / "experiments/results/best5/shape_inventory.jsonl"))
    args = ap.parse_args()

    from benchsuite import suite
    from provenance import provenance

    names = [c.name for c in suite()
             if args.only is None or c.name in args.only]
    tasks = [(nm, args.d) for nm in names]
    prov = provenance(args.allow_dirty)   # before the output file exists:
    # an untracked out-file would itself dirty the tree (chicken-and-egg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh:
        fh.write(json.dumps({**prov, "record": "provenance",
                             "argv": sys.argv}) + "\n")
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for row in pool.imap_unordered(_one_case, tasks):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                tag = (f"OK steps={row.get('routed_steps')}"
                       if row.get("status") == "OK"
                       else row.get("status"))
                print(f"{row['name']:20s} {tag} ({row['seconds']}s)",
                      flush=True)


if __name__ == "__main__":
    main()
