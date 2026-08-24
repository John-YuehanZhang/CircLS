"""Nearest-first vs the four fixed greedy orders (nearest-first.txt).

Two phases in one script:

  collect   compile a sweep of cases with _legal_greedy_trees
            instrumented; every real invocation above the EMV cap dumps
            (edges, groups, case, step tag) to instances.jsonl.  The
            default router is UNCHANGED — the wrapper only records.

  compare   replay every instance: best-of-4 fixed orders (the shipped
            candidate set) vs the Prim-style nearest-first tree.
            Metric: corridor tree size in cells (the router's primary
            cost; every cell is held for d rounds).  Reports
            win/tie/loss and the best-of-5 gain.

Usage:
    python experiments/nearest_first_compare.py [--quick]
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

OUT = Path("experiments/results/nearest_first")


def collect(quick=False):
    import networkx as nx  # noqa: F401
    from benchsuite import suite, bv, dj
    import circls.compiler.routing as M
    from circls.pipeline import compile_qasm

    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(OUT / "instances.jsonl", "w")
    ctx = {"case": "?", "n": 0}
    orig = M._legal_greedy_trees

    def spy(G, groups):
        ctx["n"] += 1
        fh.write(json.dumps({
            "case": ctx["case"], "idx": ctx["n"],
            "nodes": sorted(map(list, G.nodes())),
            "edges": sorted([sorted(map(list, e)) for e in G.edges()]),
            "groups": [sorted(map(list, g)) for g in groups]}) + "\n")
        fh.flush()
        return orig(G, groups)

    M._legal_greedy_trees = spy
    # the k>9 population: dj's X^n hub steps survive re-selection; add
    # no-reduction bv/dj compiles where the raw hub chain routes big steps
    cases = []
    for c in suite():
        if c.name.startswith("dj_") or c.name in ("ghz_16_mixed",):
            cases.append((c.name, c.qasm, {}))
    for n in ((16, 32) if quick else (16, 32, 64)):
        cases.append((f"bv_{n}_noreduce", bv(n),
                      {"measure_reduction": False}))
        cases.append((f"dj_{n}_noreduce", dj(n),
                      {"measure_reduction": False}))
    if quick:
        cases = [c for c in cases if "64" not in c[0] and "100" not in c[0]]
    import signal

    def _boom(sig, frm):
        raise TimeoutError("collect budget")

    signal.signal(signal.SIGALRM, _boom)
    for name, qasm, kw in cases:
        ctx["case"] = name
        before = ctx["n"]
        signal.alarm(900)   # instances stream to disk BEFORE a hang, so a
        try:                # timed-out compile still contributes its steps
            with contextlib.redirect_stdout(io.StringIO()):
                compile_qasm(qasm, assignment="optimized", liveness=True,
                             keep_patches=set(), **kw)
            status = "OK"
        except Exception as e:
            status = type(e).__name__
        finally:
            signal.alarm(0)
        print(f"[{status:>14s}] {name}: {ctx['n'] - before} instances",
              flush=True)
    fh.close()
    M._legal_greedy_trees = orig


def compare():
    import networkx as nx
    from circls.compiler.routing import (_greedy_connect,
                                         _nearest_first_tree)
    rows = []
    for line in open(OUT / "instances.jsonl"):
        inst = json.loads(line)
        G = nx.Graph()
        G.add_nodes_from(tuple(c) for c in inst["nodes"])
        G.add_edges_from((tuple(a), tuple(b)) for a, b in inst["edges"])
        groups = [set(map(tuple, g)) for g in inst["groups"]]
        idx = list(range(len(groups)))
        orders = [idx, idx[::-1],
                  sorted(idx, key=lambda i: (len(groups[i]),
                                             min(groups[i]))),
                  sorted(idx, key=lambda i: min(groups[i]))]
        four = [t for t in (_greedy_connect(G, groups, s) for s in orders)
                if t is not None]
        nf = _nearest_first_tree(G, groups)
        best4 = min((len(t) for t in four), default=None)
        nfl = len(nf) if nf is not None else None
        rows.append({"case": inst["case"], "idx": inst["idx"],
                     "k": len(groups), "best4": best4, "nearest": nfl})
    with open(OUT / "compare.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    win = sum(1 for r in rows if r["nearest"] is not None
              and r["best4"] is not None and r["nearest"] < r["best4"])
    tie = sum(1 for r in rows if r["nearest"] == r["best4"])
    loss = sum(1 for r in rows if r["nearest"] is not None
               and r["best4"] is not None and r["nearest"] > r["best4"])
    only4 = sum(1 for r in rows if r["nearest"] is None
                and r["best4"] is not None)
    onlynf = sum(1 for r in rows if r["best4"] is None
                 and r["nearest"] is not None)
    cells4 = sum(r["best4"] for r in rows if r["best4"] is not None
                 and r["nearest"] is not None)
    cells5 = sum(min(r["best4"], r["nearest"]) for r in rows
                 if r["best4"] is not None and r["nearest"] is not None)
    md = ["# nearest-first vs best-of-4 fixed orders", "",
          f"instances: {len(rows)}",
          f"nearest wins: {win}   ties: {tie}   losses: {loss}   "
          f"only-4-solves: {only4}   only-nf-solves: {onlynf}",
          f"total corridor cells (both solve): best-of-4 {cells4}, "
          f"best-of-5 {cells5} "
          f"({(1 - cells5 / cells4) * 100 if cells4 else 0:+.2f}% saved)",
          "", "| case | idx | k | best4 | nearest |", "|---|---|---|---|---|"]
    for r in rows:
        mark = ""
        if r["nearest"] is not None and r["best4"] is not None:
            mark = (" **win**" if r["nearest"] < r["best4"] else
                    (" loss" if r["nearest"] > r["best4"] else ""))
        md.append(f"| {r['case']} | {r['idx']} | {r['k']} | {r['best4']} |"
                  f" {r['nearest']}{mark} |")
    (OUT / "compare.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:8]))
    print(f"-> {OUT}/compare.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-collect", action="store_true")
    args = ap.parse_args()
    if not args.skip_collect:
        collect(quick=args.quick)
    compare()


if __name__ == "__main__":
    main()
