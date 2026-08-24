"""E3: router exact-vs-greedy gap (optimality-gap experiment).

Stage 1 (collect): monkeypatch the two corridor solvers inside
circls.core.multi_patch_coupler to record every (G, groups) instance,
then compile the medium-size merge-bearing suite programs once
(full config, d=3).  Instances deduped structurally.

Stage 2 (replay): for every instance with 3 <= k <= 9, solve both
ways on the same inputs: exact = emv_group_steiner (cost = cell
count, provably optimal), greedy = min over _legal_greedy_trees
(the five production orders).  k=2 recorded as sanity (greedy must
match exact).  Output: experiments/results/best5/optgap/
e3_router_gap.jsonl (one line per instance + summary rows per k).
"""
import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "results" / "best5" / "optgap"
OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / "e3_router_gap.jsonl"

import circls.compiler.routing as mpc                  # noqa: E402
from benchsuite import suite                       # noqa: E402
from ablation import compile_kwargs                # noqa: E402
from circls.pipeline import compile_qasm   # noqa: E402

# medium set: merge-bearing programs, n_qubits <= 40 (bounds compile time)
PREFIXES = ("deutsch", "bv", "dj", "steane", "teleport", "twistedghz")
MAX_QUBITS = 40

RECORDS = []
SEEN = set()
_cur_prog = [""]


def _rec(G, groups):
    key = (frozenset(G.nodes),
           frozenset(tuple(sorted(e)) for e in G.edges()),
           tuple(sorted(tuple(sorted(g)) for g in groups)))
    if key in SEEN:
        return
    SEEN.add(key)
    RECORDS.append((_cur_prog[0], G.copy(),
                    [list(g) for g in groups]))


_orig_emv_cand = mpc.emv_corridor_candidates
_orig_greedy = mpc._legal_greedy_trees


def _wrap_emv(G, groups, limit=24, extra_cost=None):
    _rec(G, groups)
    return _orig_emv_cand(G, groups, limit=limit, extra_cost=extra_cost)


def _wrap_greedy(G, groups):
    _rec(G, groups)
    return _orig_greedy(G, groups)


mpc.emv_corridor_candidates = _wrap_emv
mpc._legal_greedy_trees = _wrap_greedy


def main():
    if OUT_FILE.exists():
        print(f"skip: {OUT_FILE} exists", flush=True)
        return
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    cases = [c for c in suite()
             if c.name.startswith(PREFIXES) and c.n_qubits <= MAX_QUBITS]
    prov = {"record": "provenance", "experiment": "optgap-E3",
            "circls_sha": sha,
            "programs": [c.name for c in cases],
            "note": "instances recorded at the production dispatch site "
                    "(emv_corridor_candidates / _legal_greedy_trees), "
                    "structurally deduped; full config, d=3"}
    for case in cases:
        _cur_prog[0] = case.name
        t0 = time.perf_counter()
        try:
            kw = compile_kwargs("full", case)
            with contextlib.redirect_stdout(io.StringIO()):
                compile_qasm(case.qasm, **kw)
            print(f"collected {case.name}: total {len(RECORDS)} instances "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"collect FAILED {case.name}: {str(e)[:100]}", flush=True)

    # stage 2: replay
    per_k = {}
    with open(OUT_FILE, "w") as f:
        f.write(json.dumps(prov) + "\n")
        for prog, G, groups in RECORDS:
            k = len(groups)
            row = {"record": "instance", "program": prog, "k": k,
                   "n_free_tiles": G.number_of_nodes()}
            if k > 9:
                row["skip"] = "k above exact cap"
                f.write(json.dumps(row) + "\n")
                continue
            t0 = time.perf_counter()
            exact = mpc.emv_group_steiner(G, groups)
            if exact is None:
                row["skip"] = "no exact solution"
                f.write(json.dumps(row) + "\n")
                continue
            trees = _orig_greedy(G, groups)
            row["exact_s"] = round(time.perf_counter() - t0, 2)
            g_best = min((len(t) for t in trees), default=None)
            row.update(exact_cost=exact[0], greedy_cost=g_best,
                       gap_pct=(100 * (g_best / exact[0] - 1)
                                if g_best else None),
                       hit=(g_best == exact[0]) if g_best else False)
            f.write(json.dumps(row) + "\n")
            if k >= 2 and g_best is not None:
                per_k.setdefault(k, []).append(100 * (g_best / exact[0] - 1))
        for k in sorted(per_k):
            gaps = sorted(per_k[k])
            n = len(gaps)
            summ = {"record": "summary", "k": k, "n": n,
                    "hit_rate": sum(1 for g in gaps if g == 0) / n,
                    "median_gap_pct": gaps[n // 2],
                    "p90_gap_pct": gaps[min(n - 1, int(0.9 * n))],
                    "max_gap_pct": gaps[-1]}
            f.write(json.dumps(summ) + "\n")
            print("SUMMARY " + json.dumps(summ), flush=True)
    print("E3 DONE", flush=True)


if __name__ == "__main__":
    main()
