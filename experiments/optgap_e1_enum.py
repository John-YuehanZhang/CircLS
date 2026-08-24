"""E1: exhaustive placement enumeration (optimality-gap experiment).

Per program: one baseline compile with assignment='optimized' (the
pipeline's own choice), then every ordered placement of the patches
onto the mapper's odd-odd slot set, each replayed through the full
pipeline with placement= forced (assignment dropped: the injected
placement replaces it).  Metric: allocated volume (V1).  No symmetry
reduction: 90-degree rotations flip X/Z boundary orientation, so only
brute force is provably safe.  96 workers.

Output: one jsonl per program under
experiments/results/best5/optgap/e1_<name>.jsonl
  line 1: provenance
  line 2: baseline (CircLS's own placement + volume)
  then one line per enumerated placement {i, v} (+rounds r)
  last line: summary {min, circls, gap, rank, n_evals, n_failed}
"""
import io
import itertools
import json
import contextlib
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "results" / "best5" / "optgap"
OUT.mkdir(parents=True, exist_ok=True)

PROGRAMS = ["twistedghz_4", "steane_encode", "bv_8", "dj_8", "teleport_4"]
WORKERS = 96

_G = {}


def _init(qasm, kw, names):
    _G["qasm"], _G["kw"], _G["names"] = qasm, kw, names


def _eval(task):
    i, slots = task
    try:
        from circls.pipeline import compile_qasm
        from circls.metrics import experiment_stats
        from circls.metrics.report import to_dict
        forced = dict(zip(_G["names"], slots))
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(_G["qasm"], placement=forced, **_G["kw"])
        st = to_dict(experiment_stats(cp.experiment, cp.circuit))
        return (i, st["V1_volume_blocks"], st["T1_rounds"], None)
    except Exception as e:  # record, never abort the sweep
        return (i, None, None, str(e)[:120])


def main():
    from benchsuite import suite
    from ablation import compile_kwargs
    from circls.pipeline import compile_qasm
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    cases = {c.name: c for c in suite()}
    for name in PROGRAMS:
        out_path = OUT / f"e1_{name}.jsonl"
        if out_path.exists():
            print(f"skip {name}: {out_path} exists", flush=True)
            continue
        case = cases[name]
        kw = compile_kwargs("full", case)
        with contextlib.redirect_stdout(io.StringIO()):
            cp0 = compile_qasm(case.qasm, **kw)
        st0 = to_dict(experiment_stats(cp0.experiment, cp0.circuit))
        pnames = sorted(cp0.placement.keys())
        xs = sorted({p[0] for p in cp0.placement.values()})
        ys = sorted({p[1] for p in cp0.placement.values()})
        slots = [(x, y) for x in xs for y in ys]
        kw_forced = dict(kw)
        kw_forced.pop("assignment", None)
        perms = list(itertools.permutations(slots, len(pnames)))
        prov = {"record": "provenance", "experiment": "optgap-E1",
                "circls_sha": sha, "name": name, "slots": slots,
                "patch_names": pnames, "n_perms": len(perms),
                "workers": WORKERS, "note":
                "domain = odd-odd slots spanned by the pipeline's own "
                "placement; no symmetry reduction (orientation unsafe)"}
        base = {"record": "baseline",
                "placement": {k: list(v) for k, v in cp0.placement.items()},
                "volume": st0["V1_volume_blocks"],
                "rounds": st0["T1_rounds"]}
        t0 = time.time()
        n_fail = 0
        best = None
        vols = []
        with open(out_path, "w") as f:
            f.write(json.dumps(prov) + "\n")
            f.write(json.dumps(base) + "\n")
            with mp.get_context("spawn").Pool(
                    WORKERS, initializer=_init,
                    initargs=(case.qasm, kw_forced, pnames),
                    maxtasksperchild=400) as pool:
                for i, v, r, err in pool.imap_unordered(
                        _eval, enumerate(perms), chunksize=8):
                    if err is not None:
                        n_fail += 1
                        f.write(json.dumps({"i": i, "err": err}) + "\n")
                        continue
                    vols.append(v)
                    if best is None or v < best[1]:
                        best = (i, v)
                    f.write(json.dumps({"i": i, "v": v, "r": r}) + "\n")
                    if len(vols) % 20000 == 0:
                        el = time.time() - t0
                        print(f"{name}: {len(vols)}/{len(perms)} "
                              f"{el/60:.0f}min best={best[1]:.1f}",
                              flush=True)
            v_c = st0["V1_volume_blocks"]
            rank = sum(1 for v in vols if v < v_c)
            summ = {"record": "summary", "name": name,
                    "n_evals": len(vols), "n_failed": n_fail,
                    "min_volume": best[1] if best else None,
                    "argmin_index": best[0] if best else None,
                    "argmin_placement": ({k: list(s) for k, s in
                                          zip(pnames, perms[best[0]])}
                                         if best else None),
                    "circls_volume": v_c,
                    "gap_pct": (100 * (v_c / best[1] - 1)) if best else None,
                    "n_strictly_better": rank,
                    "seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(summ) + "\n")
        print("SUMMARY " + json.dumps(summ), flush=True)
    print("E1 DONE", flush=True)


if __name__ == "__main__":
    main()
