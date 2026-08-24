"""Static (deterministic) Table-2 stats for the in-flight rows.

Reuses compare_topols' exact code paths: side_metrics on both circuits,
experiment_stats for V1, identical compile settings.  No sampling, so
this does not disturb the running LER jobs.  Output is scratch; the
official compare_topols rows supersede/confirm it when they land.
"""
import contextlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import stim

from compare_topols import CIRC_DIR, matching_qasm, side_metrics

PAIRS = [("c45_ghz_n40", "c45_ghz_n40_k1_f0", 3),
         ("c45_steane_encode", "c45_steane_encode_k2_f0", 5),
         ("c45_cat_n130", "c45_cat_n130_k1_f0", 3)]

manifest = json.loads((CIRC_DIR / "manifest.json").read_text())
out = open(Path(__file__).parent / "results/static_stats.jsonl", "w")
for name, key, d in PAIRS:
    from circls.pipeline import compile_qasm
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict
    entry = manifest["entries"][key]
    base = stim.Circuit((CIRC_DIR / entry["file"]).read_text())
    qasm = matching_qasm(name, entry["fill"])
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(qasm, distance=d, assignment="optimized",
                              liveness=True, keep_patches=set())
        times.append(round(time.perf_counter() - t0, 3))
    row = {"pair": key, "name": name, "d": d,
           "topols": side_metrics(base), "topols_cubes": entry["cubes"],
           "ours": side_metrics(cp.circuit),
           "ours_stats": to_dict(experiment_stats(cp.experiment, cp.circuit)),
           "compile_seconds": times, "compile_median": sorted(times)[1]}
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(name, "done", flush=True)
print("STATIC DONE", flush=True)
