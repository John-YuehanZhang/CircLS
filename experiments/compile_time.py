"""T4 backfill for tab:comparison: timed recompiles of the matched instances.

The comparison harness never recorded T4_compile_seconds (the stats
snapshot is taken from an already-compiled experiment), so the paper's
compile-time column for our side comes from this dedicated run: the same
matched QASM (manifest fill -> prologue/epilogue), same compile_qasm
settings, wall-clock around the compile call only.  Three repetitions,
median reported.
"""
import contextlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

from compare_topols import CIRC_DIR, matching_qasm   # noqa: E402
from provenance import provenance                    # noqa: E402

PAIRS = [
    ("ghz_16", "ghz_16_k1_f1", 3),
    ("ghz_16", "ghz_16_k2_f1", 5),
    ("bv_16", "bv_16_k2_f0", 5),
    ("CNOT", "CNOT_k1_f0", 3),
    ("CNOT", "CNOT_k2_f0", 5),
]
REPS = 3


def main():
    from circls.pipeline import compile_qasm

    prov = provenance()
    manifest = json.loads((CIRC_DIR / "manifest.json").read_text())
    out = _ROOT / "experiments/results/best5/topols_compare/compile_time.jsonl"
    fh = open(out, "a")
    fh.write(json.dumps({**prov, "record": "provenance", "reps": REPS}) + "\n")
    fh.flush()

    for name, key, d in PAIRS:
        entry = manifest["entries"][key]
        qasm = matching_qasm(name, entry["fill"])
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                compile_qasm(qasm, distance=d, assignment="optimized",
                             liveness=True, keep_patches=set())
            times.append(round(time.perf_counter() - t0, 3))
        row = {"record": "point", "pair": key, "name": name, "d": d,
               "seconds": times, "median": statistics.median(times)}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(f"{key} d={d}: {times} median={row['median']}", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
