import contextlib, io, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]     # the repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

# serial, uncontended per-compile timing (compile-time is wall-clock; parallel load inflates it)
PAIRS = [("c45_qrng_n4", "c45_qrng_n4_k1_f0", 3, "qrng_n4", 0.02),
         ("c45_cat_state_n4", "c45_cat_state_n4_k1_f0", 3, "cat_state_n4", 0.02),
         ("c45_cat_state_n22", "c45_cat_state_n22_k1_f0", 3, "cat_state_n22", 0.9),
         ("c45_cat_n35", "c45_cat_n35_k1_mXX", 3, "cat_n35", 2.8),
         ("c45_cat_n130", "c45_cat_n130_k1_f0", 3, "cat_n130", 53.3),
         ("c45_ghz_n40", "c45_ghz_n40_k1_f0", 3, "ghz_n40", 1.1),
         ("c45_ghz_n78", "c45_ghz_n78_k1_f0", 3, "ghz_n78", 792.3),
         ("c45_steane_encode", "c45_steane_encode_k2_f0", 5, "Steane", 1.0)]


def main():
    from compare_topols import CIRC_DIR, matching_qasm
    from circls.pipeline import compile_qasm
    manifest = json.loads((CIRC_DIR / "manifest.json").read_text())["entries"]
    out = open(SP / "p1_compile_times.jsonl", "w")
    for name, key, d, label, pub in PAIRS:
        qasm = matching_qasm(name, manifest[key]["fill"])
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                compile_qasm(qasm, distance=d, assignment="optimized", liveness=True, keep_patches=set())
            times.append(round(time.perf_counter() - t0, 3))
        med = sorted(times)[1]
        row = {"label": label, "name": name, "pair": key, "d": d,
               "compile_seconds": times, "compile_median": med, "published": pub}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"{label:14s} median={med:8.3f}s  (published {pub})", flush=True)
    print("P1 DONE", flush=True)


if __name__ == "__main__":
    main()
