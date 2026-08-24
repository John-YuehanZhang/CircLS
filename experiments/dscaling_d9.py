"""d-scaling extension: DJ-16 at distance 9, dynamic and static.

Extends the d = 3,5,7 grid of the d-scaling figure to d = 9 for the
figure's program (DJ-16).  Protocol matches dscaling_static.py /
points_ext2: target 100 errors or 1e6 shots (the dynamic p = 5e-4
point deepens to 400 errors, matching the deep-sampling policy of the
existing dynamic points), PyMatching-only gate, per-point seeds.
Output: points_d9.jsonl (provenance first).
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
OUT_FILE = ROOT / "experiments" / "results" / "best5" / "dscaling" / "points_d9.jsonl"

from benchsuite import suite                      # noqa: E402
from ablation import compile_kwargs               # noqa: E402
from circls.pipeline import compile_qasm          # noqa: E402
from noise_inject import inject_uniform_noise     # noqa: E402
from compare_tqec import adaptive_ler             # noqa: E402

TARGET, DEEP_TARGET, MAX_SHOTS = 100, 400, 1_000_000
PROGRAMS = ["dj_16"]
D = 9
PS = [5e-4, 1e-3]
SEED0 = 9771


def config_kwargs(config, case):
    if config == "full":
        return compile_kwargs("full", case)
    kw = compile_kwargs("no_live", case)
    kw["first_use_init"] = False
    return kw


def main():
    from provenance import provenance
    if OUT_FILE.exists():
        print(f"skip: {OUT_FILE} exists", flush=True)
        return
    prov = {**provenance(False), "experiment": "dscaling-d9",
            "what": "d = 9 extension of the dynamic-vs-static d-scaling "
                    "figure (DJ-16)",
            "protocol": {"target_errors": TARGET,
                         "deep_target_dynamic_5e-4": DEEP_TARGET,
                         "max_shots": MAX_SHOTS, "decoder": "mwpm",
                         "seed0": SEED0},
            "matrix": {"programs": PROGRAMS,
                       "configs": ["full", "static"], "d": [D], "p": PS}}
    cases = {c.name: c for c in suite()}
    idx = 0
    with open(OUT_FILE, "w") as f:
        f.write(json.dumps(prov) + "\n")
        f.flush()
        for name in PROGRAMS:
            for config in ("full", "static"):
                case = cases[name]
                kw = config_kwargs(config, case)
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    cp = compile_qasm(case.qasm, distance=D, **kw)
                t_compile = round(time.perf_counter() - t0, 1)
                det, _ = cp.circuit.compile_detector_sampler(
                    seed=0).sample(128, separate_observables=True)
                silent = bool(not det.any())
                row = {"record": "precheck", "name": name,
                       "config": config, "d": D, "silent_p0": silent,
                       "compile_s": t_compile}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
                assert silent, f"{name}/{config} d={D} not silent at p=0"
                for p in PS:
                    idx += 1
                    seed = SEED0 + idx
                    target = (DEEP_TARGET if config == "full"
                              and p == 5e-4 else TARGET)
                    noisy = inject_uniform_noise(cp.circuit, p)
                    t0 = time.perf_counter()
                    import pymatching as _pm
                    m = _pm.Matching.from_detector_error_model(
                        noisy.detector_error_model(decompose_errors=True))
                    shots, joint, per = adaptive_ler(
                        noisy, m, seed, target, MAX_SHOTS)
                    row = {"record": "point", "name": name,
                           "config": config, "d": D, "p": p,
                           "seed": seed, "decoder": "mwpm",
                           "target_errors": target,
                           "silent_p0": silent, "compile_s": t_compile,
                           "shots": shots, "errors": joint,
                           "ler": joint / shots if shots else None,
                           "sample_s": round(time.perf_counter() - t0, 1)}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(json.dumps(row), flush=True)
    print("D9 DONE", flush=True)


if __name__ == "__main__":
    main()
