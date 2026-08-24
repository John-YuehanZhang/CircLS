"""d-scaling supplement: the fully static configuration.

static = full pipeline minus BOTH lifetime-allocation knobs
(first_use_init off + liveness/last-use freeing off); placement,
re-selection, scheduling, parallel all stay on, so the pair
(full, static) reads as dynamic vs static allocation.

Same protocol as points_ext2 (target 100 errors / 1e6 shots,
PyMatching-only gate, per-point seeds), six programs x d=3,5,7 x
p=5e-4,1e-3.  Output: points_static.jsonl (provenance first).
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
OUT_FILE = ROOT / "experiments" / "results" / "best5" / "dscaling" / "points_static.jsonl"

from benchsuite import suite                      # noqa: E402
from ablation import compile_kwargs               # noqa: E402
from circls.pipeline import compile_qasm  # noqa: E402
from noise_inject import inject_uniform_noise     # noqa: E402
from compare_tqec import adaptive_ler             # noqa: E402

TARGET, MAX_SHOTS = 100, 1_000_000
PROGRAMS = ["teleport_4", "teleport_8", "bv_8", "bv_16", "dj_8", "dj_16"]
DS = [3, 5, 7]
PS = [5e-4, 1e-3]
SEED0 = 771


def static_kwargs(case):
    # no_live turns liveness off; on top of it first_use_init goes off
    kw = compile_kwargs("no_live", case)
    kw["first_use_init"] = False
    return kw


def main():
    import stim, pymatching, numpy
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    prov = {"record": "provenance", "experiment": "dscaling-static",
            "what": "fully static allocation (no fui, no freeing) for the "
                    "dynamic-vs-static d-scaling figure",
            "circls_sha": sha,
            "versions": {"python": sys.version.split()[0],
                         "stim": stim.__version__,
                         "pymatching": pymatching.__version__,
                         "numpy": numpy.__version__},
            "protocol": {"target_errors": TARGET, "max_shots": MAX_SHOTS,
                         "decoder": "mwpm", "seed0": SEED0,
                         "gate": "all 3 circuits must decompose for "
                                 "PyMatching, else program flagged"},
            "matrix": {"programs": PROGRAMS, "configs": ["static"],
                       "d": DS, "p": PS}}
    cases = {c.name: c for c in suite()}
    idx = 0
    with open(OUT_FILE, "w") as f:
        f.write(json.dumps(prov) + "\n")
        f.flush()
        for name in PROGRAMS:
            built = {}
            ok = True
            for d in DS:
                case = cases[name]
                kw = static_kwargs(case)
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    cp = compile_qasm(case.qasm, distance=d, **kw)
                t_compile = round(time.perf_counter() - t0, 1)
                det, _ = cp.circuit.compile_detector_sampler(
                    seed=0).sample(128, separate_observables=True)
                silent = bool(not det.any())
                try:
                    import pymatching as _pm
                    probe = inject_uniform_noise(cp.circuit, 5e-4)
                    _pm.Matching.from_detector_error_model(
                        probe.detector_error_model(decompose_errors=True))
                    dec = "mwpm"
                except ValueError as e:
                    if "Failed to decompose" not in str(e):
                        raise
                    dec = "mwpf-required"
                    ok = False
                row = {"record": "precheck", "name": name,
                       "config": "static", "d": d, "decoder": dec,
                       "silent_p0": silent, "compile_s": t_compile}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
                built[d] = (cp.circuit, t_compile, silent)
            if not ok:
                f.write(json.dumps({"record": "disqualified", "name": name,
                                    "reason": "mwpf required"}) + "\n")
                f.flush()
                print(f"{name} DISQUALIFIED", flush=True)
                continue
            for d in DS:
                circuit, t_compile, silent = built[d]
                for p in PS:
                    idx += 1
                    seed = SEED0 + idx
                    noisy = inject_uniform_noise(circuit, p)
                    t0 = time.perf_counter()
                    import pymatching as _pm
                    m = _pm.Matching.from_detector_error_model(
                        noisy.detector_error_model(decompose_errors=True))
                    shots, joint, per = adaptive_ler(
                        noisy, m, seed, TARGET, MAX_SHOTS)
                    row = {"record": "point", "name": name,
                           "config": "static", "d": d, "p": p,
                           "seed": seed, "decoder": "mwpm",
                           "silent_p0": silent, "compile_s": t_compile,
                           "shots": shots, "errors": joint,
                           "ler": joint / shots if shots else None,
                           "sample_s": round(time.perf_counter() - t0, 1)}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(json.dumps(row), flush=True)
    print("STATIC DONE", flush=True)


if __name__ == "__main__":
    main()
