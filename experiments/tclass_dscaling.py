"""Non-Clifford distance scaling: Toffoli-3 under the Y-state
approximation, dynamic vs static, d = 3 to 11 at p = 5e-4.

Panel (b) of the distance-scaling figure: the same dynamic/static
comparison as the DJ-16 panel, on the S-proxy toffoli_n3 (7 source T
gates, 7 |Y> gadgets).  Static = liveness and first-use init both
removed, as in the Clifford runs.  Protocol: every point targets 100
failures; the dynamic p = 5e-4 points at d = 5, 7, 9 deepen to 400,
matching the six-program policy; shot caps 1e6 (d = 3..7), 2e6
(d = 9), 4e6 (d = 11), both configs.  PyMatching-only gate,
per-point deterministic seeds, raw counts.  Two-stage pools.
Output: results/tclass/points_toffoli_dscaling.jsonl.
"""
import json
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
OUT_FILE = (ROOT / "experiments" / "results" / "tclass"
            / "points_toffoli_dscaling.jsonl")

NAME = "toffoli_n3"
CONFIGS = ["full", "static"]
DS = [3, 5, 7, 9, 11]
P = 5e-4
SEED0 = 231100
COMPILE_WORKERS = 10
SAMPLE_WORKERS = 10


def _depth(config, d):
    cap = {9: 2_000_000, 11: 4_000_000}.get(d, 1_000_000)
    target = 400 if config == "full" and d in (5, 7, 9) else 100
    return target, cap


def _compile_job(task):
    config, d, seed, workdir = task
    import contextlib, io, time
    from benchsuite import tclass_suite
    from ablation import compile_kwargs
    from circls.pipeline import compile_qasm
    from circls.tools.evaluate import verify
    case = {c.name: c for c in tclass_suite()}[NAME]
    if config == "full":
        kw = compile_kwargs("full", case)
    else:
        kw = compile_kwargs("no_live", case)
        kw["first_use_init"] = False
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, distance=d, t_as_s=True, **kw)
    except Exception as e:
        return {"row": {"record": "error", "config": config, "d": d,
                        "error": f"{type(e).__name__}: {str(e)[:200]}"}}
    t_compile = round(time.time() - t0, 1)
    r = verify(cp)
    ok = bool(r.ok and r.logical is True)
    row = {"record": "precheck", "name": NAME, "config": config, "d": d,
           "qubits": cp.circuit.num_qubits,
           "silent_p0": bool(r.silent), "logical_check": bool(r.logical),
           "verify_ok": ok, "compile_s": t_compile}
    if not ok:
        return {"row": row}
    path = Path(workdir) / f"{config}_d{d}.stim"
    path.write_text(str(cp.circuit))
    return {"row": row, "circuit": str(path), "seed": seed,
            "compile_s": t_compile, "config": config, "d": d}


def _sample_job(task):
    import time
    import stim
    import pymatching as _pm
    from noise_inject import inject_uniform_noise
    from compare_tqec import adaptive_ler
    circuit = stim.Circuit(Path(task["circuit"]).read_text())
    noisy = inject_uniform_noise(circuit, P)
    target, cap = _depth(task["config"], task["d"])
    t0 = time.time()
    try:
        m = _pm.Matching.from_detector_error_model(
            noisy.detector_error_model(decompose_errors=True))
    except ValueError as e:
        return {"record": "disqualified", "config": task["config"],
                "d": task["d"], "reason": str(e)[:120]}
    shots, joint, per = adaptive_ler(noisy, m, task["seed"], target, cap)
    return {"record": "point", "name": NAME, "config": task["config"],
            "d": task["d"], "p": P, "seed": task["seed"],
            "decoder": "mwpm", "target_errors": target, "max_shots": cap,
            "compile_s": task["compile_s"], "shots": shots, "errors": joint,
            "ler": joint / shots if shots else None,
            "sample_s": round(time.time() - t0, 1)}


def main():
    from provenance import provenance
    if OUT_FILE.exists():
        print(f"skip: {OUT_FILE} exists", flush=True)
        return
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prov = {**provenance(False), "experiment": "tclass-toffoli-dscaling",
            "what": "toffoli_n3 (Y-state approximation) dynamic vs static "
                    "at d = 3..11, p = 5e-4 — panel (b) of fig:dscaling",
            "protocol": {"p": P, "target_errors": 100,
                         "deep_dynamic": {"d": [5, 7, 9], "target": 400},
                         "caps": {"default": 1_000_000, "d9": 2_000_000,
                                  "d11": 4_000_000},
                         "decoder": "mwpm", "seed0": SEED0}}
    workdir = tempfile.mkdtemp(prefix="toffoli_dsc_")
    tasks = []
    for i, (config, d) in enumerate((c, d) for c in CONFIGS for d in DS):
        tasks.append((config, d, SEED0 + i, workdir))
    with open(OUT_FILE, "w") as f:
        f.write(json.dumps(prov) + "\n")
        f.flush()
        compiled = []
        with mp.get_context("spawn").Pool(COMPILE_WORKERS) as pool:
            for r in pool.imap_unordered(_compile_job, tasks):
                f.write(json.dumps(r["row"]) + "\n")
                f.flush()
                print(json.dumps(r["row"]), flush=True)
                if r.get("circuit"):
                    compiled.append(r)
        with mp.get_context("spawn").Pool(SAMPLE_WORKERS) as pool:
            for row in pool.imap_unordered(_sample_job, compiled):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
    print("TOFFOLI DSCALING DONE", flush=True)


if __name__ == "__main__":
    main()
