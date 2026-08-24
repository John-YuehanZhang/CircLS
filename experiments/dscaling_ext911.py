"""d-scaling extension: the remaining five programs at d = 9 and 11.

Fills tab:dscaling's d = 9, 11 cells for teleport_4, teleport_8,
bv_8, bv_16, dj_8 (dj_16 runs separately in points_d9 / points_d11).
Protocol as before: PyMatching-only gate, per-task deterministic
seeds (identical formula to the single-stage version), raw counts.
Depth: dynamic 5e-4 targets 400 errors under 2e6 shots at d = 9 and
100 errors under 4e6 shots at d = 11; every other point targets 100
errors under 1e6 shots.

Two-stage execution to use the machine: stage 1 compiles all 20
(program, config, d) jobs in parallel and dumps circuit text to a
machine-local temp dir; stage 2 samples all points in parallel.
Output: points_ext911.jsonl (provenance first; rows stream in
completion order, keys make each row self-identifying).
"""
import json
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
OUT_FILE = (ROOT / "experiments" / "results" / "best5" / "dscaling"
            / "points_ext911.jsonl")

PROGRAMS = ["teleport_4", "teleport_8", "bv_8", "bv_16", "dj_8"]
CONFIGS = ["full", "static"]
DS = [9, 11]
PS = [5e-4, 1e-3]
SEED0 = 91100
COMPILE_WORKERS = 20
SAMPLE_WORKERS = 40


def _depth(config, d, p):
    # review 2026-08-20: the raised caps must apply to BOTH arms — the
    # original full-only gate starved the static arm at d = 11 (22..71
    # failures); the shipped points were re-sampled by dscaling_deepen.py
    if p == 5e-4 and d == 9:
        return (400 if config == "full" else 100, 2_000_000)
    if p == 5e-4 and d == 11:
        return (100, 8_000_000)
    return (100, 1_000_000)


def _compile_job(task):
    """Stage 1: compile one (program, config, d); dump circuit text."""
    name, config, d, seed_base, workdir = task
    import contextlib, io, time
    from benchsuite import suite
    from ablation import compile_kwargs
    from circls.pipeline import compile_qasm
    case = next(c for c in suite() if c.name == name)
    if config == "full":
        kw = compile_kwargs("full", case)
    else:
        kw = compile_kwargs("no_live", case)
        kw["first_use_init"] = False
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, distance=d, **kw)
    except Exception as e:
        return {"row": {"record": "error", "name": name, "config": config,
                        "d": d, "error": str(e)[:200]}}
    t_compile = round(time.time() - t0, 1)
    det, _ = cp.circuit.compile_detector_sampler(seed=0).sample(
        128, separate_observables=True)
    silent = bool(not det.any())
    path = Path(workdir) / f"{name}_{config}_d{d}.stim"
    path.write_text(str(cp.circuit))
    return {"row": {"record": "precheck", "name": name, "config": config,
                    "d": d, "silent_p0": silent, "compile_s": t_compile},
            "circuit": str(path), "silent": silent,
            "seed_base": seed_base, "compile_s": t_compile,
            "name": name, "config": config, "d": d}


def _sample_job(task):
    """Stage 2: one (program, config, d, p) point."""
    import time
    import stim
    import pymatching as _pm
    from noise_inject import inject_uniform_noise
    from compare_tqec import adaptive_ler
    circuit = stim.Circuit(Path(task["circuit"]).read_text())
    noisy = inject_uniform_noise(circuit, task["p"])
    t0 = time.time()
    try:
        m = _pm.Matching.from_detector_error_model(
            noisy.detector_error_model(decompose_errors=True))
    except ValueError as e:
        return {"record": "disqualified", "name": task["name"],
                "config": task["config"], "d": task["d"], "p": task["p"],
                "reason": str(e)[:120]}
    shots, joint, per = adaptive_ler(
        noisy, m, task["seed"], task["target"], task["cap"])
    return {"record": "point", "name": task["name"],
            "config": task["config"], "d": task["d"], "p": task["p"],
            "seed": task["seed"], "decoder": "mwpm",
            "target_errors": task["target"], "max_shots": task["cap"],
            "silent_p0": True, "compile_s": task["compile_s"],
            "shots": shots, "errors": joint,
            "ler": joint / shots if shots else None,
            "sample_s": round(time.time() - t0, 1)}


def main():
    from provenance import provenance
    if OUT_FILE.exists():
        print(f"skip: {OUT_FILE} exists", flush=True)
        return
    prov = {**provenance(False), "experiment": "dscaling-ext911",
            "what": "d = 9, 11 cells of tab:dscaling for the five "
                    "programs outside the DJ-16 figure runs",
            "protocol": {"target_errors": 100,
                         "deep_dynamic_5e-4": {"d9": [400, 2_000_000],
                                               "d11": [100, 4_000_000]},
                         "max_shots": 1_000_000, "decoder": "mwpm",
                         "seed0": SEED0,
                         "workers": [COMPILE_WORKERS, SAMPLE_WORKERS]},
            "matrix": {"programs": PROGRAMS, "configs": CONFIGS,
                       "d": DS, "p": PS}}
    workdir = tempfile.mkdtemp(prefix="ext911_circuits_")
    tasks = []
    for i, (name, config, d) in enumerate(
            (n, c, d) for n in PROGRAMS for c in CONFIGS for d in DS):
        tasks.append((name, config, d, SEED0 + 10 * i, workdir))
    with open(OUT_FILE, "w") as f:
        f.write(json.dumps(prov) + "\n")
        f.flush()
        compiled = []
        with mp.get_context("spawn").Pool(COMPILE_WORKERS) as pool:
            for r in pool.imap_unordered(_compile_job, tasks):
                f.write(json.dumps(r["row"]) + "\n")
                f.flush()
                print(json.dumps(r["row"]), flush=True)
                if r.get("silent"):
                    compiled.append(r)
        points = []
        for r in compiled:
            for k, p in enumerate(PS):
                target, cap = _depth(r["config"], r["d"], p)
                points.append({**{x: r[x] for x in
                                  ("circuit", "name", "config", "d",
                                   "compile_s")},
                               "p": p, "seed": r["seed_base"] + k,
                               "target": target, "cap": cap})
        with mp.get_context("spawn").Pool(SAMPLE_WORKERS) as pool:
            for row in pool.imap_unordered(_sample_job, points):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
    print("EXT911 DONE", flush=True)


if __name__ == "__main__":
    main()
