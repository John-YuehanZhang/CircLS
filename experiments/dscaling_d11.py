"""d-scaling extension: DJ-16 at distance 11, dynamic and static.

Protocol as the single-stage version: target 100 errors, PyMatching
gate, per-point seeds 11772..11775 (identical mapping), 2e6-shot cap
for the dynamic p = 5e-4 point raised to 4e6 to match the ext911
policy at d = 11; 1e6 elsewhere.  Two-stage execution: both configs
compile in parallel, then all four points sample in parallel.
Output: points_d11.jsonl (provenance first).
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
            / "points_d11.jsonl")

PROGRAMS = ["dj_16"]
CONFIGS = ["full", "static"]
DS = [11]
PS = [5e-4, 1e-3]
SEED0 = 11771


def _depth(config, d, p):
    if config == "full" and p == 5e-4:
        return (100, 4_000_000)
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
    prov = {**provenance(False), "experiment": "dscaling-d11",
            "what": "d = 11 extension of the dynamic-vs-static "
                    "d-scaling figure (DJ-16)",
            "protocol": {"target_errors": 100,
                         "max_shots": 1_000_000,
                         "max_shots_dynamic_5e-4": 4_000_000,
                         "decoder": "mwpm", "seed0": SEED0,
                         "workers": [2, 4]},
            "matrix": {"programs": PROGRAMS, "configs": CONFIGS,
                       "d": DS, "p": PS}}
    workdir = tempfile.mkdtemp(prefix="d11_circuits_")
    # 种子映射与单阶段版逐点一致: full:5e-4=+1,1e-3=+2; static:+3,+4
    seed_of = {("full", 5e-4): SEED0 + 1, ("full", 1e-3): SEED0 + 2,
               ("static", 5e-4): SEED0 + 3, ("static", 1e-3): SEED0 + 4}
    tasks = [(PROGRAMS[0], c, 11, 0, workdir) for c in CONFIGS]
    with open(OUT_FILE, "w") as f:
        f.write(json.dumps(prov) + "\n")
        f.flush()
        compiled = []
        with mp.get_context("spawn").Pool(2) as pool:
            for r in pool.imap_unordered(_compile_job, tasks):
                f.write(json.dumps(r["row"]) + "\n")
                f.flush()
                print(json.dumps(r["row"]), flush=True)
                if r.get("silent"):
                    compiled.append(r)
        points = []
        for r in compiled:
            for p in PS:
                target, cap = _depth(r["config"], 11, p)
                points.append({**{x: r[x] for x in
                                  ("circuit", "name", "config", "d",
                                   "compile_s")},
                               "p": p, "seed": seed_of[(r["config"], p)],
                               "target": target, "cap": cap})
        with mp.get_context("spawn").Pool(4) as pool:
            for row in pool.imap_unordered(_sample_job, points):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
    print("D11 DONE", flush=True)


if __name__ == "__main__":
    main()
