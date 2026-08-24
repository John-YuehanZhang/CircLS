"""T-class core LER sweep: the 9 core S-proxy programs at d = 3, 5.

Protocol matches the Clifford sweeps: paper full config
(ablation.compile_kwargs('full') + t_as_s=True), p in {5e-4, 1e-3},
with ONE per-program exception in the SCHED_UNFAITHFUL style:
multiply_n13 compiles with auto_rotate=True (its gadget corridors
hit a seam dead-end the feasibility-only rotation fallback cannot
repair; proactive rotation avoids it, all four checks pass).  The
knob stays off elsewhere: simon_n6 under auto_rotate trips a
logical-count tracker error (auto_rotate x t_as_s interaction,
tracked as a known issue), and every other program passes plain
full.
target 100 errors under 1e6 shots, PyMatching-only gate, per-point
deterministic seeds, raw counts.  Every compile is verified before
sampling: silent at p = 0, deterministic observables, and the logical
check against the T->S substituted program (the S-proxy semantics);
a compile failing any check is recorded and NOT sampled.

Two-stage execution: stage 1 compiles all (program, d) jobs in parallel
and dumps circuit text to a machine-local temp dir; stage 2 samples all
points in parallel.  Output: results/tclass/points_core.jsonl
(provenance first; rows stream in completion order, keys make each row
self-identifying).
"""
import json
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
OUT_FILE = ROOT / "experiments" / "results" / "tclass" / "points_core.jsonl"

CORE = ["teleportation_n3", "qec_en_n5", "toffoli_n3", "fredkin_n3",
        "bell_n4", "adder_n4", "simon_n6", "multiply_n13", "sat_n7"]
DS = [3, 5]
PS = [5e-4, 1e-3]
TARGET, MAX_SHOTS = 100, 1_000_000
SEED0 = 77190
COMPILE_WORKERS = 18
SAMPLE_WORKERS = 36


def _compile_job(task):
    """Stage 1: compile one (program, d); verify; dump circuit text."""
    name, d, seed_base, workdir = task
    import contextlib, io, time
    from benchsuite import tclass_suite
    from ablation import compile_kwargs
    from circls.pipeline import compile_qasm
    from circls.tools.evaluate import verify
    case = {c.name: c for c in tclass_suite()}[name]
    kw = compile_kwargs("full", case)
    if name == "multiply_n13":
        kw["auto_rotate"] = True     # see module docstring
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, distance=d, t_as_s=True, **kw)
    except Exception as e:
        return {"row": {"record": "error", "name": name, "d": d,
                        "error": f"{type(e).__name__}: {str(e)[:200]}"}}
    t_compile = round(time.time() - t0, 1)
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict
    stats = to_dict(experiment_stats(cp.experiment, cp.circuit))
    r = verify(cp)
    ok = bool(r.ok and r.logical is True)
    row = {"record": "precheck", "name": name, "d": d,
           "gadgets": len(cp.program.gadgets),
           "qubits": cp.circuit.num_qubits,
           "rotations": cp.experiment.rotation_count,
           "silent_p0": bool(r.silent), "deterministic": bool(r.deterministic),
           "distance_check": r.distance, "logical_check": bool(r.logical),
           "verify_ok": ok, "compile_s": t_compile,
           "normalization": case.normalization, "stats": stats}
    if not ok:
        return {"row": row}
    path = Path(workdir) / f"{name}_d{d}.stim"
    path.write_text(str(cp.circuit))
    return {"row": row, "circuit": str(path), "seed_base": seed_base,
            "compile_s": t_compile, "name": name, "d": d}


def _sample_job(task):
    """Stage 2: one (program, d, p) point."""
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
        shots, joint, per = adaptive_ler(
            noisy, m, task["seed"], TARGET, MAX_SHOTS)
        decoder = "mwpm"
    except ValueError as e:
        if "Failed to decompose" not in str(e):
            raise
        # paper protocol: non-graphlike error models fall back to the
        # MWPF hypergraph decoder, decoder recorded (six of the eight
        # d = 5 proxy circuits carry dormant-cell hyperedges, the
        # ghz_16-mixed mechanism; measured 2026-08-20)
        from formula_deviation import _mwpf_sample_ler
        shots, joint, _audit = _mwpf_sample_ler(
            noisy, task["seed"], TARGET, MAX_SHOTS)
        decoder = "mwpf"
    return {"record": "point", "name": task["name"], "d": task["d"],
            "p": task["p"], "seed": task["seed"], "decoder": decoder,
            "target_errors": TARGET, "max_shots": MAX_SHOTS,
            "compile_s": task["compile_s"], "shots": shots, "errors": joint,
            "ler": joint / shots if shots else None,
            "sample_s": round(time.time() - t0, 1)}


def main():
    from provenance import provenance
    if OUT_FILE.exists():
        print(f"skip: {OUT_FILE} exists", flush=True)
        return
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prov = {**provenance(False), "experiment": "tclass-core-ler",
            "what": "S-proxy (t_as_s) LER of the 9 core QASMBench T-class "
                    "programs, paper full config",
            "protocol": {"target_errors": TARGET, "max_shots": MAX_SHOTS,
                         "decoder": "mwpm", "seed0": SEED0,
                         "config": "full + t_as_s "
                                   "(+ auto_rotate on multiply_n13)",
                         "workers": [COMPILE_WORKERS, SAMPLE_WORKERS]},
            "matrix": {"programs": CORE, "d": DS, "p": PS}}
    workdir = tempfile.mkdtemp(prefix="tclass_circuits_")
    tasks = []
    for i, (name, d) in enumerate((n, d) for n in CORE for d in DS):
        tasks.append((name, d, SEED0 + 10 * i, workdir))
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
        points = []
        for r in compiled:
            for k, p in enumerate(PS):
                points.append({**{x: r[x] for x in
                                  ("circuit", "name", "d", "compile_s")},
                               "p": p, "seed": r["seed_base"] + k})
        with mp.get_context("spawn").Pool(SAMPLE_WORKERS) as pool:
            for row in pool.imap_unordered(_sample_job, points):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
    print("TCLASS CORE DONE", flush=True)


if __name__ == "__main__":
    main()
