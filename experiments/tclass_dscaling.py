"""Non-Clifford distance scaling under the Y-state approximation:
dynamic vs static, d = 3 to 11 at p = 5e-4, for any tclass program
(``--program``, default toffoli_n3).

The dynamic/static comparison of the DJ-16 panel, on an S-proxy
non-Clifford program.  Static = liveness and first-use init both
removed, as in the Clifford runs.  Protocol: every point targets 100
failures; the dynamic p = 5e-4 points at d = 5, 7, 9 deepen to 400,
matching the six-program policy; shot caps 1e6 (d = 3..7), 2e6
(d = 9), 4e6 (d = 11), both configs.  PyMatching-only gate,
per-point deterministic seeds, raw counts.  Two-stage pools.
Output: results/tclass/points_<program>_dscaling.jsonl.

fig:dscaling panel (b) shows adder_n4; its shipped data file
(results/tclass/points_adder_panel.jsonl) was assembled from
development-session shard runs that PREDATE this parameterized entry
point, so its rows carry no seeds/commit (see the file's leading note
row).  ``--program adder_n4`` regenerates the panel with fresh seeds:
statistically compatible values, not byte-identical.
"""
import json
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
def _out_file(name):
    return (ROOT / "experiments" / "results" / "tclass"
            / f"points_{name}_dscaling.jsonl")

NAME = "toffoli_n3"          # overridden by --program
CONFIGS = ["full", "static"]
DS = [3, 5, 7, 9, 11]        # overridden by --ds
P = 5e-4
SEED0 = 231100               # overridden by --seed0; per-program defaults below
SEED0_DEFAULTS = {"toffoli_n3": 231100, "adder_n4": 241100,
                  "qec_en_n5": 251100}
COMPILE_WORKERS = 10
SAMPLE_WORKERS = 10


def _depth(config, d):
    cap = {9: 2_000_000, 11: 4_000_000}.get(d, 1_000_000)
    target = 400 if config == "full" and d in (5, 7, 9) else 100
    return target, cap


def _compile_job(task):
    config, d, seed, workdir, name = task
    global NAME
    NAME = name
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
            "compile_s": t_compile, "config": config, "d": d,
            "name": name}


def _sample_job(task):
    global NAME
    NAME = task["name"]
    import time
    import stim
    import pymatching as _pm
    from noise_inject import inject_uniform_noise
    from compare_tqec import adaptive_ler
    circuit = stim.Circuit(Path(task["circuit"]).read_text())
    noisy = inject_uniform_noise(circuit, P)
    target, cap = _depth(task["config"], task["d"])
    if task.get("cap"):
        cap = task["cap"]
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
    import argparse
    global NAME, DS, SEED0
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default="toffoli_n3",
                    help="tclass benchmark name (e.g. adder_n4)")
    ap.add_argument("--ds", nargs="+", type=int, default=None,
                    help="distances (default 3 5 7 9 11)")
    ap.add_argument("--seed0", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="throwaway runs from a dirty tree")
    ap.add_argument("--cap", nargs="*", default=[], metavar="D=SHOTS",
                    help="per-distance shot-cap override, e.g. 11=80000000 "
                         "(the shipped d=11 panel aggregates 72M shots; the "
                         "built-in 4M cap reaches only ~10-15 failures there)")
    args = ap.parse_args()
    NAME = args.program
    if args.ds:
        DS = args.ds
    SEED0 = (args.seed0 if args.seed0 is not None
             else SEED0_DEFAULTS.get(NAME, 231100 + 7000 * (hash(NAME) % 97)))
    caps = {int(k): int(v) for k, v in
            (x.split("=") for x in args.cap)}
    out_file = _out_file(NAME)
    from provenance import provenance
    if out_file.exists():
        print(f"skip: {out_file} exists", flush=True)
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    prov = {**provenance(args.allow_dirty), "experiment": f"tclass-{NAME}-dscaling",
            "what": f"{NAME} (Y-state approximation) dynamic vs static "
                    f"at d = {DS}, p = 5e-4",
            "protocol": {"p": P, "target_errors": 100,
                         "deep_dynamic": {"d": [5, 7, 9], "target": 400},
                         "caps": {"default": 1_000_000, "d9": 2_000_000,
                                  "d11": 4_000_000},
                         "cap_overrides": caps,
                         "decoder": "mwpm", "seed0": SEED0}}
    workdir = tempfile.mkdtemp(prefix=f"{NAME}_dsc_")
    tasks = []
    for i, (config, d) in enumerate((c, d) for c in CONFIGS for d in DS):
        tasks.append((config, d, SEED0 + i, workdir, NAME))
    with open(out_file, "w") as f:
        f.write(json.dumps(prov) + "\n")
        f.flush()
        compiled = []
        with mp.get_context("spawn").Pool(COMPILE_WORKERS) as pool:
            for r in pool.imap_unordered(_compile_job, tasks):
                f.write(json.dumps(r["row"]) + "\n")
                f.flush()
                print(json.dumps(r["row"]), flush=True)
                if r.get("circuit"):
                    if r["d"] in caps:
                        r["cap"] = caps[r["d"]]
                    compiled.append(r)
        with mp.get_context("spawn").Pool(SAMPLE_WORKERS) as pool:
            for row in pool.imap_unordered(_sample_job, compiled):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
    print(f"{NAME.upper()} DSCALING DONE", flush=True)


if __name__ == "__main__":
    main()
