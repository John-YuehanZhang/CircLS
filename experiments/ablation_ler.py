"""A2-LER: system-level LER per ablation configuration.

Design decision 2026-08-11: the ablation table reports all five metrics,
including LER.  Roster = the LER-panel cases (the informative-LER
regime at d=3); configs = the paper's six ablation rows (the merged
scheduling switch, not the single switches); 10k-failure target for
~1% relative error per point.

Points where the error model is not graphlike fall back to mwpf and
are marked, as everywhere else.
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

CASES = ["ghz_8", "bv_8", "dj_8", "teleport_4", "twistedghz_4", "bbpssw_4"]
CONFIGS = ["full", "no_reduce", "no_place", "no_fui", "no_schedpar",
           "no_live", "reselect_only"]


def _point(task):
    case_name, config, p, seed, target, max_shots = task
    import numpy as np
    from benchsuite import suite
    from ablation import compile_kwargs
    from compare_tqec import adaptive_ler
    from formula_deviation import _faithful_audit
    from noise_inject import inject_uniform_noise
    from circls.pipeline import compile_qasm

    t0 = time.perf_counter()
    row = {"record": "point", "name": case_name, "config": config, "p": p,
           "d": 3, "seed": seed}
    try:
        case = next(c for c in suite() if c.name == case_name)
        kw = compile_kwargs(config, case)
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, **kw)
        noisy = inject_uniform_noise(cp.circuit, p)
        try:
            m, mis, n_mech = _faithful_audit(noisy)
        except ValueError as e:
            if "Failed to decompose" not in str(e):
                raise
            row.update({"status": "non-graphlike"})
            return row
        shots, joint, per = adaptive_ler(noisy, m, seed, target, max_shots)
        row.update({"status": "OK", "decoder": "mwpm", "shots": shots,
                    "joint_errors": joint, "ler": joint / shots,
                    "audit": {"miscorrected": mis, "mechanisms": n_mech},
                    "seconds": round(time.perf_counter() - t0, 1)})
    except Exception as e:
        row.update({"status": type(e).__name__, "error": str(e)[:200],
                    "seconds": round(time.perf_counter() - t0, 1)})
    return row


def main():
    from formula_deviation import _mwpf_sample_ler, _audit_dict
    from provenance import provenance

    ap = argparse.ArgumentParser()
    ap.add_argument("--p", nargs="+", type=float, default=[1e-3, 5e-4])
    ap.add_argument("--target-errors", type=int, default=10000)
    ap.add_argument("--max-shots", type=int, default=4_000_000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out",
                    default="experiments/results/best5/ablation/ler.jsonl")
    args = ap.parse_args()
    prov = provenance(allow_dirty=args.allow_dirty)

    done = set()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            if r.get("record") == "point" and r.get("status") == "OK":
                done.add((r["name"], r["config"], r["p"]))
    fh = open(out, "a")
    fh.write(json.dumps({**prov, "record": "provenance",
                         "args": vars(args)}) + "\n")
    fh.flush()

    tasks = []
    idx = 0
    for config in CONFIGS:
        for name in CASES:
            for p in args.p:
                idx += 1
                if (name, config, p) in done:
                    continue
                tasks.append((name, config, p, args.seed + idx,
                              args.target_errors, args.max_shots))

    hyper = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(_point, tasks):
            if row.get("status") == "non-graphlike":
                hyper.append(row)
                continue
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{row.get('status'):>9s}] {row['config']:12s} "
                  f"{row['name']:14s} p={row['p']:g} "
                  f"ler={row.get('ler')}", flush=True)

    # non-graphlike points: mwpf serially in the main process (its own
    # worker pool cannot be spawned from a daemonic Pool worker)
    for row in hyper:
        from benchsuite import suite
        from ablation import compile_kwargs
        from noise_inject import inject_uniform_noise
        from circls.pipeline import compile_qasm
        case = next(c for c in suite() if c.name == row["name"])
        kw = compile_kwargs(row["config"], case)
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, **kw)
        noisy = inject_uniform_noise(cp.circuit, row["p"])
        try:
            shots, joint, audit = _mwpf_sample_ler(
                noisy, row["seed"], args.target_errors, args.max_shots)
            row.update({"status": "OK", "decoder": "mwpf", "shots": shots,
                        "joint_errors": joint, "ler": joint / shots,
                        "audit": audit})
        except Exception as e:
            row.update({"status": type(e).__name__,
                        "error": str(e)[:200]})
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(f"[{row.get('status'):>9s}] {row['config']:12s} "
              f"{row['name']:14s} p={row['p']:g} mwpf "
              f"ler={row.get('ler')}", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
