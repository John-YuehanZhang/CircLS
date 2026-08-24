"""Week-1 road test: drive every benchmark case through the full pipeline.

Per case (ascending n): compile (row_major; optimized for n <= 32), collect
the aggregator report, noiseless silence check (n <= 64), fault distance
(n <= 10, noisy graphlike), and a liveness pass for consumable cases with
the approved keep policies.  Each case runs in its own subprocess with a
timeout so a wall on one circuit never stalls the sweep; results append to
a JSONL as they arrive.

Usage:
    python experiments/roadtest.py [--timeout 600] [--workers 8] \
        [--out experiments/results/roadtest.jsonl] [--only NAME ...]
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT.parent / "LightStim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchsuite import suite  # noqa: E402


def _keep_policy(case):
    """Approved keep policies (benchmark_set.md): teleport keeps the far end
    (relays retire); bbpssw keeps the target pair; shchain keeps all data
    patches (the |Y> ancillas are the consumables)."""
    if case.name.startswith("teleport_"):
        return {f"q{case.n_qubits - 1}"}
    if case.name.startswith("bbpssw_"):
        return {"q0", "q1"}
    if case.name.startswith("twistedghz_"):
        return {f"q{i}" for i in range(case.n_qubits)}
    return None


def _run_case(case, policy, liveness, conn):
    from circls.pipeline import compile_qasm
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict
    t0 = time.perf_counter()
    out = {"name": case.name, "policy": policy, "liveness": liveness,
           "n": case.n_qubits, "source": case.source, "tags": list(case.tags)}
    try:
        kw = {}
        if liveness:
            kw = {"liveness": True, "keep_patches": _keep_policy(case)}
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, assignment=policy, **kw)
        out["build_seconds"] = round(time.perf_counter() - t0, 2)
        es = experiment_stats(cp.experiment, cp.circuit,
                              compile_seconds=out["build_seconds"])
        out["stats"] = to_dict(es)
        if case.n_qubits <= 64:
            det, _ = cp.circuit.compile_detector_sampler(seed=0).sample(
                128, separate_observables=True)
            out["silent"] = bool(not det.any())
        out["num_observables"] = cp.circuit.num_observables
        if case.n_qubits <= 10 and cp.circuit.num_observables > 0:
            from lightstim.noise.config import NoiseConfig
            noisy = compile_qasm(case.qasm, assignment=policy,
                                 noise=NoiseConfig(p_1q=1e-3, p_2q=1e-3,
                                                   p_meas=1e-3, p_reset=1e-3,
                                                   p_idle=1e-3), **kw)
            out["graphlike_distance"] = len(
                noisy.circuit.shortest_graphlike_error())
        out["status"] = "OK"
    except Exception as e:
        out["build_seconds"] = round(time.perf_counter() - t0, 2)
        out["status"] = f"{type(e).__name__}"
        out["error"] = str(e)[:400]
        out["trace_tail"] = traceback.format_exc().splitlines()[-3:]
    conn.send(out)
    conn.close()


def run_one(case, policy, liveness, timeout):
    parent, child = mp.Pipe()
    p = mp.Process(target=_run_case, args=(case, policy, liveness, child))
    p.start()
    if parent.poll(timeout):
        out = parent.recv()
    else:
        out = {"name": case.name, "policy": policy, "liveness": liveness,
               "n": case.n_qubits, "source": case.source,
               "tags": list(case.tags), "status": "TIMEOUT",
               "build_seconds": timeout}
        p.terminate()
    p.join(5)
    if p.is_alive():
        p.kill()
    return out


def jobs_for(case):
    yield (case, "row_major", False)
    if case.n_qubits <= 32:
        yield (case, "optimized", False)
    if "consumable" in case.tags:
        yield (case, "row_major", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="experiments/results/roadtest.jsonl")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # coverage ruling (user 2026-08-05): mandatory coverage tops out at
    # TopoLS's largest benchmark (BV/DJ n=100) — n>100 QASMBench giants
    # that still pass stay as bonus scale evidence, but the two bv giants
    # are out of the default sweep entirely (pass --only to run them).
    skip_xl = {"bv_n140", "bv_n280"}
    cases = sorted((c for c in suite() if c.name not in skip_xl),
                   key=lambda c: (c.n_qubits, c.name))
    if args.only:
        cases = [c for c in cases if c.name in args.only]
    jobs = [j for c in cases for j in jobs_for(c)]
    print(f"{len(cases)} cases, {len(jobs)} jobs, timeout {args.timeout}s, "
          f"workers {args.workers}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    fh = open(args.out, "a")

    def worker(job):
        case, policy, liveness = job
        out = run_one(case, policy, liveness, args.timeout)
        line = json.dumps(out)
        fh.write(line + "\n")
        fh.flush()
        tag = "L" if liveness else " "
        extra = ""
        if out["status"] == "OK":
            st = out.get("stats", {})
            extra = (f"rounds={st.get('T1_rounds')} "
                     f"V1={st.get('V1_volume_blocks', 0):.0f} "
                     f"t={out['build_seconds']}s"
                     + (f" gl={out['graphlike_distance']}"
                        if "graphlike_distance" in out else "")
                     + (f" silent={out['silent']}" if "silent" in out else ""))
        else:
            extra = f"{out['status']}: {out.get('error', '')[:80]}"
        print(f"[{out['status']:>7s}]{tag} {case.name:22s} {policy:9s} {extra}",
              flush=True)
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(worker, jobs))
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\ndone: {ok}/{len(results)} OK -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
