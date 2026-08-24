"""T-class scale-tier compile metrics: the 10 compile-only programs.

Compiles each scale-tier program (t_as_s, paper full config) at d = 3
and records the cost metrics (experiment_stats) plus the p = 0 silence
check; no LER sampling (T counts 56 to 7560 put these programs beyond
any sampling budget; the census note at benchsuite._TCLASS).  Cases run
smallest-first in a small pool, each in its own process with a hard
timeout, recorded honestly as TIMEOUT rows.  Output:
results/tclass/scale_metrics.jsonl (provenance first).
"""
import json
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
OUT_FILE = ROOT / "experiments" / "results" / "tclass" / "scale_metrics.jsonl"

SCALE = ["seca_n11", "qram_n20", "adder_n28", "multiplier_n15", "sat_n11",
         "adder_n64", "adder_n118", "adder_n433", "multiplier_n45",
         "multiplier_n75"]
D = 3
TIMEOUT_S = 4 * 3600
WORKERS = 10


def _job(name, conn):
    import contextlib, io, time, traceback
    from benchsuite import tclass_suite
    from ablation import compile_kwargs
    from circls.pipeline import compile_qasm
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict
    case = {c.name: c for c in tclass_suite()}[name]
    t0 = time.perf_counter()
    out = {"record": "case", "name": name, "n": case.n_qubits, "d": D,
           "normalization": case.normalization}
    try:
        kw = compile_kwargs("full", case)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cp = compile_qasm(case.qasm, distance=D, t_as_s=True, **kw)
            out["auto_rotate"] = False
        except Exception as first:
            # the multiply_n13 seam dead-end family: retry once with
            # proactive rotation, recorded per program (core-tier policy)
            if "BentLayoutError" not in type(first).__name__:
                raise
            out["first_attempt_s"] = round(time.perf_counter() - t0, 1)
            t0 = time.perf_counter()   # compile_s = the SUCCESSFUL attempt
            kw["auto_rotate"] = True
            with contextlib.redirect_stdout(io.StringIO()):
                cp = compile_qasm(case.qasm, distance=D, t_as_s=True, **kw)
            out["auto_rotate"] = True
        out["compile_s"] = round(time.perf_counter() - t0, 1)
        out["gadgets"] = len(cp.program.gadgets)
        out["qubits"] = cp.circuit.num_qubits
        out["rotations"] = cp.experiment.rotation_count
        out["stats"] = to_dict(experiment_stats(cp.experiment, cp.circuit))
        det, _ = cp.circuit.compile_detector_sampler(seed=0).sample(
            128, separate_observables=True)
        out["silent_p0"] = bool(not det.any())
        out["status"] = "OK"
    except Exception as e:
        out["compile_s"] = round(time.perf_counter() - t0, 1)
        out["status"] = type(e).__name__
        out["error"] = str(e)[:300]
        out["trace_tail"] = traceback.format_exc().splitlines()[-3:]
    conn.send(out)
    conn.close()


def run_one(name):
    import time as _time
    parent, child = mp.Pipe()
    p = mp.get_context("spawn").Process(target=_job, args=(name, child))
    p.start()
    child.close()   # parent must drop its copy or a dead child never EOFs
    deadline = _time.monotonic() + TIMEOUT_S
    out = None
    while _time.monotonic() < deadline:
        if parent.poll(5):
            try:
                out = parent.recv()
            except EOFError:
                out = {"record": "case", "name": name, "d": D,
                       "status": "CHILD_DIED",
                       "exitcode": p.exitcode}
            break
        if not p.is_alive():
            out = {"record": "case", "name": name, "d": D,
                   "status": "CHILD_DIED", "exitcode": p.exitcode}
            break
    if out is None:
        p.terminate()
        out = {"record": "case", "name": name, "d": D, "status": "TIMEOUT",
               "compile_s": TIMEOUT_S}
    p.join(30)
    if p.is_alive():
        p.kill()
        p.join()
    return out


def main():
    from provenance import provenance
    if OUT_FILE.exists():
        print(f"skip: {OUT_FILE} exists", flush=True)
        return
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prov = {**provenance(False), "experiment": "tclass-scale-metrics",
            "what": "compile metrics of the 10 scale-tier T-class programs "
                    "(t_as_s, paper full config, d = 3, no sampling)",
            "protocol": {"d": D, "timeout_s": TIMEOUT_S,
                         "workers": WORKERS,
                         "order": "all cases start concurrently"}}
    with open(OUT_FILE, "w") as f:
        f.write(json.dumps(prov) + "\n")
        f.flush()
        with ThreadPool(WORKERS) as pool:
            for row in pool.imap(run_one, SCALE):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps({k: row[k] for k in row
                                  if k != "stats"}), flush=True)
    print("TCLASS SCALE DONE", flush=True)


if __name__ == "__main__":
    main()
