"""Distance probes for the shape-coverage audit.

Closes the two gaps the inventory found: (a) no graphlike-distance
evidence at d=5 for any program, (b) two d=3 shapes (corridor branch,
parallel window) that occur only in programs the roadtest never
searched.  Each probe compiles the full pipeline (ablation "full"
config) at the given distance with circuit-level noise and runs
stim's shortest_graphlike_error, in its own subprocess with a
timeout.

Usage:
    python experiments/shape_probes.py [--timeout 3600] [--workers 8] \
        [--allow-dirty] [--out experiments/results/best5/shape_probes.jsonl]
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
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

#: the 14 programs of the d=3 graphlike sweep, re-probed at d=5, plus the
#: two shapes only larger programs exercise: corridor branches (bv_n14)
#: and the one parallel window in the suite (ghz_16_mixed).
PROBES = [
    ("deutsch_n2", 5), ("grover_n2", 5), ("iswap_n2", 5),
    ("cat_state_n4", 5), ("hs4_n4", 5), ("lpn_n5", 5),
    ("ghz_8", 5), ("bv_8", 5), ("dj_8", 5),
    ("teleport_4", 5), ("twistedghz_4", 5), ("twistedghz_8", 5),
    ("bbpssw_4", 5), ("steane_encode", 5),
    ("bv_n14", 3), ("bv_n14", 5),
    ("ghz_16_mixed", 3), ("ghz_16_mixed", 5),
]


def _run_case(name, d, conn):
    import ablation
    from benchsuite import suite
    from circls.pipeline import compile_qasm
    from lightstim.noise.config import NoiseConfig

    out = {"record": "probe", "name": name, "d": d}
    t0 = time.perf_counter()
    try:
        case = {c.name: c for c in suite()}[name]
        kw = ablation.compile_kwargs("full", case)
        kw["distance"] = d
        with contextlib.redirect_stdout(io.StringIO()):
            noisy = compile_qasm(
                case.qasm,
                noise=NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3,
                                  p_reset=1e-3, p_idle=1e-3), **kw)
        out["num_observables"] = noisy.circuit.num_observables
        if noisy.circuit.num_observables == 0:
            out["status"] = "no_observables"
        else:
            out["graphlike_distance"] = len(
                noisy.circuit.shortest_graphlike_error())
            out["status"] = "OK"
    except Exception as e:
        out["status"] = type(e).__name__
        out["error"] = str(e)[:300]
        out["trace_tail"] = traceback.format_exc().splitlines()[-3:]
    out["seconds"] = round(time.perf_counter() - t0, 1)
    conn.send(out)
    conn.close()


def _drain(running, fh, now):
    """Reap finished/expired probes; returns the still-running list."""
    keep = []
    for name, d, proc, conn, deadline in running:
        if conn.poll(0):
            row = conn.recv()
            proc.join()
        elif not proc.is_alive():
            row = {"record": "probe", "name": name, "d": d,
                   "status": "died", "seconds": 0}
        elif now > deadline:
            proc.terminate()
            proc.join()
            row = {"record": "probe", "name": name, "d": d,
                   "status": "timeout", "seconds": round(now - (deadline
                   - _TIMEOUT[0]), 1)}
        else:
            keep.append((name, d, proc, conn, deadline))
            continue
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        gl = row.get("graphlike_distance", "-")
        print(f"{row['name']:16s} d={row['d']} {row['status']:12s} "
              f"gl={gl} ({row.get('seconds')}s)", flush=True)
    return keep


_TIMEOUT = [3600]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default=str(
        _ROOT / "experiments/results/best5/shape_probes.jsonl"))
    args = ap.parse_args()

    from provenance import provenance
    prov = provenance(args.allow_dirty)

    tasks = [(nm, d, args.timeout) for nm, d in PROBES
             if args.only is None or nm in args.only]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _TIMEOUT[0] = args.timeout
    ctx = mp.get_context("spawn")
    with open(out, "a") as fh:
        fh.write(json.dumps({**prov, "record": "provenance",
                             "argv": sys.argv}) + "\n")
        pending = list(tasks)
        running = []
        while pending or running:
            while pending and len(running) < args.workers:
                name, d, timeout = pending.pop(0)
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_run_case, args=(name, d, child))
                proc.start()
                running.append((name, d, proc, parent,
                                time.time() + timeout))
            running = _drain(running, fh, time.time())
            time.sleep(1)


if __name__ == "__main__":
    main()
