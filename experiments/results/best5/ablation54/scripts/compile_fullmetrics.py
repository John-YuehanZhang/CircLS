"""Full-metric ablation re-compile: 54 programs x 7 table configs at d=3,
capturing the COMPLETE experiment_stats (bbox volume, peak footprint, live
tile-cycles, utilization -- geometric, deterministic) that the first campaign
did not persist.  Feeds tab:ablation-full and tab:ablation-consumable.
Per-cell JSON, resumable, parallel.  compile time (T4) is NOT used from here
(wall-clock, noisy); the geometric metrics are deterministic."""
import json, sys, time, contextlib, io
import multiprocessing as mp
from pathlib import Path
SP = Path("<workdir>/claude-1041/"
          "<host>/2f1c189c-a42c-406b-a2ef-c01703612f0b/scratchpad")
sys.path.insert(0, str(SP / "circls_dev"))
sys.path.insert(0, str(SP / "circls_dev" / "experiments"))
OUT = SP / "ablation_full" / "fullmetrics"
CFGS = ["full", "no_reduce", "no_fui", "no_schedpar",
        "no_place", "no_live", "reselect_only"]
NC = ["qec_en_n5", "teleportation_n3", "toffoli_n3", "bell_n4", "fredkin_n3",
      "adder_n4", "simon_n6", "multiply_n13", "sat_n7"]
SKIP_XL = {"bv_n140", "bv_n280"}


def targets():
    from benchsuite import suite
    cl = sorted(c.name for c in suite() if c.name not in SKIP_XL)
    t = [(n, c, "clifford") for n in cl for c in CFGS]
    t += [(n, c, "nonclifford") for n in NC for c in CFGS]
    return t


def job(task):
    name, cfg, kind = task
    tag = f"{name}__{cfg}"
    out = OUT / f"{tag}.json"
    if out.exists():
        return tag, "cached"
    from benchsuite import suite, tclass_suite
    from ablation import compile_kwargs
    from circls.pipeline import compile_qasm
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict
    from provenance import provenance
    src = suite() if kind == "clifford" else tclass_suite()
    case = {c.name: c for c in src}[name]
    kw = compile_kwargs(cfg, case)
    if kind == "nonclifford":
        kw["t_as_s"] = True
    t0 = time.time()
    rec = {"name": name, "config": cfg, "kind": kind, "prov": provenance(True)}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, distance=3, **kw)
    except Exception as e:
        rec.update(status=type(e).__name__, error=str(e)[:200],
                   wall_s=round(time.time() - t0, 1))
        out.write_text(json.dumps(rec))
        return tag, rec["status"]
    with contextlib.redirect_stdout(io.StringIO()):
        st = to_dict(experiment_stats(cp.experiment, cp.circuit))
    rec.update(status="OK", wall_s=round(time.time() - t0, 1), metrics=st)
    out.write_text(json.dumps(rec))
    return tag, "OK"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ts = [t for t in targets()
          if not (OUT / f"{t[0]}__{t[1]}.json").exists()]
    print(f"{len(ts)} compiles pending", flush=True)
    with mp.get_context("spawn").Pool(96, maxtasksperchild=1) as pool:
        for tag, status in pool.imap_unordered(job, ts):
            print(tag, status, flush=True)
    print("FULLMETRICS DONE", flush=True)
