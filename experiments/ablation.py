"""A2 ablation matrix: full config minus one component at a time.

Configs (design decisions 2026-08-05: four quantified components; the rotation
planner is NOT ablated — its value is a rotate_saving_threshold tuning
question — and multi-wall is not ablated — measured unused post-reduction):

    full        measure_reduction + optimized placement + first_use_init
                + step scheduling + liveness everywhere (approved keep
                policies on consumables, keep=() elsewhere — terminal bits
                are captured by the retirement readout)
    no_reduce   full - terminal measurement-set re-selection
    no_place    full - optimized placement (row_major)
    no_fui      full - first-use allocation (everything allocated at start)
    no_live     full - liveness retirement / corridor reuse
    no_sched    full - lifetime-aware step scheduling
    no_parallel full - shared merge windows (v4, 2026-08-06: parallel step
                execution went default-ON after its gate battery greened)
    no_schedpar full - scheduling AND parallel execution (design decision
                2026-08-10: the paper's ablation row mirrors the
                Reordering and Parallel Execution subsection, one
                switch; the single-switch rows stay in the artifact)

Every output file starts with a provenance record (dirty tree refuses to
run; see experiments/provenance.py).  Results append per-config to
``experiments/results/ablation/<config>.jsonl``; the paper table is
generated from those files by experiments/ablation_table.py — no numbers
are ever copied by hand.

Usage:
    python experiments/ablation.py [--quick] [--configs full no_reduce ...]
        [--timeout 1200] [--workers 32] [--allow-dirty] [--only NAME ...]
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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

from benchsuite import suite                     # noqa: E402
from provenance import provenance                # noqa: E402
from roadtest import _keep_policy                # noqa: E402

CONFIGS = {
    "full":        {},
    "no_reduce":   {"measure_reduction": False},
    "no_place":    {"assignment": "row_major"},
    "no_fui":      {"first_use_init": False},
    "no_live":     {"liveness": "off"},
    "no_sched":    {"step_scheduling": False},
    "no_parallel": {"parallel_steps": False},
    "no_schedpar": {"step_scheduling": False, "parallel_steps": False},
    # prior-practice floor: re-selection on, all four later stages off —
    # full vs this row = the joint contribution of the lifetime stages
    "reselect_only": {"assignment": "row_major", "first_use_init": False,
                      "step_scheduling": False, "parallel_steps": False,
                      "liveness": "off"},
}

_BASE = {"assignment": "optimized", "measure_reduction": True,
         "first_use_init": True, "step_scheduling": True,
         "parallel_steps": True, "liveness": "policy"}

QUICK_CASES = ["ghz_8", "graphstate_8", "bv_8", "dj_8", "steane_encode",
               "teleport_4", "twistedghz_4", "bbpssw_4"]


# decoder-faithfulness audit 2026-08-05: under step scheduling the teleport
# family's rescheduled merge order miscorrects single-error mechanisms
# (88/39191 at teleport_4; every other audited case 0) — geometric, not
# algebraic, so it is pinned by measurement.  experiments/ler.py enforces
# the same rule generally via its per-point gate + fallback.
SCHED_UNFAITHFUL = ("teleport_",)


def compile_kwargs(config: str, case) -> dict:
    kw = dict(_BASE)
    kw.update(CONFIGS[config])
    if case.name.startswith(SCHED_UNFAITHFUL):
        kw["step_scheduling"] = False
    live = kw.pop("liveness")
    if live == "policy":
        # liveness everywhere (2026-08-05, with step scheduling): consumable
        # cases use the approved keep policies; everywhere else keep=() —
        # every terminal bit is captured by the retirement readout, so
        # nothing needs holding to the end
        keep = _keep_policy(case) if "consumable" in case.tags else None
        kw["liveness"] = True
        kw["keep_patches"] = keep if keep is not None else set()
    return kw


def _run_case(case, config, conn):
    from circls.pipeline import compile_qasm
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict
    t0 = time.perf_counter()
    out = {"name": case.name, "config": config, "n": case.n_qubits,
           "source": case.source, "tags": list(case.tags)}
    try:
        kw = compile_kwargs(config, case)
        out["compile_kwargs"] = {k: (sorted(v) if isinstance(v, (set, frozenset))
                                     else v) for k, v in kw.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, **kw)
        out["build_seconds"] = round(time.perf_counter() - t0, 2)
        es = experiment_stats(cp.experiment, cp.circuit,
                              compile_seconds=out["build_seconds"])
        out["stats"] = to_dict(es)
        out["rotations"] = len(cp.experiment.rotation_log)
        if case.n_qubits <= 64:
            det, _ = cp.circuit.compile_detector_sampler(seed=0).sample(
                128, separate_observables=True)
            out["silent"] = bool(not det.any())
            out["silence_shots"] = 128
            out["silence_seed"] = 0
        out["num_observables"] = cp.circuit.num_observables
        out["status"] = "OK"
    except Exception as e:
        out["build_seconds"] = round(time.perf_counter() - t0, 2)
        out["status"] = type(e).__name__
        out["error"] = str(e)[:400]
        out["trace_tail"] = traceback.format_exc().splitlines()[-3:]
    conn.send(out)
    conn.close()


def run_one(case, config, timeout):
    parent, child = mp.Pipe()
    p = mp.Process(target=_run_case, args=(case, config, child))
    p.start()
    if parent.poll(timeout):
        out = parent.recv()
    else:
        out = {"name": case.name, "config": config, "n": case.n_qubits,
               "source": case.source, "tags": list(case.tags),
               "status": "TIMEOUT", "build_seconds": timeout}
        p.terminate()
    p.join(5)
    if p.is_alive():
        p.kill()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="kick-the-tires subset (minutes, not hours)")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--outdir", default="experiments/results/ablation")
    args = ap.parse_args()

    prov = provenance(allow_dirty=args.allow_dirty)   # raises on dirty tree
    unknown = set(args.configs) - set(CONFIGS)
    if unknown:
        ap.error(f"unknown configs: {sorted(unknown)}")

    skip_xl = {"bv_n140", "bv_n280"}     # coverage ruling 2026-08-05
    cases = sorted((c for c in suite() if c.name not in skip_xl),
                   key=lambda c: (c.n_qubits, c.name))
    if args.quick:
        cases = [c for c in cases if c.name in QUICK_CASES]
        args.timeout = min(args.timeout, 300)
    if args.only:
        cases = [c for c in cases if c.name in args.only]

    outdir = Path(args.outdir if not args.quick
                  else args.outdir.rstrip("/") + "_quick")
    outdir.mkdir(parents=True, exist_ok=True)
    handles = {}
    for cfg in args.configs:
        fh = open(outdir / f"{cfg}.jsonl", "a")
        fh.write(json.dumps({**prov, "config": cfg}) + "\n")
        fh.flush()
        handles[cfg] = fh

    jobs = [(case, cfg) for cfg in args.configs for case in cases]
    print(f"{len(cases)} cases x {len(args.configs)} configs = {len(jobs)} "
          f"jobs, timeout {args.timeout}s, workers {args.workers}, "
          f"sha {prov['circls_sha'][:9]}{' DIRTY' if prov['dirty'] else ''}",
          flush=True)

    from concurrent.futures import ThreadPoolExecutor

    def worker(job):
        case, cfg = job
        out = run_one(case, cfg, args.timeout)
        handles[cfg].write(json.dumps(out) + "\n")
        handles[cfg].flush()
        if out["status"] == "OK":
            st = out["stats"]
            extra = (f"V1={st.get('V1_volume_blocks', 0):.0f} "
                     f"rounds={st.get('T1_rounds')} "
                     f"t={out['build_seconds']}s")
        else:
            extra = f"{out.get('error', '')[:70]}"
        print(f"[{out['status']:>7s}] {cfg:10s} {case.name:20s} {extra}",
              flush=True)
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(worker, jobs))
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\ndone: {ok}/{len(results)} OK -> {outdir}/", flush=True)


if __name__ == "__main__":
    main()
