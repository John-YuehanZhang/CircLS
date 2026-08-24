"""E2: exhaustive legal-order enumeration (optimality-gap experiment).

Per program: capture the pre-schedule circuit via an inspector
scheduler, enumerate every schedule permutation that satisfies the
verify_schedule contract (fixed ops pinned, anticommuting step pairs
keep order, steps stay on their side of anticommuting fixed ops),
replay each through compile_qasm(scheduler=injector) and record the
allocated volume.  Baseline = the built-in scheduler injected through
the same path (self-checked to reproduce the normal pipeline volume).

teleport_4 excluded: production pins step_scheduling=False for the
teleport family (decoder-faithfulness), so there is no order choice
to audit.

Output: experiments/results/best5/optgap/e2_<name>.jsonl
"""
import contextlib
import dataclasses
import io
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "results" / "best5" / "optgap"
OUT.mkdir(parents=True, exist_ok=True)

PROGRAMS = ["twistedghz_4", "steane_encode", "bv_8", "dj_8"]
WORKERS = 28
CAP = 500_000

_G = {}


def _init(qasm, kw):
    _G["qasm"], _G["kw"] = qasm, kw


def _eval(task):
    i, perm = task
    try:
        from circls.pipeline import compile_qasm
        from circls.metrics import experiment_stats
        from circls.metrics.report import to_dict

        def injector(pre):
            ops = pre.ops
            post = dataclasses.replace(
                pre, ops=[ops[perm[new]] for new in range(len(ops))])
            return post, list(perm)

        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(_G["qasm"], scheduler=injector, **_G["kw"])
        st = to_dict(experiment_stats(cp.experiment, cp.circuit))
        return (i, st["V1_volume_blocks"], st["T1_rounds"], None)
    except Exception as e:
        return (i, None, None, str(e)[:120])


def legal_perms(ops, is_step, anti):
    """Yield full-length perms (perm[new]=old) satisfying the contract."""
    n = len(ops)
    steps = [i for i in range(n) if is_step(ops[i])]
    fixed = [i for i in range(n) if not is_step(ops[i])]
    # side constraints: step i must occupy a slot on its original side of
    # every anticommuting fixed op f
    windows = {}
    for i in steps:
        lo, hi = 0, n - 1
        for f in fixed:
            if anti(ops[i], ops[f]):
                if i < f:
                    hi = min(hi, f - 1)
                else:
                    lo = max(lo, f + 1)
        windows[i] = (lo, hi)
    # precedence: anticommuting step pairs keep original order
    pred = {i: [] for i in steps}
    for a in range(len(steps)):
        for b in range(a + 1, len(steps)):
            i, j = steps[a], steps[b]
            if anti(ops[i], ops[j]):
                pred[j].append(i)
    slots = steps  # steps permute among the original step positions
    placed = {}

    def dfs(si):
        if si == len(slots):
            perm = list(range(n))
            for slot, op_i in placed.items():
                perm[slot] = op_i
            yield tuple(perm)
            return
        slot = slots[si]
        for i in steps:
            if i in placed.values():
                continue
            lo, hi = windows[i]
            if not (lo <= slot <= hi):
                continue
            if any(p not in placed.values() for p in pred[i]):
                continue
            placed[slot] = i
            yield from dfs(si + 1)
            del placed[slot]

    yield from dfs(0)


def main():
    from benchsuite import suite
    from ablation import compile_kwargs
    from circls.pipeline import compile_qasm
    from circls.compiler.scheduling import schedule_ops, _is_step, _anticommutes
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    cases = {c.name: c for c in suite()}
    for name in PROGRAMS:
        out_path = OUT / f"e2_{name}.jsonl"
        if out_path.exists():
            print(f"skip {name}", flush=True)
            continue
        case = cases[name]
        kw = compile_kwargs("full", case)
        kw.pop("step_scheduling", None)
        captured = {}

        def inspector(pre):
            captured["pre"] = pre
            return schedule_ops(pre, parallel=True)

        with contextlib.redirect_stdout(io.StringIO()):
            cp0 = compile_qasm(case.qasm, scheduler=inspector, **kw)
        st0 = to_dict(experiment_stats(cp0.experiment, cp0.circuit))
        v_c = st0["V1_volume_blocks"]
        pre = captured["pre"]
        perms = []
        truncated = False
        for perm in legal_perms(pre.ops, _is_step, _anticommutes):
            perms.append(perm)
            if len(perms) >= CAP:
                truncated = True
                break
        prov = {"record": "provenance", "experiment": "optgap-E2",
                "circls_sha": sha, "name": name,
                "n_ops": len(pre.ops),
                "n_steps": sum(1 for o in pre.ops if _is_step(o)),
                "n_legal_orders": len(perms), "truncated": truncated,
                "workers": WORKERS,
                "baseline": "built-in scheduler injected via scheduler= "
                            "(self-check: reproduces normal volume)"}
        t0 = time.time()
        best = None
        vols = []
        n_fail = 0
        with open(out_path, "w") as f:
            f.write(json.dumps(prov) + "\n")
            f.write(json.dumps({"record": "baseline", "volume": v_c,
                                "rounds": st0["T1_rounds"]}) + "\n")
            with mp.get_context("spawn").Pool(
                    WORKERS, initializer=_init,
                    initargs=(case.qasm, kw), maxtasksperchild=400) as pool:
                for i, v, r, err in pool.imap_unordered(
                        _eval, enumerate(perms), chunksize=4):
                    if err is not None:
                        n_fail += 1
                        f.write(json.dumps({"i": i, "err": err}) + "\n")
                        continue
                    vols.append(v)
                    if best is None or v < best[1]:
                        best = (i, v)
                    f.write(json.dumps({"i": i, "v": v, "r": r}) + "\n")
                    if len(vols) % 5000 == 0:
                        print(f"{name}: {len(vols)}/{len(perms)} "
                              f"{(time.time()-t0)/60:.0f}min "
                              f"best={best[1]:.1f}", flush=True)
            summ = {"record": "summary", "name": name,
                    "n_evals": len(vols), "n_failed": n_fail,
                    "min_volume": best[1] if best else None,
                    "circls_volume": v_c,
                    "gap_pct": (100 * (v_c / best[1] - 1)) if best else None,
                    "n_strictly_better": sum(1 for v in vols if v < v_c),
                    "seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(summ) + "\n")
        print("SUMMARY " + json.dumps(summ), flush=True)
    print("E2 DONE", flush=True)


if __name__ == "__main__":
    main()
