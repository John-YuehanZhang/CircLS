"""LER evaluation line: noisy sampling + pymatching decoding per point.

A point = (case, distance, p, seed).  Compile under the paper's full
config, build the detector error model (decompose_errors=True — the
graphlike precheck 2026-08-05 cleared every planned case except steane,
whose wall block is hypergraph territory deferred to mwpf), decode in
batches, and stop adaptively: >= --target-errors joint logical errors or
--max-shots, whichever first.  Published numbers are RAW COUNTS (shots,
per-observable errors, joint errors) with seed and provenance — rates are
derived downstream, never stored alone.

Usage:
    python experiments/ler.py --cases teleport_4 dj_8 --d 3 5 \
        --p 1e-3 5e-4 2e-4 [--target-errors 100] [--max-shots 1000000] \
        [--seed 7] [--workers 16] [--out experiments/results/ler/main.jsonl]
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

from ablation import compile_kwargs                    # noqa: E402  (full config)
from benchsuite import suite                           # noqa: E402
from provenance import provenance                      # noqa: E402

BATCH = 10_000


def _eval_point(case, d, p, seed, target_errors, max_shots, conn):
    import numpy as np
    import pymatching
    from circls.pipeline import compile_qasm
    from lightstim.noise.config import NoiseConfig
    t0 = time.perf_counter()
    out = {"name": case.name, "d": d, "p": p, "seed": seed,
           "n": case.n_qubits, "target_errors": target_errors,
           "max_shots": max_shots}
    try:
        from noise_inject import inject_uniform_noise
        kw = compile_kwargs("full", case)
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, distance=d, **kw)
        noisy = inject_uniform_noise(cp.circuit, p)   # uniform pass, unifies
        # the noise model with the headline/tclass/ablation tables (2026-08-23)
        # decoder-faithfulness gate (2026-08-05, the teleport-floor lesson):
        # EVERY single-error mechanism in the DEM must decode to its own
        # observable flip — a forced hyperedge decomposition can mislead
        # matching while graphlike AND hypergraph distance still read full,
        # producing a silent LER floor ~ (miscorrected mechanisms) x p / 2.
        # Publishing a rate from an unfaithful decoder would be garbage
        # with a straight face, so the point fails loudly instead.
        def _single_error_audit(circ):
            dem = circ.detector_error_model(decompose_errors=True)
            assert dem.num_detectors == circ.num_detectors
            assert dem.num_observables == circ.num_observables
            matching = pymatching.Matching.from_detector_error_model(dem)
            flat = dem.flattened()
            dets, obss = [], []
            for inst in flat:
                if inst.type != "error":
                    continue
                dv = np.zeros(flat.num_detectors, dtype=bool)
                ov = np.zeros(flat.num_observables, dtype=bool)
                for t in inst.targets_copy():
                    if t.is_relative_detector_id():
                        dv[t.val] ^= True
                    elif t.is_logical_observable_id():
                        ov[t.val] ^= True
                dets.append(dv)
                obss.append(ov)
            mis = int((matching.decode_batch(np.array(dets)).astype(bool)
                       ^ np.array(obss)).any(axis=1).sum())
            return mis, len(dets), matching

        mis, n_mech, matching = _single_error_audit(noisy)
        out["single_error_mechanisms"] = n_mech
        out["miscorrected_singles"] = mis
        # fallback ladder (the teleport lesson, extended 2026-08-06 when
        # shared merge windows went default-ON): drop the optimization the
        # decoder disagrees with, newest first, re-gate, and record what
        # this point actually ran with so the paper reports it honestly
        for label, over in (
                ("parallel_fallback", {"parallel_steps": False}),
                ("schedule_fallback", {"parallel_steps": False,
                                       "step_scheduling": False})):
            if not mis:
                break
            if all(kw.get(k2, True) == v2 for k2, v2 in over.items()):
                continue          # already effectively this config
            out[label] = True
            kw = dict(kw, **over)
            with contextlib.redirect_stdout(io.StringIO()):
                cp = compile_qasm(case.qasm, distance=d, **kw)
            noisy = inject_uniform_noise(cp.circuit, p)
            mis, n_mech, matching = _single_error_audit(noisy)
            out[f"miscorrected_singles_after_{label}"] = mis
        if mis:
            raise RuntimeError(
                f"decoder unfaithful: {mis}/{n_mech} single-error "
                f"mechanisms miscorrected — rate would be meaningless")
        sampler = noisy.compile_detector_sampler(seed=seed)
        n_obs = noisy.num_observables
        shots = 0
        joint_errors = 0
        per_obs = np.zeros(n_obs, dtype=np.int64)
        while joint_errors < target_errors and shots < max_shots:
            batch = min(BATCH, max_shots - shots)
            det, obs = sampler.sample(batch, separate_observables=True)
            pred = matching.decode_batch(det)
            wrong = pred.astype(bool) ^ obs.astype(bool)
            per_obs += wrong.sum(axis=0)
            joint_errors += int(wrong.any(axis=1).sum())
            shots += batch
        out.update(shots=shots, joint_errors=joint_errors,
                   per_obs_errors=per_obs.tolist(),
                   num_detectors=noisy.num_detectors,
                   num_observables=n_obs,
                   ler_joint=joint_errors / shots,
                   seconds=round(time.perf_counter() - t0, 2),
                   status="OK")
    except Exception as e:
        out.update(status=type(e).__name__, error=str(e)[:400],
                   trace_tail=traceback.format_exc().splitlines()[-3:],
                   seconds=round(time.perf_counter() - t0, 2))
    conn.send(out)
    conn.close()


def run_point(case, d, p, seed, target_errors, max_shots, timeout):
    parent, child = mp.Pipe()
    proc = mp.Process(target=_eval_point,
                      args=(case, d, p, seed, target_errors, max_shots, child))
    proc.start()
    if parent.poll(timeout):
        out = parent.recv()
    else:
        out = {"name": case.name, "d": d, "p": p, "seed": seed,
               "status": "TIMEOUT", "seconds": timeout}
        proc.terminate()
    proc.join(5)
    if proc.is_alive():
        proc.kill()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--d", nargs="+", type=int, default=[3])
    ap.add_argument("--p", nargs="+", type=float, default=[1e-3])
    ap.add_argument("--target-errors", type=int, default=100)
    ap.add_argument("--max-shots", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default="experiments/results/ler/main.jsonl")
    args = ap.parse_args()

    prov = provenance(allow_dirty=args.allow_dirty)
    by_name = {c.name: c for c in suite()}
    missing = [n for n in args.cases if n not in by_name]
    if missing:
        ap.error(f"unknown cases: {missing}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "a")
    fh.write(json.dumps({**prov, "args": vars(args)}) + "\n")
    fh.flush()

    points = [(by_name[n], d, p) for n in args.cases
              for d in args.d for p in args.p]
    print(f"{len(points)} points, target_errors {args.target_errors}, "
          f"max_shots {args.max_shots}, sha {prov['circls_sha'][:9]}",
          flush=True)

    from concurrent.futures import ThreadPoolExecutor

    def worker(idx_point):
        idx, (case, d, p) = idx_point
        out = run_point(case, d, p, args.seed + idx, args.target_errors,
                        args.max_shots, args.timeout)
        fh.write(json.dumps(out) + "\n")
        fh.flush()
        if out["status"] == "OK":
            extra = (f"ler={out['ler_joint']:.2e} "
                     f"({out['joint_errors']}/{out['shots']}) "
                     f"det={out['num_detectors']} t={out['seconds']}s")
        else:
            extra = out.get("error", "")[:70]
        print(f"[{out['status']:>7s}] {case.name:15s} d={d} p={p:g} {extra}",
              flush=True)
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(worker, enumerate(points)))
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\ndone: {ok}/{len(results)} OK -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
