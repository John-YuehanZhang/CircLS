"""Sharded precision LER for the GHZ-16 d5 comparison row.

The serial harness cannot reach 10k failures at d=5: the baseline
circuit decodes at ~0.37 s/shot (mwpm, 27k detectors) and our side at
~0.25 s/shot even on a 32-way mwpf pool.  This harness shards the mwpm
side across a spawn Pool and drives the mwpf side through one wide
pool, one process per (side, p) point, each appending its own row.

Feasible-target matrix (design decision 2026-08-11: 10k where reachable):
  p=1e-3  both sides 10k failures   (the cell tab:comparison reports)
  p=5e-4  mwpm 10k, mwpf 1k         (10k on mwpf = ~62 h, out of reach)
  p=2e-3  both sides 1k
  p=2e-4  not run at d5 (d3 covers it; 4M-shot cap alone = days)

Usage:
  --side topols --p 1e-3 --target 10000 --workers 24
  --side ours   --p 1e-3 --target 10000        (pool via MWPF_WORKERS)
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

PAIR = "ghz_16_k2_f1"
OUT = _ROOT / ("experiments/results/best5/topols_compare/"
               "ghz16_k2_precision_sharded.jsonl")


def _t_init(circ_text, dem_text):
    global _SRC, _MATCH
    import pymatching
    import stim
    _SRC = stim.Circuit(circ_text)
    _MATCH = pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel(dem_text))


def _t_batch(task):
    seed, shots = task
    det, obs = _SRC.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    pred = _MATCH.decode_batch(det)
    errs = int((pred.astype(bool) != obs.astype(bool)).any(axis=1).sum())
    return shots, errs


def main():
    from compare_topols import CIRC_DIR, matching_qasm
    from formula_deviation import _audit_dict, _faithful_audit
    from noise_inject import inject_uniform_noise
    from provenance import provenance

    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["topols", "ours"], required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--target", type=int, default=10000)
    ap.add_argument("--max-shots", type=int, default=4_000_000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--est-ler", type=float, required=True,
                    help="rough LER from the 100-failure run, sizes waves")
    ap.add_argument("--seed", type=int, default=131)
    args = ap.parse_args()
    prov = provenance()

    import stim
    manifest = json.loads((CIRC_DIR / "manifest.json").read_text())
    entry = manifest["entries"][PAIR]
    t0 = time.perf_counter()
    if args.side == "topols":
        circ = stim.Circuit((CIRC_DIR / entry["file"]).read_text())
    else:
        from circls.pipeline import compile_qasm
        with contextlib.redirect_stdout(io.StringIO()):
            circ = compile_qasm(matching_qasm("ghz_16", entry["fill"]),
                                distance=5, assignment="optimized",
                                liveness=True, keep_patches=set()).circuit
    noisy = inject_uniform_noise(circ, args.p)

    row = {"record": "point", "pair": PAIR, "side": args.side,
           "p": args.p, "d": 5, "target": args.target,
           "workers": args.workers, "seed": args.seed}
    if args.side == "ours":
        # non-graphlike at d=5 -> mwpf, already pooled via MWPF_WORKERS
        from formula_deviation import _mwpf_sample_ler
        shots, errs, audit = _mwpf_sample_ler(noisy, args.seed,
                                              args.target, args.max_shots)
        row.update({"decoder": "mwpf", "audit": audit})
    else:
        m, mis, n_mech = _faithful_audit(noisy)
        dem = noisy.detector_error_model(decompose_errors=True)
        shots = errs = 0
        seed_i = args.seed
        with mp.get_context("spawn").Pool(
                args.workers, initializer=_t_init,
                initargs=(str(noisy), str(dem))) as pool:
            while errs < args.target and shots < args.max_shots:
                need = (args.target - errs) / max(args.est_ler, 1e-9)
                wave = min(int(1.2 * need) + args.workers,
                           args.max_shots - shots)
                per = max(2000, wave // args.workers + 1)
                tasks = []
                while wave > 0:
                    tasks.append((seed_i, min(per, wave)))
                    seed_i += 1
                    wave -= per
                for s, e in pool.imap_unordered(_t_batch, tasks):
                    shots += s
                    errs += e
                print(f"[{args.side} p={args.p:g}] {errs}/{shots}",
                      flush=True)
        row.update({"decoder": "mwpm", "audit": _audit_dict(mis, n_mech)})
    row.update({"shots": shots, "joint_errors": errs,
                "ler": errs / shots,
                "seconds": round(time.perf_counter() - t0, 1)})
    with open(OUT, "a") as fh:
        fh.write(json.dumps({**prov, "record": "provenance",
                             "args": vars(args)}) + "\n")
        fh.write(json.dumps(row) + "\n")
    print(f"[{args.side} p={args.p:g}] ler={errs/shots:.6f} "
          f"({errs}/{shots}) {row['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
