"""Sharded, audit-free LER runner for giant Table-2 rows.

One process = one (pair, side); each p point shards sampling across a
spawn Pool (the ghz16_d5_precision pattern).  No per-point audit:
system-level LER ruling (user 2026-08-13) — the LER is the joint
error rate of circuit + decoder, nothing else is recorded.

Everything sits under a __main__ guard: spawn workers re-import this
file, and module-level provenance() would die on a dirty tree.
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
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))


def _w_init(circ_text, dem_text):
    global _SRC, _MATCH
    import pymatching
    import stim
    _SRC = stim.Circuit(circ_text)
    _MATCH = pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel(dem_text))


def _w_batch(task):
    seed, shots = task
    det, obs = _SRC.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    pred = _MATCH.decode_batch(det)
    errs = int((pred.astype(bool) != obs.astype(bool)).any(axis=1).sum())
    return shots, errs


def main():
    import stim

    from compare_topols import CIRC_DIR, matching_qasm
    from compare_tqec import side_metrics
    from formula_deviation import _mwpf_sample_ler
    from noise_inject import inject_uniform_noise
    from provenance import provenance

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--side", choices=["topols", "ours"], required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--p", nargs="+", type=float, required=True)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--max-shots", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--seed", type=int, default=419)
    args = ap.parse_args()

    prov = provenance()
    entry = json.loads(
        (CIRC_DIR / "manifest.json").read_text())["entries"][args.pair]
    row_static = {"record": "giant_static", "pair": args.pair,
                  "name": args.name, "side": args.side, "d": args.d,
                  "topols_cubes": entry.get("cubes")}
    if args.side == "topols":
        circuit = stim.Circuit((CIRC_DIR / entry["file"]).read_text())
    else:
        from circls.pipeline import compile_qasm
        from circls.metrics import experiment_stats
        from circls.metrics.report import to_dict
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(matching_qasm(args.name, entry["fill"]),
                              distance=args.d, assignment="optimized",
                              liveness=True, keep_patches=set())
        circuit = cp.circuit
        row_static["ours_stats"] = to_dict(
            experiment_stats(cp.experiment, cp.circuit))
    row_static["metrics"] = side_metrics(circuit)

    OUT = _ROOT / ("experiments/results/best5/topols_compare/"
                   f"giant_{args.name}_{args.side}_d{args.d}.jsonl")
    fh = open(OUT, "a")
    fh.write(json.dumps({**prov, "record": "provenance",
                         "args": vars(args)}) + "\n")
    fh.write(json.dumps(row_static) + "\n")
    fh.flush()

    seed_i = args.seed
    for p in args.p:
        t0 = time.perf_counter()
        noisy = inject_uniform_noise(circuit, p)
        pt = {"record": "giant_point", "pair": args.pair,
              "name": args.name, "side": args.side, "d": args.d, "p": p,
              "workers": args.workers}
        try:
            dem = noisy.detector_error_model(decompose_errors=True)
            shots = errs = 0
            with mp.get_context("spawn").Pool(
                    args.workers, initializer=_w_init,
                    initargs=(str(noisy), str(dem))) as pool:
                wave = args.workers * 100
                while errs < args.target and shots < args.max_shots:
                    wave = min(wave, args.max_shots - shots)
                    per = max(25, wave // args.workers + 1)
                    tasks = []
                    left = wave
                    while left > 0:
                        tasks.append((seed_i, min(per, left)))
                        seed_i += 1
                        left -= per
                    for s, e in pool.imap_unordered(_w_batch, tasks):
                        shots += s
                        errs += e
                    print(f"[{args.name} {args.side} p={p:g}] "
                          f"{errs}/{shots}", flush=True)
                    if errs:
                        need = (args.target - errs) / (errs / shots)
                        wave = int(1.3 * need) + args.workers
                    else:
                        wave *= 4
            pt.update({"decoder": "mwpm", "shots": shots, "errors": errs,
                       "ler": errs / shots})
        except ValueError as e:
            if ("Failed to decompose" not in str(e) and
                    "observable ids larger than 63" not in str(e)):
                raise
            shots, errs, _ = _mwpf_sample_ler(noisy, seed_i, args.target,
                                              args.max_shots)
            seed_i += 1000
            pt.update({"decoder": "mwpf", "shots": shots, "errors": errs,
                       "ler": errs / shots})
        pt["seconds"] = round(time.perf_counter() - t0, 1)
        fh.write(json.dumps(pt) + "\n")
        fh.flush()
        print(f"{args.name} {args.side} p={p:g}: ler={pt['ler']:.5g} "
              f"({pt['decoder']}, {pt['seconds']}s)", flush=True)
    print("GIANT DONE", args.name, args.side, flush=True)


if __name__ == "__main__":
    main()
