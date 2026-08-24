"""Our side of a layout-only comparison row (the DJ-16 recipe,
generalized).

For programs where the baseline's layout completes but its stim
conversion does not, the manifest entry carries the fill contract and
cube count without a circuit file.  This runner compiles the matched
QASM on our side and samples the system-level LER; the baseline's
circuit-level cells stay crosses.
"""
import contextlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

from compare_tqec import adaptive_ler, side_metrics   # noqa: E402
from compare_topols import CIRC_DIR, matching_qasm    # noqa: E402
from formula_deviation import (_audit_dict, _faithful_audit,  # noqa: E402
                               _mwpf_sample_ler)
from noise_inject import inject_uniform_noise         # noqa: E402
from provenance import provenance                     # noqa: E402

P_POINTS = [2e-3, 1e-3, 5e-4]


def main():
    import argparse
    from circls.pipeline import compile_qasm
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--seed", type=int, default=59)
    ap.add_argument("--original-qasm", action="store_true",
                    help="compile the suite program's own QASM (its "
                    "published terminal measurements) instead of a "
                    "fill-matched synthesis — the contract for rows "
                    "where the baseline never reaches a circuit")
    args = ap.parse_args()
    d = args.d
    seed = args.seed + 100 * d

    prov = provenance()
    entry = json.loads((CIRC_DIR / "manifest.json").read_text())[
        "entries"][args.pair]
    if args.original_qasm:
        from benchsuite import suite
        qasm = {c.name: c.qasm for c in suite()}[
            args.name.removeprefix("c45_")]
    else:
        qasm = matching_qasm(args.name, entry["fill"])

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(qasm, distance=d, assignment="optimized",
                              liveness=True, keep_patches=set())
        times.append(round(time.perf_counter() - t0, 3))

    row = {"pair": args.pair, "name": args.name, "d": d,
           "topols_cubes": entry.get("cubes"),
           "topols_status": entry.get("status"),
           "ours": side_metrics(cp.circuit),
           "ours_stats": to_dict(experiment_stats(cp.experiment,
                                                  cp.circuit)),
           "compile_seconds": times,
           "compile_median": statistics.median(times),
           "ler": []}
    for i, p in enumerate(P_POINTS):
        noisy = inject_uniform_noise(cp.circuit, p)
        pt_seed = seed + i
        t0 = time.perf_counter()
        try:
            import pymatching
            m = pymatching.Matching.from_detector_error_model(
                noisy.detector_error_model(decompose_errors=True))
            shots, joint, per = adaptive_ler(noisy, m, pt_seed, 100,
                                             1_000_000)
            # audit dropped: system-level LER ruling (user 2026-08-13)
            decoder, audit = "mwpm", {"skipped": "ruling 2026-08-13"}
        except ValueError as e:
            if ("Failed to decompose" not in str(e) and
                    "observable ids larger than 63" not in str(e)):
                raise
            shots, joint, audit = _mwpf_sample_ler(noisy, pt_seed, 100,
                                                   1_000_000)
            decoder = "mwpf"
        row["ler"].append({"p": p, "shots": shots, "joint_errors": joint,
                           "ler": joint / shots, "decoder": decoder,
                           "audit": audit, "seed": pt_seed,
                           "seconds": round(time.perf_counter() - t0, 1)})
        print(f"{args.name} p={p:g}: ler={joint/shots:.5f} ({decoder})",
              flush=True)

    out = (_ROOT / "experiments/results/best5/topols_compare/"
           f"layoutonly_{args.name}_d{d}.jsonl")
    with open(out, "a") as fh:
        fh.write(json.dumps({**prov, "record": "provenance"}) + "\n")
        fh.write(json.dumps(row) + "\n")
    st = row["ours_stats"]
    print(f"{args.name} d={d}: V1={st['V1_volume_blocks']:.2f} "
          f"qr={st['V2_qubit_rounds']} rounds={st['T1_rounds']} "
          f"compile={row['compile_median']}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
