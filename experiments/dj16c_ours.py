"""Our side of the DJ-16 comparison row (Clifford oracle, dj16_c).

Design decision 2026-08-11: the DJ-16 row uses the Clifford variant.  The
baseline's own dj_16.qasm carries 6 t gates (a Toffoli-style balanced
oracle), outside the Clifford-only scope; dj16_c implements a linear
balanced oracle.  Their layout runs on dj16_c (repo defaults; paper
flags crash their MCTS), their stim conversion does not (spatial
vertical Hadamard unimplemented in tqec), so the fill contract is
recorded in the manifest without a circuit file and only our side has
circuit-level numbers.

Protocol identical to compare_topols.py: matched QASM from the fill's
port bases, d=3, system-level LER with mwpf fallback, plus a timed
recompile (3 reps, median) for the compile-time cell.
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
    ap.add_argument("--d", type=int, default=3)
    args = ap.parse_args()
    d = args.d
    k = (d - 1) // 2
    seed = 47 + 100 * d

    prov = provenance()
    # the fill contract and cube count are k-independent (port bases come
    # from the layout, not the scale factor) — one manifest entry serves
    # every d
    entry = json.loads((CIRC_DIR / "manifest.json").read_text())[
        "entries"]["dj16_c_k1_f0"]
    qasm = matching_qasm("dj16_c", entry["fill"])

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(qasm, distance=d, assignment="optimized",
                              liveness=True, keep_patches=set())
        times.append(round(time.perf_counter() - t0, 3))

    row = {"pair": f"dj16_c_k{k}_f0", "name": "dj16_c", "k": k, "d": d,
           "topols_cubes": entry["cubes"],
           "topols_status": entry["status"],
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
            m, mis, n_mech = _faithful_audit(noisy)
            shots, joint, per = adaptive_ler(noisy, m, pt_seed, 100,
                                             1_000_000)
            decoder, audit = "mwpm", _audit_dict(mis, n_mech)
        except ValueError as e:
            if "Failed to decompose" not in str(e):
                raise
            shots, joint, audit = _mwpf_sample_ler(noisy, pt_seed, 100,
                                                   1_000_000)
            decoder = "mwpf"
        row["ler"].append({"p": p, "shots": shots, "joint_errors": joint,
                           "ler": joint / shots, "decoder": decoder,
                           "audit": audit, "seed": pt_seed,
                           "seconds": round(time.perf_counter() - t0, 1)})
        print(f"p={p:g}: ler={joint/shots:.5f} ({decoder})", flush=True)

    out = (_ROOT / "experiments/results/best5/topols_compare/"
           f"dj16c_ours_d{d}.jsonl")
    with open(out, "a") as fh:
        fh.write(json.dumps({**prov, "record": "provenance"}) + "\n")
        fh.write(json.dumps(row) + "\n")
    st = row["ours_stats"]
    print(f"V1={st['V1_volume_blocks']:.2f} V3={st['V3_bbox_volume_blocks']:.2f} "
          f"qr={st['V2_qubit_rounds']} rounds={st['T1_rounds']} "
          f"compile={row['compile_median']}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
