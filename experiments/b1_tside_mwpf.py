"""B1 cross-decoder check: GHZ-16 d5 baseline side decoded with MWPF.

Table 2's GHZ-16 d5 LER cell compares T(mwpm) against C(mwpf).  This
run decodes the SAME baseline circuit with mwpf at p=1e-3 so the 22%
gap can be attributed to the circuits, not the decoders.  Reuses
_mwpf_sample_ler (MWPF_WORKERS pool), same pair and seed lineage as
the precision harness.

Everything lives under a __main__ guard: the mwpf pool uses the
spawn context, which re-imports this file in every worker, and
module-level provenance() would re-run there (and die on a dirty
tree mid-run — the crash of 2026-08-12).
"""
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

PAIR = "ghz_16_k2_f1"
P = float(sys.argv[sys.argv.index("--p") + 1]) if "--p" in sys.argv else 1e-3
TARGET = (int(sys.argv[sys.argv.index("--target") + 1])
          if "--target" in sys.argv else 1000)
OUT = _ROOT / ("experiments/results/best5/topols_compare/"
               "ghz16_k2_tside_mwpf.jsonl")


def main():
    import stim

    from compare_topols import CIRC_DIR
    from formula_deviation import _mwpf_sample_ler
    from noise_inject import inject_uniform_noise
    from provenance import provenance

    prov = provenance()
    entry = json.loads(
        (CIRC_DIR / "manifest.json").read_text())["entries"][PAIR]
    noisy = inject_uniform_noise(
        stim.Circuit((CIRC_DIR / entry["file"]).read_text()), P)
    t0 = time.perf_counter()
    shots, errs, audit = _mwpf_sample_ler(noisy, 131, TARGET, 4_000_000)
    row = {**prov, "record": "b1_cross_decoder", "pair": PAIR,
           "side": "topols", "decoder": "mwpf", "p": P, "d": 5,
           "target": TARGET, "shots": shots, "errors": errs,
           "ler": errs / shots, "audit": audit,
           "seconds": round(time.perf_counter() - t0, 1)}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"T-side mwpf p={P:g}: {errs}/{shots} = {errs/shots:.5f} "
          f"(mwpm reference 0.08939) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
