"""Standalone measurement-reduction correctness record (user request
2026-08-05: keep the verification as citable evidence).

Runs the execution-level equivalence check (same as
tests/test_reduction_equivalence.py) and FREEZES the outcome per case:
the deterministic-parity affine structure of both compilations (masks +
values, the complete invariant of a stabilizer output distribution),
shot counts, seeds, per-case placement, and provenance.  The paper's
"the pass provably does not change the computation" sentence points at
this file's frozen output.

Usage:
    python experiments/verify_reduction.py \
        [--shots 512] [--seed 23] [--out experiments/results/reduction_equivalence/record.jsonl]
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

from benchsuite import (bbpssw_chain, bv, dj, ghz, graph_state_ring,  # noqa: E402
                        steane_encode, twisted_ghz)
from provenance import provenance                                     # noqa: E402

from circls.pipeline import compile_qasm                      # noqa: E402
from circls.tools.reporting import (affine_structure,               # noqa: E402
                                      sample_program_bits)

CASES = [("ghz_8", lambda: ghz(8)), ("graphstate_8", lambda: graph_state_ring(8)),
         ("bv_8", lambda: bv(8)), ("dj_8", lambda: dj(8)),
         ("steane", steane_encode), ("twistedghz_4", lambda: twisted_ghz(4)),
         ("twistedghz_8", lambda: twisted_ghz(8)),
         ("bbpssw_4", lambda: bbpssw_chain(4))]
# per-case placement keeping BOTH configs inside their feasibility domain
PLACEMENT = {"steane": "optimized"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default="experiments/results/"
                    "reduction_equivalence/record.jsonl")
    args = ap.parse_args()
    prov = provenance(allow_dirty=args.allow_dirty)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "a")
    fh.write(json.dumps({**prov, "args": vars(args)}) + "\n")

    all_ok = True
    for name, fn in CASES:
        qasm = fn()
        row = {"case": name, "shots": args.shots, "seed": args.seed,
               "placement": PLACEMENT.get(name, "row_major")}
        try:
            structs = {}
            for tag, red in (("off", False), ("on", True)):
                with contextlib.redirect_stdout(io.StringIO()):
                    cp = compile_qasm(qasm, assignment=row["placement"],
                                      measure_reduction=red)
                bits = sample_program_bits(cp, args.shots, args.seed)
                masks, vals = affine_structure(bits)
                structs[tag] = {"masks": [format(m, "x") for m in masks],
                                "values": vals, "n_bits": bits.shape[1]}
            row["off"] = structs["off"]
            row["on"] = structs["on"]
            row["identical"] = structs["off"] == structs["on"]
        except Exception as e:
            row["identical"] = False
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        all_ok &= row["identical"]
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(f"{name:14s} identical={row['identical']}"
              + (f"  ({row.get('error', '')})" if not row["identical"]
                 else f"  constraints={len(row['on']['masks'])}"),
              flush=True)
    print(("ALL IDENTICAL" if all_ok else "MISMATCH PRESENT")
          + f" -> {args.out}", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
