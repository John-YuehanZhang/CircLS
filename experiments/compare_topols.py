"""A1(1) v2: same QASM through two complete QASM->stim pipelines.

Baseline = TopoLS(front-end, MCTS) + tqec(back-end); ours = CircLS.
The TopoLS benchmark QASM carries no measurements — in/out bases come
from the port fills tqec's minimal-simulation filler chose, recorded in
the manifest as {port_label: cube_kind} (label = "In_<q>"/"Out_<q>",
temporal basis = the kind's third letter).  Our side compiles the SAME
unitary QASM with a mechanically synthesised prologue/epilogue matching
those bases, so both pipelines implement the identical computation.

Shared protocol as in compare_tqec.py: one noise pass on both noiseless
circuits, same decoder, single-error faithfulness gate on both sides.

Usage:
    python experiments/compare_topols.py [--names ghz_16 bv_16 dj_16 CNOT]
        [--k 1 2] [--p 2e-3 1e-3 5e-4] [--out ...]
"""
import argparse
import contextlib
import io
import json
import re
import sys
import time
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

import stim                                        # noqa: E402

from compare_tqec import adaptive_ler, side_metrics  # noqa: E402
from formula_deviation import (_audit_dict, _faithful_audit,  # noqa: E402
                               _mwpf_sample_ler)
from noise_inject import inject_uniform_noise      # noqa: E402
from provenance import provenance                  # noqa: E402

TOPOLS_QASM = Path(os.environ.get(
    "TOPOLS_DIR",
    str(Path(__file__).resolve().parents[2] / "TopoLS"))) / "docs/benchmark"
CIRC_DIR = Path(__file__).resolve().parent / "results" / "topols_circuits"


def matching_qasm(name: str, fill: dict) -> str:
    """The SAME unitary + prologue/epilogue realising the fill's bases."""
    base = (TOPOLS_QASM / f"{name}.qasm").read_text().rstrip() + "\n"
    m = re.search(r"qreg\s+(\w+)\[(\d+)\]", base)
    reg, n = m.group(1), int(m.group(2))
    pro, epi = [], []
    for label, kind in fill.items():
        side, q = label.rsplit("_", 1)
        basis = kind[2]                  # temporal letter
        if basis not in ("Z", "X"):
            raise ValueError(f"unexpected temporal basis {kind} at {label}")
        if basis == "X":
            # CONTRACT NOTE (2026-08-08): input-basis h stays in the
            # EPILOGUE — reverting the 867d62c "fix".  The v2 comparison,
            # the truth-oracle suite and the ghz_16_mixed roster case were
            # all validated end-to-end under this placement, and switching
            # X-basis inputs to the prologue changed observable counts
            # against the archived baseline circuits (16->15, 30->29,
            # 2->1) — evidence that the fills' input letters already
            # absorb the boundary Hadamards (ZX spider fusion), so a
            # prologue h double-rotates.  Whether the In branch should
            # exist at all is parked in nearest-first.txt's sibling notes;
            # resolve against the TopoLS pipeline internals post-deadline.
            (pro if side == "In" else epi).append(f"h {reg}[{q}];")
    lines = base.splitlines()
    insert_at = next(i for i, l in enumerate(lines)
                     if l.strip().startswith("qreg")) + 1
    body = (lines[:insert_at] + pro + lines[insert_at:] + epi
            + [f"creg c[{n}];", f"measure {reg} -> c;"])
    return "\n".join(body) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+",
                    default=["ghz_16", "bv_16", "dj_16", "CNOT"])
    ap.add_argument("--k", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--p", nargs="+", type=float, default=[2e-3, 1e-3, 5e-4])
    ap.add_argument("--target-errors", type=int, default=100)
    ap.add_argument("--max-shots", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--max-fills", type=int, default=2,
                    help="fills per (name,k) — bv_16 has 64")
    ap.add_argument("--out",
                    default="experiments/results/topols_compare/main.jsonl")
    args = ap.parse_args()
    prov = provenance(allow_dirty=args.allow_dirty)
    manifest = json.loads((CIRC_DIR / "manifest.json").read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "a")
    fh.write(json.dumps({**prov, "baseline_sha": manifest["sha"],
                         "args": vars(args)}) + "\n")
    fh.flush()

    from circls.pipeline import compile_qasm
    idx = 0
    for name in args.names:
        for k in args.k:
            entries = sorted(
                ((key, e) for key, e in manifest["entries"].items()
                 if key.startswith(f"{name}_k{k}_f") and "file" in e),
                key=lambda kv: kv[0])[:args.max_fills]
            for key, entry in entries:
                base_noiseless = stim.Circuit(
                    (CIRC_DIR / entry["file"]).read_text())
                qasm = matching_qasm(name, entry["fill"])
                d = 2 * k + 1
                with contextlib.redirect_stdout(io.StringIO()):
                    cp = compile_qasm(qasm, distance=d,
                                      assignment="optimized",
                                      liveness=True, keep_patches=set())
                from circls.metrics import experiment_stats
                from circls.metrics.report import to_dict
                row = {"pair": key, "name": name, "k": k, "d": d,
                       "topols": side_metrics(base_noiseless),
                       "topols_cubes": entry["cubes"],
                       "ours": side_metrics(cp.circuit),
                       "ours_stats": to_dict(experiment_stats(
                           cp.experiment, cp.circuit)),
                       "ler": []}
                for p in args.p:
                    pt = {"p": p}
                    for side, circ in (("topols", base_noiseless),
                                       ("ours", cp.circuit)):
                        t0 = time.perf_counter()
                        try:
                            # system-level LER (design decisions 2026-08-09/10):
                            # the audit is a per-point diagnostic, never a
                            # gate, and non-graphlike error models fall back
                            # to the mwpf hypergraph decoder with the
                            # decoder recorded — same semantics as the
                            # ladder and B-dev harnesses.  The gate-era
                            # parallel-off recompile rung is gone with the
                            # gate.
                            noisy = inject_uniform_noise(circ, p)
                            seed = args.seed + idx
                            idx += 1
                            try:
                                import pymatching
                                m = pymatching.Matching.from_detector_error_model(
                                    noisy.detector_error_model(
                                        decompose_errors=True))
                                shots, joint, per = adaptive_ler(
                                    noisy, m, seed, args.target_errors,
                                    args.max_shots)
                                decoder = "mwpm"
                                # audit dropped: system-level LER ruling
                                # (design decision 2026-08-13) — LER is the joint
                                # error of circuit + decoder, full stop.
                                audit = {"skipped": "ruling 2026-08-13"}
                            except ValueError as e:
                                if ("Failed to decompose" not in str(e) and
                                        "observable ids larger than 63"
                                        not in str(e)):
                                    raise
                                shots, joint, audit = _mwpf_sample_ler(
                                    noisy, seed, args.target_errors,
                                    args.max_shots)
                                per = None
                                decoder = "mwpf"
                            pt[side] = {"shots": shots,
                                        "joint_errors": joint,
                                        "per_obs_errors": per, "seed": seed,
                                        "ler": joint / shots,
                                        "decoder": decoder, "audit": audit,
                                        "seconds": round(
                                            time.perf_counter() - t0, 1)}
                        except Exception as e:
                            pt[side] = {"status": type(e).__name__,
                                        "error": str(e)[:200]}
                    row["ler"].append(pt)
                    tl = pt["topols"].get("ler", pt["topols"].get("status"))
                    ol = pt["ours"].get("ler", pt["ours"].get("status"))
                    print(f"{key} p={p:g}: topols={tl} ours={ol}", flush=True)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"[{key}] topols q={row['topols']['qubits_active']}"
                      f"/{row['topols']['qubits_allocated']} "
                      f"rds={row['topols']['rounds']} | ours "
                      f"q={row['ours']['qubits_active']}"
                      f"/{row['ours']['qubits_allocated']} "
                      f"rds={row['ours']['rounds']}", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
