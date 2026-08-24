"""A1(2) scale-ladder final table from ladder_ours_*.jsonl.

Reads every experiments/results/<set>/topols_compare/ladder_ours_*.jsonl,
takes the last OK row per (case, d), and renders the system-level LER
table (audit column = worst per-point miscorrected/mechanisms).
"""
import argparse
import glob
import json
from pathlib import Path

ORDER = ["bv_32", "bv_64", "bv_100", "dj_32", "dj_64", "dj_100"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir",
                    default="experiments/results/best5/topols_compare")
    args = ap.parse_args()
    indir = Path(args.indir)

    rows, sha = {}, "?"
    for f in sorted(glob.glob(str(indir / "ladder_ours_*.jsonl"))):
        for line in open(f):
            d = json.loads(line)
            if d.get("record") == "provenance":
                sha = d.get("circls_sha", sha)[:9]
                continue
            if d.get("status") == "OK" and "name" in d:
                rows[(d["name"], d["d"])] = d

    out = [f"# A1(2) scale ladder — FINAL (system-level LER, sha {sha})",
           "",
           "Baseline TopoLS+tqec: capability_x for every row below "
           "(spatial-Hadamard NotImplementedError, see topols_circuits "
           "manifest).",
           "LER = circuit + decoder as one system (audit counts stay in "
           "the jsonl, not tabulated).",
           "",
           "| case | d | qubits | rounds | qubit-rounds | LER p=1e-3 | "
           "LER p=5e-4 |",
           "|---|---|---|---|---|---|---|"]
    for name in ORDER:
        for d in (3, 5):
            r = rows.get((name, d))
            if r is None:
                out.append(f"| {name} | {d} | MISSING |")
                continue
            st = r.get("stats", {})
            qubits = st.get("S4_qubits_active")
            rounds = st.get("T1_rounds")
            qr = st.get("V2_qubit_rounds")
            pts = {pt["p"]: pt for pt in r.get("ler", [])}
            if not pts:
                l1 = l2 = "skipped (saturated)"
            else:
                l1 = (f"{pts[1e-3]['ler']:.4f}" if 1e-3 in pts else "—")
                l2 = (f"{pts[5e-4]['ler']:.4f}" if 5e-4 in pts else "—")
            out.append(f"| {name} | {d} | {qubits} | {rounds} | {qr} | "
                       f"{l1} | {l2} |")
    dest = indir / "ladder_table_final.md"
    dest.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
