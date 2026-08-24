"""Appendix per-program table: full-pipeline results for the 45-program
roster at d=3, from the frozen ablation full-config run.

Renders a two-column-group tabular body (23 + 22 rows) with
human-readable benchmark names (paper rule: no code identifiers).
"""
import json
from pathlib import Path

SRC = Path("experiments/results/best5/ablation/full.jsonl")
OUT = Path("experiments/results/best5/ablation/perprogram_table.tex")

PRETTY = {
    "steane_encode": "Steane encoder",
    "ghz_16_mixed": "GHZ-16 (mixed)",
    "twistedghz_4": "Twisted GHZ-4", "twistedghz_8": "Twisted GHZ-8",
    "teleport_4": "Teleportation-4", "teleport_8": "Teleportation-8",
    "bbpssw_4": "BBPSSW-4", "bbpssw_8": "BBPSSW-8",
}
for fam, label in (("ghz", "GHZ"), ("bv", "BV"), ("dj", "DJ"),
                   ("graphstate", "Graph state")):
    for n in (8, 16, 32, 64):
        PRETTY[f"{fam}_{n}"] = f"{label}-{n}"


def pretty(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    return r"\texttt{%s}" % name.replace("_", r"\_")   # QASMBench id


def fmt(v, nd=1):
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v:.0f}"


def main():
    cov = {}
    covsrc = SRC.parent.parent / "topols_compare/coverage45.jsonl"
    for line in open(covsrc):
        r = json.loads(line)
        if r.get("record") != "case":
            continue
        lay = r["layout"]
        lcell = ("P" if lay.get("settings") == "paper" else
                 "D" if lay.get("settings") == "defaults" else
                 r"$\times$")
        ccell = (r"\checkmark" if r["circuit"]["status"] == "OK"
                 else r"$\times$")
        cov[r["name"]] = (lcell, ccell)
    rows = []
    for line in open(SRC):
        r = json.loads(line)
        if not r.get("name") or r.get("record") == "provenance":
            continue
        st = r["stats"]
        rows.append((r["name"], r["n"], st["V1_volume_blocks"],
                     st["V3_bbox_volume_blocks"], st["V2_qubit_rounds"],
                     st["T1_rounds"], r["build_seconds"]))
    rows.sort(key=lambda x: (x[1], x[0]))
    cells = []
    for name, n, v1, v3, qr, t1, cs in rows:
        lc, cc = cov[name]
        cells.append(f"    {pretty(name)} & {n} & {fmt(v1)} & {fmt(v3)}"
                     f" & {qr} & {t1} & {fmt(cs)} & {lc} & {cc} \\\\")
    half = (len(cells) + 1) // 2
    left, right = cells[:half], cells[half:]
    right += ["    & & & & & & & & \\\\"] * (len(left) - len(right))
    body = "\n".join(
        l.rstrip("\\\\").rstrip() + " & " + r.lstrip()
        for l, r in zip(left, right))
    OUT.write_text(body + "\n")
    print(f"{len(rows)} programs -> {OUT}")


if __name__ == "__main__":
    main()
