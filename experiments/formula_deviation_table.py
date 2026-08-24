"""Summarize B-dev: measured program LER vs the composition formula.

Reads deviation.jsonl (append-only, last row per (name, d, p) wins),
reports per-(d, p) deviation statistics in the LINEAR regime
(measured < 0.1 and prediction < 0.1: near saturation both the joint
LER and the capped prediction compress toward 1 and the ratio is
meaningless), plus the calibration-vs-published block rates and the
excluded cases.  Markdown to <outdir>/deviation_table.md.
"""
import argparse
import json
import statistics
from collections import OrderedDict
from pathlib import Path

LIN = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments/results/formula_dev")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    cases, cals, prov = OrderedDict(), {}, None
    for line in open(outdir / "deviation.jsonl"):
        r = json.loads(line)
        rec = r.get("record")
        if rec == "case":
            cases[(r["name"], r["d"], r["p"])] = r
        elif rec == "calibration":
            cals[(r["d"], r["p"])] = r      # last wins: recal rows override
        elif rec == "provenance":
            prov = r
    # ratios recomputed against the LATEST calibration (case rows store
    # the ratio against whichever calibration was current when they ran)
    for (nm, d, p), r in cases.items():
        if r["status"] == "OK":
            eps = cals[(d, p)]["eps_block"]
            r["pred_calibrated"] = min(1.0, r["V1"] * eps)
            r["ratio_calibrated"] = (r["measured"] / r["pred_calibrated"]
                                     if r["pred_calibrated"] else None)

    md = [f"# B-dev: measured LER vs V1-block composition formula "
          f"(sha {prov['circls_sha'][:9]})", "",
          f"prediction = V1_blocks x eps_block;  published: "
          f"a={prov['pub_a']} (p/{prov['pub_pth']})^((d+1)/2);  calibrated: "
          f"single-patch memory slope through the same pipeline "
          f"(all compiler passes OFF, plain protocol)", "",
          "## calibration vs published eps_block", "",
          "| d | p | calibrated | published | cal/pub |", "|---|---|---|---|---|"]
    for (d, p), c in sorted(cals.items()):
        pub = prov["pub_a"] * (p / prov["pub_pth"]) ** ((d + 1) / 2)
        md.append(f"| {d} | {p:g} | {c['eps_block']:.3e} | {pub:.3e} | "
                  f"{c['eps_block'] / pub:.2f} |")

    md += ["", "## deviation in the linear regime "
           f"(measured < {LIN} and prediction < {LIN})", "",
           "| d | p | cases | median meas/pred (cal) | IQR | "
           "median meas/pred (pub) | excluded (saturated) | failed |",
           "|---|---|---|---|---|---|---|---|"]
    for (d, p) in sorted({(d, p) for _, d, p in cases}):
        rows = [r for (nm, dd, pp), r in cases.items()
                if dd == d and pp == p]
        ok = [r for r in rows if r["status"] == "OK"]
        lin = [r for r in ok if r["measured"] < LIN
               and r["pred_calibrated"] < LIN and r["joint_errors"] >= 5]
        sat = [r for r in ok if r not in lin]
        fails = [r for r in rows if r["status"] != "OK"]
        if lin:
            rc = sorted(r["ratio_calibrated"] for r in lin)
            rp = sorted(r["ratio_published"] for r in lin)
            med = statistics.median(rc)
            q1 = rc[len(rc) // 4]
            q3 = rc[(3 * len(rc)) // 4]
            medp = statistics.median(rp)
            md.append(f"| {d} | {p:g} | {len(lin)} | {med:.2f} | "
                      f"[{q1:.2f}, {q3:.2f}] | {medp:.2f} | {len(sat)} | "
                      f"{len(fails)} |")
        else:
            md.append(f"| {d} | {p:g} | 0 | — | — | — | {len(sat)} | "
                      f"{len(fails)} |")

    md += ["", "## per-case ratio (calibrated), linear-regime rows", "",
           "| case | n | d | p | V1 | measured | predicted | ratio |",
           "|---|---|---|---|---|---|---|---|"]
    for (nm, d, p), r in sorted(cases.items()):
        if r["status"] != "OK":
            continue
        if r["measured"] < LIN and r["pred_calibrated"] < LIN \
                and r["joint_errors"] >= 5:
            md.append(f"| {nm} | {r['n']} | {d} | {p:g} | {r['V1']:.0f} | "
                      f"{r['measured']:.2e} | {r['pred_calibrated']:.2e} | "
                      f"{r['ratio_calibrated']:.2f} |")
    md += ["", "## failures / timeouts", "",
           "| case | d | p | status |", "|---|---|---|---|"]
    for (nm, d, p), r in sorted(cases.items()):
        if r["status"] != "OK":
            md.append(f"| {nm} | {d} | {p:g} | {r['status']} |")

    (outdir / "deviation_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:30]))
    print(f"\n-> {outdir}/deviation_table.md")


if __name__ == "__main__":
    main()
