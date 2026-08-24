"""Generate the A2 ablation table from experiments/results/ablation/*.jsonl.

Zero hand-copied numbers: this script is the ONLY path from raw results to
the paper table.  It refuses mixed provenance (all configs must come from
the same clean commit) unless --allow-mixed.

Outputs (to the results dir): ablation_table.md (human review),
ablation_table.csv (archive), and with --tex a LaTeX tabular body.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

METRICS = [("V1", lambda s: s["V1_volume_blocks"]),
           ("rounds", lambda s: s["T1_rounds"]),
           ("peak", lambda s: s["S3_tiles_peak"]),
           ("L1", lambda s: s["L1_live_tile_rounds"])]
# L2 utilization is a RATIO — aggregated as sum(L1)/sum(tile_rounds), never
# a mean of per-case ratios
_TILE_ROUNDS = lambda s: s["internal"]["tile_rounds"]     # noqa: E731
# columns read "what happens to <metric> when this component is REMOVED":
# + == the component was saving that much (user misread 'vs full', 2026-08-05).
# TWO relative columns (design decision 2026-08-06): components are judged on
# the metric they target — volume components on V1, parallel step execution
# on rounds (a latency optimization keeps V1 flat by construction: the
# shared window runs the same tiles concurrently).
_REL_COLS = [("V1 +% when removed", "V1"),
             ("rounds +% when removed", "rounds")]


def load(outdir: Path, allow_mixed: bool):
    runs, provs = {}, {}
    for f in sorted(outdir.glob("*.jsonl")):
        cfg = f.stem
        rows = [json.loads(l) for l in f.open()]
        provs[cfg] = [r for r in rows if r.get("record") == "provenance"]
        # last result per (name) wins — reruns append
        per = {}
        for r in rows:
            if r.get("record") == "provenance":
                continue
            per[r["name"]] = r
        runs[cfg] = per
    shas = {p["circls_sha"] for ps in provs.values() for p in ps}
    dirty = any(p["dirty"] for ps in provs.values() for p in ps)
    if (len(shas) > 1 or dirty) and not allow_mixed:
        sys.exit(f"provenance check failed: shas={sorted(s[:9] for s in shas)} "
                 f"dirty={dirty} — rerun from one clean commit or pass "
                 f"--allow-mixed for exploratory reading")
    return runs, sorted(shas), dirty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments/results/ablation")
    ap.add_argument("--allow-mixed", action="store_true")
    ap.add_argument("--tex", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    runs, shas, dirty = load(outdir, args.allow_mixed)
    if "full" not in runs:
        sys.exit("no full.jsonl — the baseline column is mandatory")

    configs = ["full"] + [c for c in ("no_reduce", "no_place", "no_fui",
                                      "no_schedpar",
                                      "no_live", "no_sched", "no_parallel")
                          if c in runs]
    names = sorted({n for per in runs.values() for n in per},
                   key=lambda n: (runs["full"].get(n, {}).get("n", 0), n))

    # aggregate: sums over cases OK in EVERY config (fair comparison set),
    # plus per-config failure counts over all attempted cases
    common = [n for n in names
              if all(runs[c].get(n, {}).get("status") == "OK"
                     for c in configs)]
    agg = {c: {m: sum(fn(runs[c][n]["stats"]) for n in common)
               for m, fn in METRICS} for c in configs}
    for c in configs:
        tr = sum(_TILE_ROUNDS(runs[c][n]["stats"]) for n in common)
        agg[c]["L2"] = agg[c]["L1"] / tr if tr else 0.0
    fails = {c: sorted(n for n in names
                       if runs[c].get(n, {}).get("status") not in (None, "OK"))
             for c in configs}

    def _rels(a, c):
        return [((a[c][m] / a["full"][m] - 1) * 100 if a["full"][m] else 0.0)
                for _, m in _REL_COLS]

    md = [f"# A2 ablation ({len(common)} common-OK cases; "
          f"sha {','.join(s[:9] for s in shas)}{' DIRTY' if dirty else ''})",
          "", "| config | " + " | ".join(m for m, _ in METRICS)
          + " | L2 | " + " | ".join(h for h, _ in _REL_COLS)
          + " | failures |",
          "|---|" + "---|" * (len(METRICS) + 2 + len(_REL_COLS))]
    for c in configs:
        md.append(f"| {c} | "
                  + " | ".join(f"{agg[c][m]:.0f}" for m, _ in METRICS)
                  + f" | {agg[c]['L2']:.2f} | "
                  + " | ".join(f"{r:+.1f}%" for r in _rels(agg, c))
                  + f" | {len(fails[c])}"
                  + (f" ({', '.join(fails[c][:4])}"
                     + ("…" if len(fails[c]) > 4 else "") + ")"
                     if fails[c] else "") + " |")
    # liveness only acts on consumable cases (elsewhere every patch is a
    # kept output) — the all-case aggregate dilutes its row, so report the
    # applicable subset separately
    cons = [n for n in common
            if "consumable" in runs["full"][n].get("tags", ())]
    if cons:
        cagg = {c: {m: sum(fn(runs[c][n]["stats"]) for n in cons)
                    for m, fn in METRICS} for c in configs}
        for c in configs:
            tr = sum(_TILE_ROUNDS(runs[c][n]["stats"]) for n in cons)
            cagg[c]["L2"] = cagg[c]["L1"] / tr if tr else 0.0
        md += ["", f"## consumable subset ({len(cons)} cases — where "
               "liveness applies)", "",
               "| config | " + " | ".join(m for m, _ in METRICS)
               + " | L2 | " + " | ".join(h for h, _ in _REL_COLS) + " |",
               "|---|" + "---|" * (len(METRICS) + 1 + len(_REL_COLS))]
        for c in configs:
            md.append(f"| {c} | "
                      + " | ".join(f"{cagg[c][m]:.0f}" for m, _ in METRICS)
                      + f" | {cagg[c]['L2']:.2f} | "
                      + " | ".join(f"{r:+.1f}%" for r in _rels(cagg, c))
                      + " |")
    for sec, fn0 in (("V1", METRICS[0][1]), ("rounds", METRICS[1][1])):
        md += ["", f"## per-case {sec}", "",
               "| case | " + " | ".join(configs) + " |",
               "|---|" + "---|" * len(configs)]
        for n in names:
            cells = []
            for c in configs:
                r = runs[c].get(n)
                if r is None:
                    cells.append("—")
                elif r["status"] != "OK":
                    cells.append("T/O" if r["status"] == "TIMEOUT"
                                 else f"✗{r['status'][:12]}")
                else:
                    cells.append(f"{fn0(r['stats']):.0f}")
            md.append(f"| {n} | " + " | ".join(cells) + " |")

    (outdir / "ablation_table.md").write_text("\n".join(md) + "\n")
    with (outdir / "ablation_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "config", "status"] + [m for m, _ in METRICS]
                   + ["build_seconds"])
        for n in names:
            for c in configs:
                r = runs[c].get(n)
                if r is None:
                    continue
                vals = ([f"{fn(r['stats']):.6g}" for _, fn in METRICS]
                        if r["status"] == "OK" else [""] * len(METRICS))
                w.writerow([n, c, r["status"]] + vals
                           + [r.get("build_seconds", "")])
    if args.tex:
        tex = [r"% generated by experiments/ablation_table.py — do not edit",
               r"\begin{tabular}{l" + "r" * (len(METRICS) + 1
                                             + len(_REL_COLS)) + "}"]
        tex.append("config & " + " & ".join(m for m, _ in METRICS)
                   + " & L2 & "
                   + " & ".join(h.replace("%", r"\\%") for h, _ in _REL_COLS)
                   + r" \\ \hline")
        for c in configs:
            tex.append(f"{c.replace('_', ' ')} & "
                       + " & ".join(f"{agg[c][m]:.0f}" for m, _ in METRICS)
                       + f" & {agg[c]['L2']:.2f} & "
                       + " & ".join(f"{r:+.1f}\\%" for r in _rels(agg, c))
                       + r" \\")
        tex.append(r"\end{tabular}")
        (outdir / "ablation_table.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md[:12]))
    print(f"\n-> {outdir}/ablation_table.md, .csv"
          + (", .tex" if args.tex else ""))


if __name__ == "__main__":
    main()
