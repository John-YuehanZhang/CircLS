"""Emit the LaTeX body of tab:dscaling (appendix) from the jsonl set.

Same data-selection rules as dscaling_segments.py: dynamic = full
(deep rows supersede), static = points_static / the d9-d11 runners;
no_live rows are not used.  Distances come from what is on disk, so
the table grows to d = 9, 11 when those files land.
"""
from pathlib import Path

from dscaling_segments import collect, PROGRAMS

LABEL = {"teleport_4": "Teleportation-4", "teleport_8": "Teleportation-8",
         "bv_8": "BV-8", "bv_16": "BV-16", "dj_8": "DJ-8",
         "dj_16": "DJ-16"}


def fmt(pts, name, cfg, d, p):
    r = pts.get((name, cfg, d, p))
    if r is None:
        return "--"
    ler = r["ler"]
    if ler is None or ler == 0:
        return "0"
    if ler >= 1e-3:
        return f"{ler:.4f}"
    exp = 0
    while ler < 1:
        ler *= 10
        exp -= 1
    return f"${ler:.1f}\\times10^{{{exp}}}$"


def main():
    pts = collect()
    ds = sorted({d for (_, _, d, _) in pts})
    cols = "ll" + "rr" * len(ds)
    print(f"  \\begin{{tabular}}{{{cols}}}")
    print("    \\toprule")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$d = {d}$}}" for d in ds)
    print(f"    & & {head} \\\\")
    sub = " & ".join("dyn. & static" for _ in ds)
    print(f"    program & $p$ & {sub} \\\\")
    print("    \\midrule")
    for name in PROGRAMS:
        for i, (p, ptxt) in enumerate(
                ((5e-4, "$5\\times10^{-4}$"), (1e-3, "$10^{-3}$"))):
            cells = " & ".join(
                f"{fmt(pts, name, 'dyn', d, p)} & "
                f"{fmt(pts, name, 'static', d, p)}" for d in ds)
            lead = LABEL[name] if i == 0 else ""
            print(f"    {lead} & {ptxt} & {cells} \\\\")
    print("    \\bottomrule")
    print("  \\end{tabular}")


if __name__ == "__main__":
    main()
