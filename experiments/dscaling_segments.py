"""Suppression-ratio analysis for the d-scaling appendix.

For each (program, p) and each step d -> d+2 with both configs
sampled, R = (dynamic LER fall factor) / (static LER fall factor),
with a first-order standard error from the four binomial counts.
Anchors at LER >= 0.5 exclude the segment (near saturation).

Data: the same jsonl set as plot_dscaling.py, all six programs.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent / "results" / "best5" / "dscaling"
PROGRAMS = ["teleport_4", "teleport_8", "bv_8", "bv_16", "dj_8", "dj_16"]
FILES = ["points.jsonl", "points_ext.jsonl", "points_ext2.jsonl",
         "points_static.jsonl",
         "points_deep.jsonl", "points_ext_deep.jsonl",
         "points_ext2_deep.jsonl",           # deep supersedes
         "points_d9.jsonl", "points_d11.jsonl", "points_ext911.jsonl",
         "points_d11_deep.jsonl"]
SATURATION = 0.5


def collect():
    pts = {}
    for fn in FILES:
        f = HERE / fn
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("record") != "point" or r["name"] not in PROGRAMS:
                continue
            cfg = {"full": "dyn", "static": "static"}.get(r["config"])
            # no_live rows (liveness off, first-use init KEPT) are an
            # older arm the paper does not use: the static arm is
            # points_static / the d9-d11 runners (both stages removed)
            if cfg:
                pts[(r["name"], cfg, r["d"], r["p"])] = r
    return pts


def rel_se(r):
    if not r["ler"]:
        return float("inf")     # zero-failure point: unusable segment
    return math.sqrt(max(r["ler"] * (1 - r["ler"]) / r["shots"], 0.0)) \
        / r["ler"]


def main():
    pts = collect()
    segs = []
    for name in PROGRAMS:
        for p in (5e-4, 1e-3):
            ds = sorted({d for (n, c, d, pp) in pts
                         if n == name and pp == p})
            for d in ds:
                key = [(name, c, dd, p) for c in ("dyn", "static")
                       for dd in (d, d + 2)]
                if not all(k in pts for k in key):
                    continue
                a, b, c_, e = (pts[k] for k in key)
                if not all(x["ler"] for x in (a, b, c_, e)):
                    segs.append((name, p, d, None, None, "zero-failure"))
                    continue
                if a["ler"] >= SATURATION or c_["ler"] >= SATURATION:
                    segs.append((name, p, d, None, None, "saturated"))
                    continue
                R = (a["ler"] / b["ler"]) / (c_["ler"] / e["ler"])
                se = R * math.sqrt(sum(rel_se(x) ** 2
                                       for x in (a, b, c_, e)))
                segs.append((name, p, d, R, se, ""))
    usable = [s for s in segs if s[3] is not None]
    print(f"segments: {len(segs)} total, {len(usable)} usable, "
          f"{len(segs) - len(usable)} saturated-excluded")
    for name, p, d, R, se, note in segs:
        if R is None:
            print(f"  {name:12s} p={p:6g} {d}->{d+2}: EXCLUDED ({note})")
        else:
            flag = ""
            if R - 2 * se > 1:
                flag = "  [>1 by 2SE]"
            if R + 2 * se < 1:
                flag = "  [<1 by 2SE]"
            print(f"  {name:12s} p={p:6g} {d}->{d+2}: "
                  f"R={R:.2f} +- {se:.2f}{flag}")
    if usable:
        rs = [s[3] for s in usable]
        print(f"R range: {min(rs):.2f} .. {max(rs):.2f}")
        below = [s for s in usable if s[3] + s[4] < 1]
        print(f"below one by more than 1 SE: {len(below)}")
        above2 = [s for s in usable if s[3] - 2 * s[4] > 1]
        below2 = [s for s in usable if s[3] + 2 * s[4] < 1]
        print(f"above one by >2 SE: {len(above2)}; "
              f"below one by >2 SE: {len(below2)}")


if __name__ == "__main__":
    main()
