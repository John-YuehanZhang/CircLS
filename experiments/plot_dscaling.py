"""Rebuild figures/dscaling_dj16.pdf: DJ-16 LER against code distance,
dynamic (full pipeline) vs static allocation, p = 5e-4.

Data sources (results/best5/dscaling/):
* points_ext2.jsonl        full,   d = 3, 5, 7
* points_ext2_deep.jsonl   full,   d = 5, 7 at 5e-4 (400-error deep
                           points; SUPERSEDE the ext2 rows)
* points_static.jsonl      static, d = 3, 5, 7
* points_d9.jsonl          both configs, d = 9
* points_d11.jsonl         both configs, d = 11 (skipped if absent)

Output: dscaling_dj16.pdf + .png next to the data (copy the PDF into
the paper's figures/ by hand; the Overleaf tree is not this repo).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent / "results" / "best5" / "dscaling"
P = 5e-4
PROGRAM = "dj_16"


def rows(fname):
    f = HERE / fname
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if (r.get("record") == "point" and r.get("name") == PROGRAM
                and r.get("p") == P):
            yield r


def collect():
    pts = {}                              # (config, d) -> row
    for fn in ("points_ext2.jsonl", "points_static.jsonl",
               "points_ext2_deep.jsonl",   # later files supersede
               "points_d9.jsonl", "points_d11.jsonl"):
        for r in rows(fn):
            cfg = {"full": "dynamic", "static": "static"}.get(r["config"])
            if cfg:
                pts[(cfg, r["d"])] = r
    return pts


QEC = Path(__file__).resolve().parent / "results" / "tclass" \
    / "points_adder_panel.jsonl"


def collect_qec():
    pts = {}
    if not QEC.exists():
        return pts
    for line in QEC.read_text().splitlines():
        r = json.loads(line)
        cfg = {"full": "dynamic", "static": "static"}.get(r["config"])
        if cfg:
            pts[(cfg, r["d"])] = r
    return pts


def _draw(ax, pts, tag):
    for cfg, color, marker in (("static", "#c0392b", "s"),
                               ("dynamic", "#1f77b4", "o")):
        ds = sorted(d for c, d in pts if c == cfg)
        if not ds:
            continue
        ler = [pts[(cfg, d)]["ler"] for d in ds]
        err = [(pts[(cfg, d)]["ler"]
                * (1 - pts[(cfg, d)]["ler"])
                / pts[(cfg, d)]["shots"]) ** 0.5 for d in ds]
        ax.errorbar(ds, ler, yerr=err, color=color, marker=marker,
                    markersize=4.5, linewidth=1.8, capsize=2, label=cfg)
        print(tag, cfg, {d: f"{l:.3e}" for d, l in zip(ds, ler)})
    ax.set_yscale("log")
    ax.set_ylabel("LER")
    if pts:
        ax.set_xticks(sorted({d for _, d in pts}))
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)


def main():
    pts = collect()
    # side by side, SAME overall footprint as the old single panel
    # independent y-axes: panel (b)'s dynamic-vs-static gap is small and
    # gets flattened if it shares (a)'s wide range
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.8, 2.3))
    _draw(ax_a, pts, "(a)")
    ax_a.legend(loc="upper right", framealpha=0.9, fontsize=7,
                handlelength=1.4, borderpad=0.3)
    ax_a.text(0.06, 0.05, "(a) DJ-16", transform=ax_a.transAxes,
              fontsize=8, fontweight="bold")
    ax_a.set_xlabel("code distance $d$")
    ax_a.tick_params(labelsize=8)
    # panel (b): qec_en_n5 under the Y-state approximation
    _draw(ax_b, collect_qec(), "(b)")
    ax_b.set_ylabel("")          # (a)'s LER label serves both panels
    ax_b.set_xlabel("code distance $d$")
    ax_b.tick_params(labelsize=8)
    ax_b.text(0.06, 0.05, "(b) adder_n4", transform=ax_b.transAxes,
              fontsize=8, fontweight="bold")
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        out = HERE / f"dscaling_dj16.{ext}"
        fig.savefig(out, dpi=200)
        print("wrote", out)


if __name__ == "__main__":
    main()
