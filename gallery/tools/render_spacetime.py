"""Render the 3D spacetime block structure of a compiled experiment.

x, y = coarse tile grid; z = code cycle.  A data patch's live range is a
solid column; the tiles an ancilla path holds during a merge window show
as translucent gray bridges.  The voxel count matches the paper's
allocated-spacetime-volume metric (blocks = tile-rounds / d).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# tab10 without the gray slot (index 7) -- gray is reserved for corridors
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
            "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"]
_CORRIDOR = "#9aa0a6"


def render(stats, placement, name, out_dir, *, distance, circuit=None,
           elev=18, azim=-60):
    """stats: circls.metrics.experiment_stats result;
    placement: {patch name: (cx, cy)} coarse cells."""
    patch_live = stats.patch_live
    tile_occ = stats.tile_occupancy
    n_rounds = max(max(rs) for rs in tile_occ.values()) + 1
    xs = [t[0] for t in tile_occ]
    ys = [t[1] for t in tile_occ]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1

    # legend only for patches that actually own voxels: a patch born into
    # its first merge and retired at its last (zero standalone rounds) is
    # absent from patch_live and its tile renders as corridor occupancy
    patch_names = sorted(nm for nm in placement if nm in patch_live)
    pidx = {nm: i for i, nm in enumerate(patch_names)}
    grid = -np.ones((nx, ny, n_rounds), dtype=int)
    for (cx, cy), rounds in tile_occ.items():
        for r in rounds:
            grid[cx - x0, cy - y0, r] = len(patch_names)   # default: corridor
    for nm, (cx, cy) in placement.items():
        if nm not in patch_live:
            continue
        lo, hi = patch_live[nm]
        for r in range(lo, hi + 1):
            if grid[cx - x0, cy - y0, r] >= 0:
                grid[cx - x0, cy - y0, r] = pidx[nm]

    colors = np.empty(grid.shape, dtype=object)
    for i in range(len(patch_names)):
        colors[grid == i] = _PALETTE[i % len(_PALETTE)] + "E6"
    colors[grid == len(patch_names)] = _CORRIDOR + "66"

    fig = plt.figure(figsize=(9, 11))
    ax = fig.add_subplot(projection="3d")
    ax.voxels(grid >= 0, facecolors=colors,
              edgecolor="#00000030", linewidth=0.15)
    ax.set_xlabel("tile x")
    ax.set_ylabel("tile y")
    ax.set_zlabel("code cycle")
    ax.set_box_aspect((nx, ny, max(nx, ny) * 1.6))
    ax.view_init(elev=elev, azim=azim)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=_PALETTE[i % len(_PALETTE)])
               for i in range(len(patch_names))]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=_CORRIDOR))
    ax.legend(handles, patch_names + ["ancilla path / corridor"],
              loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=8)
    ax.set_title(f"{name}  d={distance}  spacetime occupancy "
                 f"({stats.volume_blocks:.1f} blocks)")
    png = f"{out_dir}/{name}_spacetime.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    def to_ranges(rounds):
        rounds = sorted(rounds)
        out, s, p = [], rounds[0], rounds[0]
        for r in rounds[1:]:
            if r == p + 1:
                p = r
            else:
                out.append([s, p])
                s = p = r
        out.append([s, p])
        return out

    corridor = {}
    for (cx, cy), rounds in tile_occ.items():
        rem = [r for r in rounds
               if grid[cx - x0, cy - y0, r] == len(patch_names)]
        if rem:
            corridor[f"{cx},{cy}"] = to_ranges(rem)
    sched = {
        "program": name, "distance": distance,
        "grid": {"x": [x0, x1], "y": [y0, y1]},
        "code_cycles": n_rounds,
        "allocated_volume_blocks": stats.volume_blocks,
        "patches": [{"name": nm, "tile": list(placement[nm]),
                     "birth_cycle": patch_live[nm][0],
                     "retire_cycle": patch_live[nm][1]}
                    for nm in patch_names if nm in patch_live],
        "corridor_occupancy": corridor,
    }
    js = f"{out_dir}/{name}_schedule.json"
    json.dump(sched, open(js, "w"), indent=1)

    stim_path = None
    if circuit is not None:
        stim_path = f"{out_dir}/{name}.stim"
        open(stim_path, "w").write(str(circuit))
    return png, js, stim_path
