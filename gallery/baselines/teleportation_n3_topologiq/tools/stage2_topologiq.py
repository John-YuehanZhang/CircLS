"""Section 2: topologiq, QASM -> ZX diagram (pyzx) -> block graph (.bgraph)."""
import contextlib
import io
import json
import time

import matplotlib.pyplot as plt
from topologiq.core.graph_manager import graph_manager as gm
from topologiq.input.zx_manager import ZXGraphManager

from bgraph_plot import parse_bgraph_text
from saved import record_saved


def run_topologiq(here, name, mode="bfs-cross", seed=1):
    """Run topologiq on data/1_<name>_gadgets.qasm.  -> (bgraph_text, console_text, manager, stats).
    Writes data/2_<name>_topologiq.bgraph, the console log and a stats json.
    ``seed`` must be non-zero: topologiq tests it for truth (graph_manager.py ``if kwargs["seed"]``),
    so 0 leaves its RNG unseeded.  This graph has no spider of degree > 4, the only place topologiq
    draws random numbers, so the layout is the same for every seed."""
    gm.BGRAPH_DIR = here / "data"
    zxm = ZXGraphManager()
    aug = zxm.add_graph_from_qasm(path_to_qasm_file=here / "data" / f"1_{name}_gadgets.qasm", graph_key=name)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        mgr = gm.BlockGraphManager(aug, debug=0, seed=seed, graph_traverse_mode=mode)
        mgr.build()
        mgr.write_bgraph(f"2_{name}_topologiq")
    seconds = time.perf_counter() - t0
    text = (here / "data" / f"2_{name}_topologiq.bgraph").read_text()
    console = "\n".join(line for line in buf.getvalue().splitlines() if line.strip())
    cubes, pipes = parse_bgraph_text(text)
    kinds = {}
    for _, k, _ in cubes:
        kinds[k] = kinds.get(k, 0) + 1
    zg = aug.zx_graph
    stats = {"mode": mode, "seed": seed, "seconds": round(seconds, 3),
             "zx_vertices": zg.num_vertices(), "zx_edges": zg.num_edges(),
             "zx_inputs": len(zg.inputs()), "zx_outputs": len(zg.outputs()),
             "cubes": sum(v for k, v in kinds.items() if k != "OOO"), "ports": kinds.get("OOO", 0),
             "pipes": len(pipes), "h_pipes": sum(k.endswith("H") for *_, k in pipes), "cube_kinds": kinds,
             "same_as_recorded": same_as_recorded(here, name, text)}
    (here / "data" / f"2_{name}_topologiq_console.txt").write_text(console + "\n")
    (here / "data" / f"2_{name}_topologiq_run.json").write_text(json.dumps(stats, indent=1))
    record_saved([(here / "data" / f"2_{name}_topologiq.bgraph", "topologiq's block graph (its native format)"),
                        (here / "data" / f"2_{name}_topologiq_console.txt", "topologiq's console output"),
                        (here / "data" / f"2_{name}_topologiq_run.json", "run statistics; same_as_recorded = equals the paper's run")],
                 figures_inline=("ZX diagram", "topologiq 3D view"))
    return text, console, mgr, stats


def same_as_recorded(here, name, text):
    """True when the cubes and pipes equal the graph recorded for the paper's run (None if no record)."""
    rec = here / "data" / f"2_{name}_topologiq_recorded.bgraph"
    if not rec.exists():
        return None
    rc, rp = parse_bgraph_text(rec.read_text())
    c, p = parse_bgraph_text(text)
    key = lambda pipes: {(frozenset((a, b)), k) for a, b, k in pipes}
    return set(rc) == set(c) and key(rp) == key(p)


def zx_diagram(mgr, width=760):
    """The ZX diagram topologiq laid out (pyzx's drawing of the ingested circuit graph), shown inline."""
    import pyzx as zx
    from IPython.display import Image, display
    fig = zx.draw_matplotlib(mgr.aug_zx.zx_graph, labels=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue(), width=width))


def topologiq_view(mgr, width=640):
    """topologiq's own 3D rendering of the block graph, shown inline.  Its interactive window also
    carries a search box, buttons and a ZX overlay panel; only the 3D axes are kept here."""
    from IPython.display import Image, display
    vis = mgr.draw_blockgraph(is_final_vis=True, embedded=True)
    fig, ax3d = vis.view_3d.fig, vis.view_3d.ax
    from matplotlib.offsetbox import AnchoredOffsetbox
    for ax in list(fig.axes):          # search box, buttons, ZX overlay panel
        if ax is not ax3d:
            ax.remove()
    for art in ax3d.artists:           # the HUD boxes anchored on the 3D axes
        if isinstance(art, AnchoredOffsetbox):
            art.set_visible(False)
    ax3d.set_position([0.02, 0.02, 0.96, 0.96])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close("all")
    display(Image(data=buf.getvalue(), width=width))
