"""Section 3: topologiq .bgraph -> tqec block graph.

open_port  = topologiq's graph as tqec reads it (after the Hadamard shim): the circuit's
             wire ends are tqec Port cubes, zero volume, not drawn.
closed_port = every port filled the way the QASM prepares and measures it (data wires Z in /
             Z out, magic wires X in / Z out; see topologiq_bridge.fill_rule), the graph
             that is compiled, sampled and counted.
"""
import json
import pathlib
import tempfile

from saved import record_saved
from snapshot_html import snapshot
from topologiq_bridge import (FIXED_BOUNDARY_CONVENTION, BlockGraph, ZXCube, build, compile_block_graph,
                              equivalent, fill_rule)


def build_closed_graph(here, name, original_qasm, meta):
    """-> (g_open, g_closed, fill, surfaces).  Checks the open graph against the QASM's
    Clifford skeleton before filling; writes data/3_<name>_bridge.json."""
    g_open, info = build(here / "data" / f"2_{name}_topologiq.bgraph", meta["n_data"], name=name)
    ok, why, (lead, trail, idle) = equivalent(g_open, original_qasm, drop_s=True)
    assert ok, why
    fill = fill_rule(g_open, lead, trail)
    g_closed = g_open.clone()
    g_closed.fill_ports({lb: ZXCube.from_str(v) for lb, v in fill.items()})
    g_closed.validate()
    surfaces = g_closed.find_correlation_surfaces()
    (here / "data" / f"3_{name}_bridge.json").write_text(json.dumps(
        {"bridge": info, "equivalence": why, "absorbed_h": {"in": sorted(lead), "out": sorted(trail)},
         "fill": fill, "cubes_open": g_open.num_cubes - g_open.num_ports, "ports": g_open.num_ports,
         "cubes_closed": g_closed.num_cubes, "pipes": g_closed.num_pipes, "observables": len(surfaces)}, indent=1))
    return g_open, g_closed, fill, surfaces


def export_blockgraphs(here, name, g_open, g_closed):
    """Write blockgraph/<name>_{open_port,closed_port}.{dae,html} and validate the closed one with
    tqec's own toolchain (dae round trip, compile at k=1, p=0 sample) -> blockgraph/tqec_validation.txt."""
    bg = here / "blockgraph"
    bg.mkdir(exist_ok=True)
    for tag, g in (("open_port", g_open), ("closed_port", g_closed)):
        g.to_dae_file(str(bg / f"{name}_{tag}.dae"))
        g.view_as_html(write_html_filepath=str(bg / f"{name}_{tag}.html"))
    g2 = BlockGraph.from_dae_file(str(bg / f"{name}_closed_port.dae"))
    lines = [f"{name}_closed_port.dae: {g2.num_cubes} cubes, {g2.num_pipes} pipes (round-trip via BlockGraph.from_dae_file)"]
    circ = compile_block_graph(g2, observables="auto", convention=FIXED_BOUNDARY_CONVENTION).generate_stim_circuit(
        k=1, do_not_use_database=True)
    lines.append(f"tqec compile (k=1, d=3, fixed_boundary): {circ.num_qubits} qubits, "
                 f"{circ.num_detectors} detectors, {circ.num_observables} observables")
    det, obs = circ.compile_detector_sampler(seed=0).sample(256, separate_observables=True)
    deterministic = all(bool((obs[:, i] == obs[0, i]).all()) for i in range(obs.shape[1]))
    lines.append(f"p=0 sample (256 shots, seed 0): silent={not det.any()}, all observables deterministic={deterministic}")
    (bg / "tqec_validation.txt").write_text("\n".join(lines) + "\n")
    record_saved([(bg / f"{name}_open_port.dae", "3D model, ports open (loads into tqec / SketchUp)"),
                        (bg / f"{name}_open_port.html", "rotatable viewer, ports open"),
                        (bg / f"{name}_closed_port.dae", "3D model, ports filled -- the compiled graph"),
                        (bg / f"{name}_closed_port.html", "rotatable viewer, ports filled"),
                        (bg / "tqec_validation.txt", "closed graph reloaded, compiled and sampled at p=0 by tqec"),
                        (here / "data" / f"3_{name}_bridge.json", "shim stats, equivalence result, port fill")])
    return lines


def show_closed_graph(here, name, width=640):
    """Snapshot of the closed graph (headless Chrome, if available) plus the rotatable viewer, inline.
    The viewer is WebGL, so the still image needs a browser; without one only the viewer is shown."""
    from IPython.display import IFrame, Image, display
    html = here / "blockgraph" / f"{name}_closed_port.html"
    with tempfile.TemporaryDirectory() as tmp:
        png = pathlib.Path(tmp) / "snap.png"
        if snapshot(html, png):
            display(Image(data=png.read_bytes(), width=width))
            record_saved([], figures_inline=("closed-graph snapshot",))
        else:
            print(f"no Chrome/Chromium found, so no still image of the block graph; set CHROME=<browser binary> "
                  f"for one, or open blockgraph/{name}_closed_port.html (rotatable, below).")
    display(IFrame(f"blockgraph/{name}_closed_port.html", width=760, height=480))
