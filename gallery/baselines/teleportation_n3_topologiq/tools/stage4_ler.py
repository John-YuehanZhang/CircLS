"""Section 4: tqec compile at d = 3, 5, 7 and the logical error rate.

Each distance is compiled in a fresh interpreter (compile_worker.py): tqec's emitted circuit
depends on state left by an earlier compile in the same process.  Compiled this way the d = 3
and d = 5 circuits are byte-identical to the ones behind the paper's Table 2 (sha256 prefixes
below; the check is recorded in data/4_<name>_compile_d<d>.json).
"""
import hashlib
import io
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import stim

from ler_plot import ler_figure, save_ler_points
from ler_sampler import measure_ler
from noise_inject import inject_uniform_noise
from saved import record_saved

RECORDED_SHA = {3: "4b0deee58b8fc52e", 5: "2b1031163e319e84"}


def graphlike_distance(circ, p=1e-3):
    """Smallest number of noise mechanisms that flip an observable without firing a detector
    (stim's graphlike search under this entry's noise pass)."""
    err = inject_uniform_noise(circ, p).search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False)
    return len(err)


def compile_distances(here, name, meta, ks=(1, 2, 3), p_noisy=1e-3, max_distance_search=5):
    """-> {d: clean stim.Circuit}.  Writes stim_circuit/d<d>/4_<name>_{clean,noisy_p1e-3}.stim."""
    circuits = {}
    for k in ks:
        d = 2 * k + 1
        out = here / "stim_circuit" / f"d{d}"
        out.mkdir(parents=True, exist_ok=True)
        clean = out / f"4_{name}_clean.stim"
        rep_file = here / "data" / f"4_{name}_compile_d{d}.json"
        subprocess.run([sys.executable, str(here / "tools" / "compile_worker.py"),
                        str(here / "data" / f"2_{name}_topologiq.bgraph"), str(meta["n_data"]),
                        str(here / "data" / f"1_{name}.qasm"), str(k), str(clean), str(rep_file)], check=True)
        rep = json.loads(rep_file.read_text())
        assert rep["convention"], rep["errors"]
        circ = stim.Circuit(clean.read_text())
        (out / f"4_{name}_noisy_p{p_noisy:g}.stim").write_text(str(inject_uniform_noise(circ, p_noisy)))
        sha = hashlib.sha256(clean.read_text().encode()).hexdigest()[:16]
        rep["sha256_prefix"] = sha
        rep["byte_identical_to_table2_circuit"] = (RECORDED_SHA[d] == sha) if d in RECORDED_SHA else None
        if d <= max_distance_search:      # the search cost grows quickly with d
            rep["graphlike_distance"] = graphlike_distance(circ)
        else:
            rep["graphlike_distance"] = None
            rep["graphlike_distance_note"] = f"not searched (only d <= {max_distance_search} is)"
        rep_file.write_text(json.dumps(rep, indent=1))
        circuits[d] = circ
    entries = []
    for d in circuits:
        entries += [(here / "stim_circuit" / f"d{d}" / f"4_{name}_clean.stim", f"d={d} circuit, noiseless"),
                    (here / "stim_circuit" / f"d{d}" / f"4_{name}_noisy_p{p_noisy:g}.stim", f"d={d} circuit with p={p_noisy:g} noise"),
                    (here / "data" / f"4_{name}_compile_d{d}.json", f"d={d} compile report (convention, p=0 check, Table 2 sha check)")]
    record_saved(entries)
    return circuits


def ler_curve(here, name, circuits, ps=(5e-4, 1e-3, 2e-3, 1e-2, 2e-2), workers=8, title=None):
    """PyMatching LER vs p per distance, shown inline.

    Points are cached in data/4_<name>_ler_points.json together with the sha256 prefix of the circuit
    they were sampled from, so a cache left over from other circuits (a different seed, traversal mode
    or set of distances) is detected and resampled instead of being plotted as if it were this run's.
    Delete the file to resample everything.  ``workers`` processes sample in parallel; the estimate does
    not depend on how many (each shard has its own seed), but each one holds a copy of the decoder, so
    keep it small on a laptop.
    """
    from IPython.display import Image, display
    pts_file = here / "data" / f"4_{name}_ler_points.json"
    shas = {d: json.loads((here / "data" / f"4_{name}_compile_d{d}.json").read_text()).get("sha256_prefix")
            for d in circuits}
    cached, stale = {}, []
    if pts_file.exists():
        raw = json.loads(pts_file.read_text())
        for k, v in raw.items():
            d, p = int(k.split("_p")[0][1:]), float(k.split("_p")[1])
            if d in circuits and p in ps and v.get("circuit_sha256_prefix") in (None, shas[d]):
                cached[(d, p)] = (v["ler"], v["errors"], v["shots"])
            elif d in circuits and p in ps:
                stale.append((d, p))
    wanted = [(d, p) for d in circuits for p in ps]
    missing = [k for k in wanted if k not in cached]
    reused = len(cached) > 0 and not missing
    if missing:
        if stale:
            print(f"{len(stale)} cached point(s) came from different circuits and are being resampled")
        print(f"sampling {len(missing)} LER point(s) with {workers} worker processes ...")
        points = dict(cached)
        for d, p in missing:
            points[(d, p)] = measure_ler(inject_uniform_noise(circuits[d], p), workers=workers, decoder="mwpm")
        save_ler_points(points, str(pts_file), shas=shas)
    else:
        points = cached
    fig = ler_figure(points, list(ps), title or f"{name}, topologiq + tqec")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue(), width=560))
    note = "reused from the cache shipped with this entry" if reused else "sampled by this run"
    record_saved([(pts_file, f"LER points ({note}); delete the file to resample")],
                 figures_inline=("LER plot",))
    return points
