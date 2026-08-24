"""Generate and archive the tqec comparison circuits (A1(1)).

Run with the DEDICATED tqec env (baselines/tqec pinned v0.2.0):

    conda run -n tqec python \
        experiments/generate_tqec_circuits.py

Emits NOISELESS stim circuits — the comparison applies ONE shared noise
pass to both sides (experiments/noise_inject.py), so neither compiler's
built-in noise model participates.  Per entry x k: the port fill chosen by
the max-observables search (2026-08-05), archived alongside a manifest
with the tqec version/SHA and circuit facts.
"""
import itertools
import json
import subprocess
from pathlib import Path

import tqec
import tqec.gallery as gallery
from tqec.compile.compile import compile_block_graph
from tqec.computation.cube import ZXCube

OUT = Path(__file__).resolve().parent / "results" / "tqec_circuits"

ENTRIES = ["memory", "stability", "cnot", "cz", "move_rotation",
           "three_cnots", "steane_encoding"]


def candidate_fills(fn):
    graph = fn()
    ports = dict(graph.ports)
    labels = sorted(ports)
    cand = {}
    for lab in labels:
        pos = ports[lab]
        pipes = [p for p in graph.pipes
                 if p.u.position == pos or p.v.position == pos]
        kinds = set()
        for p in pipes:
            ks = str(p.kind)
            for rep in ("Z", "X"):
                try:
                    kinds.add(ZXCube.from_str(ks.replace("O", rep)))
                except Exception:
                    pass
        cand[lab] = sorted(kinds, key=str) or [ZXCube.from_str("ZXZ")]
    return labels, cand


def best_circuit(fn, k, budget=128):
    labels, cand = candidate_fills(fn)
    if not labels:
        graph = fn()
        comp = compile_block_graph(graph)
        return {}, comp.generate_stim_circuit(k=k)
    best = None
    for combo in itertools.islice(
            itertools.product(*(cand[l] for l in labels)), budget):
        graph = fn()
        try:
            graph.fill_ports(dict(zip(labels, combo)))
            comp = compile_block_graph(graph)
            circ = comp.generate_stim_circuit(k=k)
            # tqec compiles fills whose observables are RANDOM at p=0 (cz
            # measured 2026-08-05) — require noiseless determinism, or the
            # LER comparison would decode coin flips
            det, obs = circ.compile_detector_sampler(seed=0).sample(
                64, separate_observables=True)
            if det.any() or (obs != obs[0]).any():
                continue
            key = (circ.num_observables, -circ.num_qubits)
            if best is None or key > best[0]:
                best = (key, dict(zip(labels, [str(c) for c in combo])), circ)
        except Exception:
            continue
    if best is None:
        return None, None
    return best[1], best[2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(
        ["git", "-C", str(Path(tqec.__file__).resolve().parents[2]),
         "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {"tqec_version": tqec.__version__, "tqec_sha": sha,
                "entries": {}}
    for name in ENTRIES:
        fn = getattr(gallery, name)
        for k in (1, 2):
            fill, circ = best_circuit(fn, k)
            if circ is None:
                print(f"{name} k={k}: NO deterministic-observable fill",
                      flush=True)
                manifest["entries"][f"{name}_k{k}"] = {"status": "no_fill"}
                continue
            fname = f"{name}_k{k}.stim"
            (OUT / fname).write_text(str(circ))
            manifest["entries"][f"{name}_k{k}"] = {
                "fill": fill, "num_qubits": circ.num_qubits,
                "num_detectors": circ.num_detectors,
                "num_observables": circ.num_observables,
                "file": fname}
            print(f"{name} k={k}: qubits={circ.num_qubits} "
                  f"det={circ.num_detectors} obs={circ.num_observables}",
                  flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
