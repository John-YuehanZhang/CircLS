"""Generate and archive the TopoLS+tqec comparison circuits (A1(1) v2).

Architecture per the user's ruling (2026-08-05): the baseline is the
FULL QASM->stim pipeline TopoLS(front-end)+tqec(back-end), fed the SAME
QASM files as our compiler.  Chain per benchmark:

    docs/prog.py  -f NAME    qasm -> result/topols/NAME.pkl   (MCTS)
    docs/2tqec.py -f NAME    pkl  -> result/bgraph/NAME.bgraph
    THIS SCRIPT              bgraph -> BlockGraph
                             -> fill_ports_for_minimal_simulation()
                             -> compile -> NOISELESS stim (k=1,2)

Only fills whose observables are ALL deterministic at p=0 are archived
(tqec's minimal-simulation fills can emit coin-flip observables —
measured on ghz_16 fill 0).  Run with the tqec env.
"""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import tqec
from tqec import BlockGraph
from tqec.compile.compile import compile_block_graph
from tqec.computation.cube import Port
from tqec.utils.position import Position3D

TOPOLS = Path(os.environ.get(
    "TOPOLS_DIR",
    str(Path(__file__).resolve().parents[2] / "TopoLS")))
OUT = Path(__file__).resolve().parent / "results" / "topols_circuits"


def load_bgraph(name: str) -> BlockGraph:
    # pickle is TopoLS's own interchange format for .bgraph, and every file
    # loaded here was produced ON THIS MACHINE by our own docs/2tqec.py run
    # over our own QASM (never downloaded) — trusted-provenance input, kept
    # for compatibility with the upstream toolchain.
    data = pickle.load(open(TOPOLS / f"docs/result/bgraph/{name}.bgraph",
                            "rb"))
    g = BlockGraph("from_topols")
    for nd in data["bgraph_metadata"].values():
        pos = Position3D(*nd["position"])
        if pos not in g:
            if pd := nd["other"]:
                if not isinstance(pd, dict):
                    # TopoLS emits bare 'T' marker nodes for magic states
                    # (their dj_16.qasm carries 6 t gates) — out of the
                    # Clifford-only scope, refuse loudly
                    raise ValueError(
                        f"non-Clifford node {pd!r} in {name}.bgraph — this "
                        f"input is outside the Clifford-only comparison")
                g.add_cube(pos, Port(), f"{pd['type']}_{pd['qubit']}")
            else:
                g.add_cube(pos, nd["tqec"])
    for src, snk in data["edge_metadata"].values():
        s, t = Position3D(*src), Position3D(*snk)
        if not g.has_pipe_between(s, t):
            g.add_pipe(s, t)
    return g


def manual_candidates(name: str):
    """Two hand-built fills instead of the minimal-simulation enumeration.

    ``fill_ports_for_minimal_simulation`` enumerates fill combinations —
    measured 2026-08-06: bv_32's block graph has 64 ports and the
    enumeration never returns.  For the ladder comparison any
    DETERMINISTIC fill is a valid shared contract (our side synthesises
    the matching QASM from the fill), so we try all-Z ports and
    Z-in/X-out, both of which close bv/dj-style circuits
    deterministically.  fill_ports mutates, hence a fresh bgraph load per
    candidate."""
    from tqec.computation.cube import ZXCube
    cands = []
    for tag, in_b, out_b in (("mZ", "Z", "Z"), ("mX", "Z", "X"),
                             ("mXZ", "X", "Z"), ("mXX", "X", "X")):
        g = load_bgraph(name)
        ports = dict(g.ports)
        fill = {}
        for lb, pos in ports.items():
            # the port's cube kind is pinned by its pipe on the two walled
            # axes; only the OPEN (temporal) axis is a free basis choice —
            # a uniform kind fails tqec's colour validation (measured
            # 2026-08-06: ZXX against an XZO pipe)
            pipes = g.pipes_at(pos)
            if len(pipes) != 1:
                raise ValueError(f"port {lb} has {len(pipes)} pipes")
            pk = str(pipes[0].kind)
            # the pipe pins the two walled axes; the OPEN axis letter is
            # free.  The basis contract reads the CUBE's kind[2] (its
            # temporal wall) — for a spatial pipe (O at x/y: TopoLS routes
            # some outputs sideways at n>=32) kind[2] is pipe-pinned and
            # the free letter is basis-irrelevant, so the substitution
            # below stays correct for every pipe direction.
            basis = (in_b if lb.rsplit("_", 1)[0] in ("In", "input")
                     else out_b)
            kind = pk.replace("O", basis)
            if len(set(kind)) == 1:            # XXX/ZZZ invalid: flip the
                kind = pk.replace("O",         # free letter
                                  "X" if basis == "Z" else "Z")
            fill[lb] = kind
        g.fill_ports({lb: ZXCube.from_str(k) for lb, k in fill.items()})
        obs = g.find_correlation_surfaces()
        cands.append((tag, g, obs, fill))
    return cands


def main():
    # --max-keep N: archive at most N deterministic fills per (name, k);
    # --max-try M: give up after M fill compilations per (name, k).  The
    # n=100 ladder OOM-killed an uncapped run (2026-08-06: every fill of a
    # 100-qubit block graph compiled at k=1,2, ~30-40 s each, memory never
    # released between fills) — the comparison only consumes the first
    # couple of fills anyway (compare_topols --max-fills 2).
    # --manual-fills: skip the minimal-simulation enumeration entirely
    # (see manual_candidates).  --fragment: write a per-name manifest part
    # (manifest_part_<name>.json) so parallel per-name runs cannot clobber
    # the shared manifest; merge with merge_manifest_parts.py.
    args = sys.argv[1:]
    max_keep, max_try = 4, 16
    manual = fragment = False
    names = []
    it = iter(args)
    for a in it:
        if a == "--max-keep":
            max_keep = int(next(it))
        elif a == "--max-try":
            max_try = int(next(it))
        elif a == "--manual-fills":
            manual = True
        elif a == "--fragment":
            fragment = True
        else:
            names.append(a)
    names = names or ["ghz_16", "bv_16"]
    OUT.mkdir(parents=True, exist_ok=True)
    sha = {}
    for repo, path in (("topols", TOPOLS), ("tqec", Path(
            tqec.__file__).resolve().parents[2])):
        sha[repo] = subprocess.run(["git", "-C", str(path), "rev-parse",
                                    "HEAD"], capture_output=True,
                                   text=True).stdout.strip()
    mf_path = OUT / "manifest.json"
    manifest = (json.loads(mf_path.read_text()) if mf_path.exists()
                else {"sha": sha, "entries": {}})
    for name in names:
        entries = {}
        # candidates: (tag, filled graph, observables, fill dict) — the
        # fill records each port's cube kind; the TEMPORAL letter is that
        # qubit's init/readout basis, the contract the comparison runner
        # uses to synthesise our matching QASM
        if manual:
            cands = manual_candidates(name)
        else:
            g = load_bgraph(name)
            ports = dict(g.ports)                 # label -> Position3D
            fgs = g.fill_ports_for_minimal_simulation()
            cands = []
            for i, fg in enumerate(fgs):
                if i >= max_try:
                    break
                by_pos = {c.position: c for c in fg.graph.cubes}
                cands.append((f"f{i}", fg.graph, list(fg.observables),
                              {label: str(by_pos[pos].kind)
                               for label, pos in ports.items()}))
        # GEN_K env var filters the scale factors (e.g. GEN_K=1); the
        # 130-qubit k=2 generation is expensive enough to warrant it
        for k in tuple(int(x) for x in
                       os.environ.get("GEN_K", "1,2").split(",")):
            kept = 0
            for tag, g2, obs_list, fill in cands:
                if kept >= max_keep:
                    break
                comp = compile_block_graph(g2, observables=obs_list)
                circ = comp.generate_stim_circuit(k=k)
                det, obs = circ.compile_detector_sampler(seed=0).sample(
                    64, separate_observables=True)
                if det.any() or (obs != obs[0]).any():
                    continue                       # coin-flip fill: skip
                fname = f"{name}_k{k}_{tag}.stim"
                (OUT / fname).write_text(str(circ))
                entries[f"{name}_k{k}_{tag}"] = {
                    "source": "topols", "num_qubits": circ.num_qubits,
                    "num_detectors": circ.num_detectors,
                    "num_observables": circ.num_observables,
                    "cubes": g2.num_cubes, "pipes": g2.num_pipes,
                    "fill": fill, "file": fname}
                kept += 1
                print(f"{name} k={k} fill {tag}: qubits={circ.num_qubits} "
                      f"det={circ.num_detectors} obs={circ.num_observables}",
                      flush=True)
            if not kept:
                entries[f"{name}_k{k}"] = {"status": "no_deterministic_fill"}
                print(f"{name} k={k}: NO deterministic fill", flush=True)
        if fragment:
            (OUT / f"manifest_part_{name}.json").write_text(
                json.dumps({"sha": sha, "entries": entries}, indent=1))
        else:
            manifest["entries"].update(entries)
    if not fragment:
        # Re-read and merge at write time: concurrent runs each held a
        # stale full copy and clobbered one another (2026-08-13, twice).
        fresh = (json.loads(mf_path.read_text()) if mf_path.exists()
                 else manifest)
        fresh["entries"].update(manifest["entries"])
        mf_path.write_text(json.dumps(fresh, indent=1))
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
