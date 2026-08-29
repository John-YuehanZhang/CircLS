"""Validate an exported block graph with tqec's own toolchain.

Run with a python that has ``tqec`` installed, from the example's
blockgraph/ folder:

    python ../../../tools/validate_blockgraph.py <name>_blockgraph.dae

Loads the .dae back, compiles it with tqec at k=1, samples the noiseless
circuit, and reports qubit/observable counts, p=0 silence, and whether
every observable is deterministic.  The committed tqec_validation.txt
files are this script's output.
"""
import sys

from tqec.computation.block_graph import BlockGraph
from tqec import compile_block_graph


def main(path):
    g = BlockGraph.from_dae_file(path)
    print(f"{path}: {g.num_cubes} cubes, {g.num_pipes} pipes "
          f"(round-trip via BlockGraph.from_dae_file)")
    circ = compile_block_graph(g, observables="auto").generate_stim_circuit(k=1)
    print(f"tqec compile (k=1, d=3): {circ.num_qubits} qubits, "
          f"{circ.num_detectors} detectors, {circ.num_observables} observables")
    det, obs = circ.compile_detector_sampler(seed=0).sample(
        256, separate_observables=True)
    silent = not det.any()
    deterministic = all(bool((obs[:, i] == obs[0, i]).all())
                        for i in range(obs.shape[1]))
    print(f"p=0 sample (256 shots, seed 0): silent={silent}, "
          f"all observables deterministic={deterministic}")
    if not (silent and deterministic):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1])
