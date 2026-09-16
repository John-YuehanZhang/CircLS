# teleportation_n3: CircLS

One program of the paper's Table 2: `teleportation_n3` (QASMBench), compiled by CircLS in the paper's configuration.
Everything is regenerated when `pipeline_demo.ipynb` runs, except `data/compile_summary.json`;
the block graph and the LER points are reused unless deleted.
The same program through the other toolchain of Table 2 is in `../../baselines/teleportation_n3_topologiq/`.

## Layout

    pipeline_demo.ipynb          QASM -> PBC -> PPM -> CircLS -> tqec export -> stim -> LER
    data/
      1_*.qasm                   input circuit
      2_*.txt, 3_*.txt, 4_*.txt  Pauli-based circuit, Y-free form, PPM sequence
      5_*.json                   placement, routing, schedule
      8_*.json                   LER points (cached)
      compile_summary.json       front-end stage counts and per-distance circuit metrics (hand-maintained)
    figures/                     spacetime view, block graph, LER plot
    blockgraph/                  tqec block graph (.dae, .html), structure, validation
    stim/d{3,5,7}/               compiled circuits, clean and noisy

## Requirements

Python 3.10-3.12.

    pip install -e ../../..              # CircLS (repository root)
    pip install -r ../../requirements.txt
    pip install tqec==0.2.0              # block-graph export and validation only

## Usage

    jupyter nbconvert --to notebook --execute --inplace pipeline_demo.ipynb

Optional environment variable:

    TQEC_PYTHON=<python with tqec>       interpreter for the block-graph export

Delete `data/8_*_ler_points.json` to resample the LER.
