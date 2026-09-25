# toffoli_n3: CircLS

One program of the paper's Table 2: `toffoli_n3` (QASMBench), compiled by CircLS in the paper's configuration, plus the ancilla-orientation rule added on 2026-09-25
(each |+> gadget ancilla is born facing the step that consumes it, `magic_proxy="X"`), so the volume is
below the paper's Table 2 cell (83.0 blocks at d = 3 with the orientation pinned).
Everything is regenerated when `pipeline_demo.ipynb` runs, except `data/compile_summary.json`;
the block graph and the LER points are reused unless deleted.

## Layout

    pipeline_demo.ipynb          QASM -> PBC -> PPM -> CircLS -> tqec export -> stim -> LER
    data/
      1_*.qasm                   input circuit
      2_*.txt, 3_*.txt, 4_*.txt  Pauli-based circuit, re-selected and Y-free forms, PPM sequence
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
