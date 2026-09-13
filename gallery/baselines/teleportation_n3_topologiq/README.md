# teleportation_n3: topologiq + tqec

One program of the paper's TQEC$_2$ baseline (topologiq + tqec): `teleportation_n3`.
Everything is regenerated when `pipeline_demo.ipynb` runs.

## Layout

    pipeline_demo.ipynb          QASM -> topologiq -> tqec -> LER
    tools/                       code the notebook calls
    data/
      1_*.qasm, 1_*.json         input circuit, gadget form, gadget map
      2_*.bgraph, 2_*.txt/json   topologiq block graph, log, run stats
      3_*.json                   shim, equivalence check, port fill
      4_*.json                   compile reports, LER points (cached)
    blockgraph/
      *_open_port.{dae,html}     tqec block graph, ports open
      *_closed_port.{dae,html}   tqec block graph, ports filled
      tqec_validation.txt        tqec reload, compile, p = 0 check
    stim_circuit/d{3,5,7}/       compiled circuits, clean and noisy

## Requirements

    pip install git+https://github.com/tqec/topologiq@5d74fe7
    pip install tqec==0.2.0

Versions used: topologiq `5d74fe7`, tqec 0.2.0 (git `5909dde`), stim 1.16.0, PyMatching 2.4.0.
CircLS is not required.

## Usage

    jupyter nbconvert --to notebook --execute --inplace pipeline_demo.ipynb

Optional environment variables:

    CHROME=<browser binary>      still image of the block graph
    TQEC_WORKERS=<n>             tqec process pool size (default 8)

Delete `data/4_*_ler_points.json` to resample the LER.
