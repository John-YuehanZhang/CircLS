# CircLS gallery: compiler-generated 3D spacetime structures

One folder per example, all with the same layout:

    examples/<example>/
    ├── pipeline_demo.ipynb   the pipeline walkthrough (outputs kept)
    ├── data/                 QASM + every intermediate representation
    ├── figures/              3D spacetime view, LER plot, block-graph snapshot
    ├── stim/d{3,5,7}/        compiled circuits, clean & noisy, per distance
    └── blockgraph/           TQEC export (.dae + rotatable .html) where
                              expressible, else the geometry + a gap note

File names in `data/`, `figures/`, and `stim/` carry a numeric prefix =
the notebook cell they belong to (`1_dj_6.qasm` is cell 1's circuit,
`8_dj_6_ler.png` is cell 8's figure).
In the 3D views, x/y are the tile grid and z is time: a patch's live
range is a solid column, corridor tiles held during a merge are the
bridges; the voxel count is the paper's allocated-volume metric.

## The examples

The gallery keeps the programs of the paper's Table 2 that have a walkthrough
here, compiled in the paper's configuration: optimized mapping, first-use
initialization and last-use freeing.  The compiler's optional passes
(terminal-measurement re-selection, the lifetime-aware step scheduler and
parallel merge windows) are switched off, as in the paper.

| example | circuit | number of joint PPMs | TQEC block graph |
|---|---|---|---|
| `toffoli_n3/` | Toffoli, 7 T gates (QASMBench `toffoli_n3`, the paper's Table 2 form; \|+⟩ stands in for every \|T⟩, the paper's X-state proxy) | 7 | **Yes** |

Every exported block graph is validated with tqec's own toolchain —
loaded back from the `.dae`, compiled, and sampled silent at p=0 (logs
in each `blockgraph/tqec_validation.txt`).

## Baseline toolchains

`baselines/` holds the same kind of walkthrough for a toolchain the paper
compares against, on a Table 2 program, so the two sides can be read next
to each other.  These entries do not use CircLS; each has its own `tools/`
and its own environment (see the entry's README).

| entry | toolchain | circuit | TQEC block graph |
|---|---|---|---|
| `baselines/teleportation_n3_topologiq/` | topologiq + tqec (the paper's TQEC$_2$ column) | QASMBench `teleportation_n3`; its T and S become injection gadgets on two magic wires, the form topologiq receives | **Yes** — two tqec exports: `open_port` (topologiq's graph, wire ends as ports) and `closed_port` (ports filled the way the QASM prepares and measures; 26 cubes, the Table 2 volume); the d = 3 and d = 5 circuits are byte-identical to the ones behind the table |

## Fine print

- LER: PyMatching throughout (the paper's published decoder class for
  these families), d = 3, 5, 7 at p = 5e-4 .. 2e-2.

## tools/

- `render_spacetime.py` — the 3D voxel renderer
- `export_blockgraph_auto.py` — schedule → TQEC block graph
  (`extract` / `export` / `auto_export` with seam-side variant search)
- `validate_blockgraph.py` — `.dae` round-trip + tqec compile + p=0 check
- `nb_helpers.py` — Game-of-Surface-Codes-style op display and notebook
  helpers
- `ler_sampler.py` — shard-parallel LER estimation (results independent
  of the worker count)

Beyond the examples: `../notebooks/custom_compilation_api.ipynb` is a
runnable guide to overriding each compiler decision (mapping,
orientation, lifetimes, routing, schedule, or entering with your own
PPM sequence).

## Install

Python 3.9–3.12, then `pip install -r requirements.txt` — see the
comments inside for the version constraint's reason and the two non-pip
pieces (tested end to end on 3.12).
