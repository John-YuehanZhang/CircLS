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

| example | circuit | number of joint PPMs | TQEC block graph |
|---|---|---|---|
| `steane/` | Steane encoding, 24 gates (the paper's Table 2 form) | 2 | **Yes** |
| `bv_8/` | Bernstein–Vazirani, 20 gates | 5 | **Yes** |
| `dj_6/` | Deutsch–Jozsa, 17 gates | 6 | **Yes** |
| `bv_6/` | Bernstein–Vazirani, 15 gates | 4 | **Yes** — needs a non-default seam-side choice, found by the exporter's variant search |
| `twistedghz_8/` | GHZ with S twists, 16 gates | 4 | **No** — no deterministic correlation surface (\|Y⟩-state consumption) |
| `toffoli/` | Toffoli, 7 T gates (\|Y⟩ proxy, the paper's Table 3 form) | 7 | **No** — outside the ZXCube wall rules (seven \|Y⟩ ancillas) |

Every exported block graph is validated with tqec's own toolchain —
loaded back from the `.dae`, compiled, and sampled silent at p=0 (logs
in each `blockgraph/tqec_validation.txt`).

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
