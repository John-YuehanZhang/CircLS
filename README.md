# CircLS — Circuit-level Lattice Surgery

CircLS is a lattice-surgery compiler built on
[Stim](https://github.com/quantumlib/Stim) and
[LightStim](https://github.com/QuTone/LightStim). It takes a Clifford
QASM program or a **PPM (Pauli-product measurement) sequence** all the
way down to a **runnable stim circuit** — detectors and observables
annotated, verified at the circuit level — so a compiled program's
logical error rate is something you measure, not estimate.

This repository is the artifact of the paper [*CircLS: Compiling
Lattice Surgery to Physical Circuits with Dynamic Allocation*
(arXiv:2608.23819)](https://arxiv.org/abs/2608.23819); the
runs behind every number in the paper are reproducible from
`experiments/REPRODUCE.md`.

## What this repo is for

- **Complete the pipeline**: compile Clifford QASM or a **PPM
  (Pauli-product measurement) sequence** all the way to a **runnable
  stim circuit**, detectors and observables annotated, through
  linear-time stabilizer construction rules.
- **Shared infrastructure for any LS compiler**: other compilers can
  lower their PPM sequences the same way, verify the compiled
  program at the circuit level, and measure the LER it actually
  delivers (PyMatching / MWPF).
- **Verify every compiled circuit**: no detector triggers at p = 0,
  the shortest graphlike logical error stays at weight d, and the
  program bits match a logical-level simulation.
- **A lifetime-aware default**: the built-in strategy is the paper's
  pipeline: a data patch is initialized at its first use, freed at
  its last use, and its tile is reused by corridors; re-selection,
  reordering, parallel execution and mapping shorten lifetimes
  further.
- **Every decision is replaceable**: result injection or strategy
  hooks for placement, routing, ordering, re-selection and lifetimes
  (`docs/API_HOOKS.md`).

## Repository layout

```text
CircLS/
├── circls/
│   ├── pipeline.py             # Entry points: compile_qasm, compile_ppm_sequence, analyze_qasm
│   ├── core/                   # Paper Section 4: lowering PPM sequences to circuits
│   │   ├── sequential_ppm_ls.py    # The execution driver
│   │   ├── multi_patch_coupler.py  # Rule-based stabilizer construction + oracle
│   │   ├── joint_merge.py          # The seam rule table
│   │   └── routed_multi_patch_ls.py, colouring.py, common.py
│   ├── compiler/               # Paper Section 3: compilation decisions (add yours here)
│   │   ├── measure_reduce.py       # Terminal measurement re-selection
│   │   ├── mapping.py, assignment.py  # Placement
│   │   ├── scheduling.py           # Lifetime-aware PPM ordering
│   │   ├── routing.py              # Corridor search (exact EMV + greedy pool)
│   │   └── rotation_plan.py, parallel_windows.py, liveness.py  # Driver mixins
│   ├── interop/
│   │   ├── ir/                 # Front-end-agnostic IR + PBC->PPM chain
│   │   └── nwqec/              # The NWQEC front-end adapter
│   ├── tools/                  # verify(), measure_ler(), per-shot reporting
│   └── metrics/                # Volume/qubit-round/utilization accounting
├── lightstim/                  # Vendored, modified LightStim (see VENDORED.md)
│   ├── tests/                  # Its test suite
│   └── benchmarks/             # Its benchmark scripts
├── experiments/                # Paper harnesses and the benchmark suite
├── tests/                      # Compiler test suite
├── notebooks/                  # Runnable API guide
├── gallery/                    # Compiler-generated 3D spacetime structures
├── docs/
│   ├── ARCHITECTURE.md         # Paper-section <-> package map; how to extend
│   └── API_HOOKS.md            # Customization hooks + PPM-sequence entry
├── pyproject.toml
├── LICENSE                     # Apache-2.0
└── VENDORED.md                 # How the lightstim copy relates to upstream
```

## Quick start

Python 3.10–3.12 (the NWQEC front-end ships wheels up to
cp312; 3.13 is rejected at install time).

```bash
pip install circls
```

To reproduce the paper's experiments or to develop, clone the
repository instead — the `experiments/` harnesses and archived
measurement records ship with the repo, not the package:

```bash
git clone https://github.com/John-YuehanZhang/CircLS.git
cd CircLS

python3.12 -m venv venv
source venv/bin/activate

pip install -e .
```

```python
from circls import compile_qasm, verify, measure_ler

qasm = """OPENQASM 2.0; include "qelib1.inc";
qreg q[4]; creg c[4];
h q[0]; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3];
measure q -> c;"""

out = compile_qasm(qasm, distance=3)     # -> runnable stim circuit
report = verify(out)                     # circuit-level checks
stats = measure_ler(out, p=1e-3)         # real logical error rate
print(report.ok, stats.logical_error_rate)
```

`verify` runs the suite's checks (p = 0 silence, observable
determinism, graphlike distance, logical-simulation match);
`measure_ler` injects the paper's uniform circuit-level noise (the
same model as `experiments/noise_inject.py`) and
decodes with PyMatching (MWPF fallback).  The compiler's decisions
are readable off the result: `out.placement`, `out.routes`,
`out.lifetimes`, `out.stats()`.

## Usage examples

### Enter with a PPM sequence (any front-end)

```python
from circls.interop.ir.gosc_gadgets import OutBit, PPMProgram, ProgramOp
from circls.pipeline import compile_ppm_sequence

prog = PPMProgram(
    num_data=3,
    ops=[ProgramOp("mpp", {0: "X", 1: "X"}),
         ProgramOp("mpp", {1: "X", 2: "X"}),
         ProgramOp("mpp", {0: "Z"}),
         ProgramOp("mpp", {1: "Z"}),
         ProgramOp("mpp", {2: "Z"})],
    gadgets=[],
    out_bits=[OutBit(rec=k, flip=0) for k in range(5)],
)
out = compile_ppm_sequence(prog, distance=3)
```

### Bring your own decisions

```python
# your placement, our construction and verification:
out = compile_qasm(qasm, distance=3,
                   placement={"q0": (0, 0), "q1": (0, 1),
                              "q2": (1, 0), "q3": (1, 1)})
```

Every stage is switchable or replaceable; injected decisions pass the
same legality checks and verification as the built-in ones.

| Stage | Shorthand flag | Hook | Default |
|---|---|---|---|
| Measurement re-selection | `measure_reduction=` | `reselector=` | minimum-weight re-selection |
| PPM ordering | `step_scheduling=` | `scheduler=` | lifetime-aware ordering |
| Patch placement | `assignment=` | `placement=` | row-major (the paper runs `"optimized"`) |
| Patch orientation | — | `orientation=` | derived from the first-use letter |
| Corridor routing | — | `router=` | corridor search |
| Patch lifetimes | `liveness=`, `keep_patches=` | `lifetime=` | first-use init / last-use free |

See `docs/API_HOOKS.md` for the contracts, validation rules, and a
worked example per hook.

## Scope and roadmap

CircLS is **Clifford-only** by design: non-Clifford input is rejected
loudly at the front-end.  Patches share one code distance and live on
the Square Sparse floor.  Custom floor geometries (grids with
defects) and per-patch distances are natural next steps on top of the
`placement=`/`router=` hooks and are not supported yet.

## Documentation

- `docs/ARCHITECTURE.md` — the paper-section ↔ package map and the
  two extension routes
- `docs/API_HOOKS.md` — customization hooks, `compile_ppm_sequence`,
  the `PPMProgram` format
- `experiments/REPRODUCE.md` — step-by-step reproduction of the
  paper's tables; `experiments/METRICS.md` — metric definitions
- `VENDORED.md` — what changed in the vendored LightStim and how it
  relates to [upstream](https://github.com/QuTone/LightStim)
- `notebooks/` — runnable API guide (`custom_compilation_api.ipynb`)
- `gallery/` — six worked examples with TQEC block-graph exports (see `gallery/README.md`)
- `experiments/` — the paper's measurement harnesses; they expect the
  baseline checkouts described in their headers (`TOPOLS_DIR` etc.)
  and are not needed to use the compiler

## Testing

```bash
pip install -e '.[test]'
python -m pytest tests -q               # compiler suite
python -m pytest lightstim/tests -q     # vendored LightStim suite
```

## Citing CircLS

See `CITATION.cff`, or use the BibTeX entry below.  If you build on
the vendored backend, please also cite
[LightStim](https://github.com/QuTone/LightStim).

```bibtex
@article{zhang2026circls,
  title={CircLS: Compiling Lattice Surgery to Physical Circuits with Dynamic Allocation},
  author={Zhang, John Yuehan},
  journal={arXiv preprint arXiv:2608.23819},
  year={2026}
}
```

## License

Apache-2.0 (`LICENSE`).  The vendored LightStim keeps its own
Apache-2.0 license verbatim.  Two files port code from Craig
Gidney's *Inplace Access to the Surface Code Y Basis* artifact
(CC-BY-4.0) and one derives from gongaa/SlidingWindowDecoder
(MIT); see the third-party notices in `VENDORED.md`.
