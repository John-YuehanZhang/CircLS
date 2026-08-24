# Architecture

The package layout mirrors the paper: Section 3 (lifetime-aware
compilation — the decisions) lives in `circls/compiler/`, Section 4
(lowering PPM sequences to physical circuits — the construction) lives
in `circls/core/`.  `circls/pipeline.py` holds the entry points that
tie them together.

## The pipeline, module by module

```
QASM ──interop/nwqec──> PauliCircuit ──interop/ir──> PPM program
                                                        │
      compiler/measure_reduce   re-selection            │  Section 3:
      compiler/scheduling       PPM ordering            │  decisions
      compiler/mapping,         placement +             │
        assignment              orientation             │
                                                        ▼
      core/sequential_ppm_ls    the execution driver ── build loop
        ├─ compiler/routing         corridor search (proposes)
        ├─ compiler/rotation_plan   rotation planning        Section 3
        ├─ compiler/parallel_windows  shared merge windows   mechanisms
        ├─ compiler/liveness        lifetime windows, retire (mixins)
        └─ core/multi_patch_coupler construction + oracle    Section 4
                                                        │
      tools/evaluate            verify(), measure_ler() ▼
      tools/reporting           per-shot program bits   stim circuit
```

Two crossing points are deliberate and safe: the driver composes the
three Section-3 mixins into `SequentialPPMExperiment` (their methods
run on the driver's state), and `core/multi_patch_coupler.route_and_build`
imports `compiler/routing` lazily inside its body to obtain corridor
proposals.  Package direction is documentation, not an import-order
constraint.

## Extending CircLS

**Route one — hooks (recommended).**  Every decision is replaceable at
runtime without touching this repository; your algorithm lives in your
own code base and both it and the built-in pass run through the
identical backend, so comparisons are apples to apples:

```python
from circls import analyze_qasm, compile_qasm

info = analyze_qasm(qasm)                  # the PPM structure to optimize against
out = compile_qasm(qasm, distance=3,
                   placement=my_mapper(info.interactions),
                   scheduler=my_scheduler,  # or reselector=, router=,
                   )                        # orientation=, lifetime=
```

Contracts, validation rules, and one worked example per hook:
`docs/API_HOOKS.md`; runnable walkthrough:
`notebooks/custom_hooks.ipynb`.

**Route two — contribute a built-in pass.**  Reach for this only when
the hook contracts cannot express your idea (for example, reordering
anticommuting PPM pairs with sign tracking, which `verify_schedule`
forbids by design).  Built-in passes live in `circls/compiler/`, one
module per decision; wire a new policy into the dispatch the existing
passes use (for placement, the `assignment=` policy switch in
`interop/ir/gosc_gadgets.to_experiment_inputs`) and add tests
alongside the pass.

## What is deliberately closed

The four-case seam table, the construction rules, the verification
suite, and the internal router's search-depth knobs define what a
CircLS circuit *is* and are not extension surface — see "Out of
scope" in `docs/API_HOOKS.md`.
