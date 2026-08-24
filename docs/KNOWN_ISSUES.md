# Known issues

## auto_rotate x t_as_s: logical-ledger desync after repeated rotations

**Symptom.** `RuntimeError: [Error] Logical Count Mismatch! Expected: 5,
Found: (1) Logicals 3, (2) Absorbed 0, (3) ...` raised from
`tracker.process_mid_measurement` during a litinski rotation's first
diagonal SE round.

**Reproducer** (deterministic, ~90 s):

```python
from benchsuite import tclass_suite
from ablation import compile_kwargs
from circls.pipeline import compile_qasm
case = {c.name: c for c in tclass_suite()}["simon_n6"]
kw = compile_kwargs("full", case)
kw["auto_rotate"] = True
compile_qasm(case.qasm, distance=3, t_as_s=True, **kw)   # raises
```

`seca_n11` hits the same wall through the scale runner's
BentLayoutError-fallback retry.

**What is established** (instrumented session 2026-08-20):

* The ledger is healthy (`standing + absorbed == expected`, 5 + 0 == 5)
  at the ENTRY of every planned rotation, including the fatal one.
* Six earlier rotations in the same compile complete with the invariant
  intact after every one of their five stages, including stages that
  reuse retired corridor (`ppm_*`) cells.
* The fatal rotation (the program's seventh; the same patch's fifth,
  its position oscillating under alternating -x/-y directions) breaks
  in stage 1: `grow_patch` itself reuses NO foreign cells and leaves
  the count intact, and the first diagonal SE after the grow then
  swallows TWO standing logical rows without registering any absorbed
  DOF — the gauge-absorb rank bookkeeping in
  `process_mid_measurement`, not the geometry engine.
* Independent of the syndrome-retirement change: with the pre-change
  `retire_measured_patch` the same compile fails earlier and
  differently (a corridor seam dead end), so there is no
  known-good configuration to bisect against.

**Status.** Not fixed: the fix sits in `lightstim/ir/tracker.py`'s
absorb accounting, which every validated Clifford result depends on;
surgery there is out of scope for the paper sprint.  Contained
instead: `auto_rotate` stays off for t_as_s compiles except
multiply_n13 (whose 17 rotations pass all four verification checks),
and seca_n11 uses an alternate recorded configuration.  Revisit after
the deadline with the reproducer above.

## add_patch atomicity covers only the ACTIVE-collision path

The 2026-08-20 pre-pass makes an ACTIVE-coordinate collision
state-free, but later failure points inside `add_patch` /
`register_coupler` (record translation, budget sync) still leave a
partially registered patch behind, and `_register_with_blocked_repair`
would then die on "already exists" instead of running its rotation
fallback.  Pre-existing behaviour, narrowed but not closed.
