# Known issues

## assignment='optimized': the qap_faq start can still differ between BLAS kernels

**Symptom.** The same program compiles to a different layout (and circuit)
on CPUs whose OpenBLAS kernel differs in FMA support.  Observed 2026-09-12:
`simon_n6` (paper configuration, d=3) is byte-identical under the SkylakeX,
Haswell and Zen kernels but different under Prescott/Nehalem/Sandybridge
(`OPENBLAS_CORETYPE=Prescott` reproduces it on any x86 machine).

**What is established.** The spectral start is deterministic across kernels
since 2026-09-12 (`_TIE_DECIMALS` / `_EIG_GAP_MIN` in
`circls/compiler/assignment.py`: interchangeable patches are ordered by
name, degenerate eigenspaces are skipped).  The remaining dependence is in
`_start_qap_faq`, which calls `scipy.optimize.quadratic_assignment(method='faq')`:
its gradient steps are `dgemm` products, and programs with interchangeable
patches have exactly tied gradient rows that the kernel's last-bit rounding
breaks.  On the Table 2 programs the FMA kernels (SkylakeX, Haswell, Zen)
agree with each other; random PPM programs can differ even between SkylakeX
and Haswell.  A first attempt to round the FAQ iterates to 6 decimals was
rejected: rounded step sizes put gradient entries exactly on rounding
boundaries (seca_n11), so it only moves the tie.  A version of FAQ in exact
rational arithmetic (the flow weights 1/(k-1), the distances and the
barycentre are small-denominator rationals) would remove it; not implemented.

**Reproducer**: compile with `assignment="optimized"` under two kernels and
compare the placement / the circuit hash (the kernel is chosen when numpy is
imported, so one process per kernel):

```bash
for k in Haswell Prescott; do OPENBLAS_CORETYPE=$k python - <<'EOF'
import hashlib, sys
sys.path[:0] = [".", "experiments"]
from benchsuite import tclass_suite
from ablation import compile_kwargs
import circls.pipeline as P
case = {c.name: c for c in tclass_suite()}["simon_n6"]
kw = compile_kwargs("full", case); kw.update(measure_reduction=False, step_scheduling=False, parallel_steps=False)
cp = P.compile_qasm(case.qasm, distance=3, t_as_s=True, **kw)
print(sorted(cp.placement.items()), hashlib.md5(str(cp.circuit).encode()).hexdigest()[:12])
EOF
done
```

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

## first_use_init=False, liveness=False: five seam qubits without QUBIT_COORDS on qram_n20

**Symptom.** Compiling qram_n20 at d = 3 with both allocation rules off
(`first_use_init=False, liveness=False`; `assignment="optimized"`,
re-selection on or off, scheduler and parallel windows off, X proxy,
the Table 2 input file) succeeds, and the circuit passes the p = 0
checks (no detector fires, all 33 observables deterministic), but five
qubits (indices 6164, 6165, 6166, 6198, 9847 of 13 738) carry R/RX,
CX and M/MX instructions and no `QUBIT_COORDS`.  `experiment_stats`
then raises `KeyError` on the first of them, so a driver that computes
the metrics before saving the circuit records the compile as failed.

**What is established** (2026-09-17).  The five qubits' two-qubit
partners all sit on tiles (10, 9) and (11, 9): they are the seam
between those two tiles, used in merges around rounds 90-103 and
307-326.  Placing each on its partners' tile adds no tile-round, so the
allocated volume is unaffected (32 820.7 blocks); the circuit compiled
with re-selection off is byte-identical.  None of the paper-
configuration circuits (Table 2, Table 3, the other nine Static
circuits) has a coordinate-less qubit, so no reported number is
touched.  Where the registration path drops the coordinates has not
been located.

**Status.** Not fixed (decided 2026-09-17): the circuit is correct and
the metric is recoverable, so the fix is bookkeeping only.  Contained:
`fig9_v2/make_fig9.py` (campaign tooling) places a coordinate-less
qubit on its most frequent partner tile and reports how many rounds
that adds (zero here).  Reproducer: the compile above, about 6 h.
