# CircLS --- LER-formula validation data

Measurement data and a self-contained reproducer for the two
LER-formula-validation tables of the CircLS paper:

- **Table `tab:formula`** --- the O3LS composition formula (per-layer
  product-sum), main text.
- **Table `tab:blockbudget`** --- the Beverland/Litinski block budget
  (V * eps), appendix.

Both tables report, per code distance `d` and physical error rate `p`, the
**median deviation** of a published LER formula's prediction from the LER we
measure on the actually-compiled circuit. The point of the tables: the
formulas, though widely used for resource estimation, deviate substantially
from the measured circuit-level LER.

## Contents

```
README.md                     this file
reproduce_tables.py           standard-library-only; prints both tables
data/
  deviation.jsonl             measured program LER + block-budget inputs
                              (volume V1, per-block calibration eps_block)
  o3ls_composition.jsonl      O3LS per-layer prediction per benchmark
generation_scripts/           reference copies (require the CircLS package)
  formula_deviation.py        produced deviation.jsonl
  o3ls_composition.py         produced o3ls_composition.jsonl
```

Both `.jsonl` files are append-only logs. The last row per
`(name, d, p)` wins, and the first line of each is a `provenance` record.

## Reproduce the tables

No dependencies beyond the Python standard library (Python 3.8+):

```
python reproduce_tables.py
```

Expected output:

```
Table: composition formula (O3LS)  [tab:formula]
  d      p     N   deviation
  3   2e-04   22   66%
  3   5e-04   22   53%
  3   1e-03   22   46%
  5   5e-04   18   49%
  5   1e-03    9   58%

Table: block budget  [tab:blockbudget]
  d      p     N   deviation
  3   2e-04   27   31%
  3   5e-04   27   22%
  3   1e-03   27   13%
  5   5e-04   23   21%
  5   1e-03   14   33%
```

## Method

Per benchmark the deviation is `|p_pred - p_meas| / p_meas`; each table cell
is the **median** of that over the qualifying benchmarks, in percent.

Qualifying set:

- **O3LS (`tab:formula`)**: compiled OK, `p_meas > 0`, and `p_pred > 0`.
  The last condition drops the benchmarks whose Clifford gates conjugate
  every terminal measurement to a single-qubit Pauli, leaving no joint
  measurement, so the per-layer formula predicts exactly zero (these are the
  excluded cases noted in the appendix).
- **Block budget (`tab:blockbudget`)**: compiled OK and `p_meas > 0`. The
  block budget `min(1, V * eps)` is always positive, so there is no
  prediction filter. `eps` is calibrated in our own pipeline from the
  single-patch memory slope (the `calibration` records in
  `deviation.jsonl`); using the published constants `a = 0.03`,
  `p* = 0.01` instead gives larger deviations.

## Provenance

- CircLS commit: `cc034399b35d6fed25370213acf3bc74c14d2b57`
- Python 3.13.13, stim 1.16.0, numpy 2.2.6, pymatching 2.4.0, networkx 3.6.1
- CPU: AMD EPYC 9534 64-Core
- Generated 2026-08-23 (UTC)
- Noise: uniform circuit-level depolarizing at rate `p` on every location,
  idle locations included.
- `formula_deviation.py --d 3 5 --p 2e-4 5e-4 1e-3 --target-errors 100
  --timeout 3600`
- `o3ls_composition.py` with `component_errors=200`, `seed=211`

## Known limitations

- The grid is `d in {3,5}`, `p in {2,5,10} x 10^-4`. The O3LS run covered
  five of these six cells; `d=5, p=2e-4` was not run for O3LS and is absent
  from `tab:formula` (the paper's table has the same five cells).
- Under uniform noise the larger `d=5` programs are expensive; several time
  out at `p=1e-3`, so that cell has a smaller sample (`N=9` for O3LS,
  `N=14` for the block budget).
- `deviation.jsonl` may contain a few rows with `measured = 0` for circuits
  that have no logical observable; they carry `joint_errors = 0` and are
  excluded by the `p_meas > 0` filter, so they do not affect any table cell.

## Full regeneration (optional)

`reproduce_tables.py` recomputes the tables from the archived measurements
above and needs nothing else. Regenerating the measurements themselves
(recompiling every benchmark and sampling its LER) requires the CircLS
package at the commit above and its benchmark suite; the two scripts under
`generation_scripts/` are the entry points.
