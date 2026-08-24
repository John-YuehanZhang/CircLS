# Metric registry (every evaluation standard used in this project)

Design decision 2026-08-06: components are judged on the metric they target,
and EVERY metric we have ever used stays recorded even when a table
displays only a subset.  Recording is automatic: each ablation/LER/compare
jsonl row carries the FULL `circls.metrics.report.to_dict` stats dict —
the tables are projections, never the record.

## 1. Resource metrics (`circls/metrics/`, recorded per case in every run)

Space
- `S1_tiles_bbox` — tiles in the layout bounding box
- `S2_tiles_touched` — tiles ever occupied
- `S3_tiles_peak` — max simultaneously live tiles
- `S4_qubits_active` / `S5_qubits_allocated` — physical qubit counts

Time
- `T1_rounds` — measurement layers = wall-clock length (the LATENCY
  metric; parallel step execution is judged here)
- `T2_logical_timesteps` — rounds / d
- `T4_compile_seconds` — build time

Spacetime volume
- `V1_volume_blocks` — occupied tile-rounds / d^3 (THE headline volume
  metric; reduction/placement/fui/liveness are judged here)
- `V2_qubit_rounds` — qubit x round exposure (the A1 comparison axis
  where CircLS beats TopoLS on ghz_16 at both k)
- `V3_bbox_volume_blocks` — bounding-box volume

Liveness / utilization
- `L1_live_tile_rounds`, `L2_utilization` (= L1 / tile_rounds; aggregated
  as ratio of sums, never mean of ratios), `L4_reuse_rate`

Internal (kept under `stats.internal`)
- `ticks`, `corridor_cells`, `tile_rounds`, `qubit_rounds_span`,
  `qubits_per_tile_round`, `num_detectors`, `num_observables`

## 2. Correctness / quality gates (recorded wherever they run)

- p=0 silence — 128 detector-sampler shots, seed 0 (`silent` field)
- graphlike distance — `shortest_graphlike_error`, must read full d
- decoder single-error audit — miscorrected single-error mechanisms out
  of the DEM total.  SYSTEM-LEVEL LER ruling (user, 2026-08-09):
  published LER treats circuit + decoder as one system, so the audit is
  a per-point DIAGNOSTIC (`audit: {miscorrected, mechanisms}`), not a
  gate; earlier gate-era rows recorded fallback rungs
  (`parallel_fallback` / `schedule_fallback`) instead
- LER — adaptive shots to 100 target joint errors (max 1M), records
  shots / joint_errors / per_obs_errors / seed
- truth-oracle equality — extraction's affine structure (deterministic
  parity basis + values) equals the logical TableauSimulator's
  (tests/test_reporting_truth.py)
- parallel-vs-serial distribution equality (tests/test_parallel_steps.py)
- measurement-reduction execution equivalence — frozen 8/8 record in
  results/reduction_equivalence/

## 3. Comparison-side metrics (`compare_tqec.side_metrics`, both sides)

`qubits_active`, `qubits_allocated`, `rounds`, `qubit_rounds`,
`num_detectors`, `num_observables`; plus `topols_cubes` on their side.

## 4. Which table shows what

- A2 ablation (`ablation_table.py`): raw V1/rounds/corr/peak/L1/L2 plus
  TWO relative columns — "V1 +% when removed" and "rounds +% when
  removed" — because a latency optimization (parallel) keeps V1 flat by
  construction while a volume optimization can COST rounds (measured
  v5: fui -8.2%, liveness -19.4%, sched -7.1% rounds when present).
  Per-case V1 and per-case rounds sections list every case.
- A1 comparisons (`compare_topols.py` / `compare_tqec.py`): side metrics
  + LER points per p.
- LER panel (`ler.py`): LER vs p per distance with gate provenance.
