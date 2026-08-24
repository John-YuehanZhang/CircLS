# The 54-program ablation campaign (tab:ablation and tab:ablation-full)

Recovered 2026-08-24 from the development session's working directory.
This is the round the paper's ablation tables come from (the sibling
`../ablation/` directory holds an earlier 35-program round whose sums do
not match the paper).  Verified on the day of archiving:
`scripts/build_ablation_table.py` run in this directory reproduces the
shipped main table exactly — 44 programs compiling in every config of 54
attempted; full row 15086 / 699k qubit-cycles / 2206 / 1146 s; LER
geometric means +180.8 / +7.9 / +12.0 / +9.1 / +10.2 / +25.0 over the
70-point panel; fails 8 / 1 / 1.

## Layout

- `compile.jsonl` — 502 case rows (54 programs x configs, last row per
  key wins); each row EMBEDS its provenance record.
- `fullmetrics/` — per-(program, config) cost metrics (volume,
  qubit-cycles, execution time, compile seconds, peak, L1).
- `points/` — the LER sample points (program x config x p in
  {5e-4, 1e-3}), the 70-point panel's source.
- `repro_shards/` — the d=11 determinism shard checks.
- `scripts/` — reference copies: the compile/metrics/sampling drivers and
  the table builders (`build_ablation_table.py` emits the main-table
  LaTeX body; `rebuild_appendix_ablation.py` the appendix variant).

Not archived: the compiled circuits (376 MB; regenerable from the same
configs, and the non-Clifford ones are in the Zenodo archive).

## Provenance status (honest)

Embedded provenance records commit `513e972...` with `dirty: true` — the
campaign ran from the development tree mid-session.  Costs are
deterministic recompiles; LER rows carry per-point seeds.
