# tab:tclass LER sampling records (the four PM/MWPF columns)

Recovered 2026-08-24 from the development session's working directory and
archived here; every LER cell of the paper's tclass table now has its
source file. Reconciled cell-by-cell against the shipped table on the day
of archiving: all four columns of all seven regular programs match
exactly; the two special cases below also match.

## Layout

- `points/` — one JSON per (program, d, decoder) point.  Names:
  `{prog}_tclass_full_d{3,5}_p0.0005_{pm,mwpf}.json` back the table's
  four LER columns; `*_panel_*` files are the qec_en_n5 companion panel.
  `pm` files nest the PyMatching result under `"pm"` (and a correlated
  second pass under `"pm2"`, not used by the table); `mwpf` files carry
  the MWPF result at top level.  Rows record tag, p, seed, target, cap,
  shots, errors, ler.
- `simon_d5_mwpf_resample.json` — SUPERSEDES the simon_n6 d=5 MWPF value
  in `points/` (0.0295 vs 0.0293): the original point's seed was lost and
  the cell was re-sampled with a recorded seed (20260823); the shipped
  table carries the re-sample.
- `multiply/` — the three multiply_n13 cells (PM d3 / MWPF d3 / MWPF d5;
  PM d5 is the table's double-dagger, PM does not apply).  These were
  sampled on the ARCHIVED circuits (Zenodo record,
  `tclass_circuits_i7vd7n13/`), which are the normative source: a fresh
  recompile reproduces every cost metric but yields a byte-different
  circuit (see experiments/REPRODUCE.md).
- `compile_stage.jsonl` — the compile/verify stage for these runs
  (verify gates recorded per circuit).
- `scripts/` — reference copies of the session scripts that produced the
  points (samplers, the simon re-sample, the compile stage, the shard
  runner).  They reference session paths and are kept as provenance
  documentation, not as push-button reruns.

## Provenance status (honest)

Point files record seed/target/cap/shots/errors but NOT the generating
commit; the compile stage rows likewise carry no commit hash.  The
circuits they sampled are the archived tclass set (per-file sha256 in the
Zenodo record).  Decoder: PyMatching for `pm`, MWPF for `mwpf`; uniform
circuit-level depolarizing noise at p = 5e-4.

## Record correction (2026-08-24)

A sibling file `results/tclass/points_qec_panel_d11.jsonl` was deleted in
the hygiene pass.  Its rows were REAL measurements (bit-exact with the
shard sums under `points/shards/`), not placeholders as that commit's
message said; it was removed because it was a provenance-less derived
aggregate — the authoritative, seed-carrying source is the per-point and
per-shard files in this directory.
