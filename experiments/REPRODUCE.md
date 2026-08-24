# Reproducing the paper's experiments

Every number in the paper comes from the scripts in this folder; the
measurement records they produced (jsonl) ship under
`experiments/results/`.  You can re-derive every table from these
records without running anything, or re-run the measurements
yourself as below.

The compiled circuit files (`*.stim`) are not tracked directly.
The bundle `experiments/results/baseline_stim_circuits.tar.zst`
carries the TopoLS/tqec baseline circuits; from the repo root,
`tar --zstd -xf experiments/results/baseline_stim_circuits.tar.zst`
restores them, or step 1 below regenerates them.  CircLS's own
compiled circuits regenerate deterministically
(`python experiments/export_circuits.py`) and are archived, with
per-circuit sha256, in the data record
<https://doi.org/10.5281/zenodo.22075313>.

The non-Clifford (T-as-S) circuits behind the tclass table are
archived in the same record (`tclass_circuits_i7vd7n13/`, sha256
manifest included); those files are the normative source of the
published tclass numbers.  One exception to determinism:
recompiling `multiply_n13` with current code (`auto_rotate=True`)
reproduces every cost metric exactly but yields a byte-different
circuit whose sampled LER differs slightly; the archived circuit is
the published one.

## Environment

```bash
pip install -e .[test]          # the compiler itself (Python 3.10-3.12)
pip install mwpf                # decoder fallback for non-graphlike rows
```

External pieces, needed only for the baseline side of Table 2 and
the coverage study:

- **TopoLS**: clone <https://github.com/tqec/TopoLS> at commit
  `4c12ea2` and set `TOPOLS_DIR=/path/to/TopoLS` (defaults to a
  sibling checkout).
- **tqec**: a separate conda env named `tqec` with `tqec==0.2.0`
  (used via `conda run -n tqec`); TopoLS layouts are turned into
  stim circuits with it.  The export step also imports this package,
  so `pip install -e .` into that env as well.
- **QASMBench**: clone <https://github.com/pnnl/QASMBench> and set
  `QASMBENCH_ROOT` (defaults to a sibling checkout).

Sampling protocol everywhere: 100 failures or 10^6 shots per LER
point (exceptions stated below); PyMatching, with MWPF where stim
cannot decompose the error model; uniform circuit-level depolarizing
noise (`circls.tools.evaluate.inject_uniform_noise`).

## Table 2 (head-to-head comparison)

1. **Baseline circuits.**  Export the benchmarks and close the
   layouts' ports:

   ```bash
   conda run -n tqec python experiments/coverage45_export.py   # exports the 45-program QASM
   GEN_K=1 conda run -n tqec python experiments/generate_topols_circuits.py \
       c45_<name> --manual-fills --max-keep 1                  # per full-comparison row
   ```

   The four d=5 rows use the baseline's own fill enumeration
   (`--max-try`) instead of `--manual-fills`.  Fills and cube counts
   land in `experiments/results/topols_circuits/manifest.json`.

   Two programs, `ghz_state_n23` and `ghz_n127`, admit no
   deterministic port fill, so their manifest entries carry only a
   cube count (397 and 5928).  Their layouts come from TopoLS's
   per-layer MCTS search, which is seeded (`prog.py -r 0`) but
   bounded by a wall-clock budget (`-t`): the cube count is stable
   on the same machine class (AMD EPYC 9534) but not guaranteed
   across very different hardware.  The regenerated ground-truth
   `.bgraph` layouts are archived with this recipe (`zenodo_item1/`);
   the cube count derives deterministically from the `.bgraph` on
   any machine, so those files are the normative source for these
   two cells.

2. **Both-sides LER rows** (small programs):

   ```bash
   python experiments/compare_topols.py --names c45_<name> --k 1 \
       --p 0.002 0.001 0.0005 --max-fills 1 --out <file>.jsonl
   ```

3. **Large rows** (cat_n130, ghz_n78, cat_state_n22, cat_n35,
   Steane d=5) shard each point over a process pool:

   ```bash
   python experiments/giant_row.py --name c45_cat_n130 \
       --pair c45_cat_n130_k1_f0 --side topols --d 3 \
       --workers 18 --p 0.002 0.001 0.0005
   ```

4. **Rows where the baseline yields no measurable circuit** compile
   the published program on the CircLS side only:

   ```bash
   python experiments/layoutonly_ours.py --name c45_<name> \
       --pair c45_<name>_k1_f0 --d 3 --original-qasm
   ```

5. The GHZ-16 d=5 cells sample deeper: `ghz16_d5_precision.py`
   (10^4 failures, sharded) and `b1_tside_mwpf.py` (the baseline
   side under MWPF, so both sides share one decoder).

Static columns (volumes, qubit-rounds) are recorded by the same
runs; `static_stats.py` spot-checks three rows standalone.  The CircLS
compile-time column is NOT recorded by the comparison harness: it comes
from the dedicated `experiments/compile_time.py` run
(`results/best5/topols_compare/compile_time.jsonl`), and
`results/best5/compile_times_serial/` re-measures the column serially —
its README discloses that the two largest published cells (53.3 s,
792.3 s) re-measure faster (31.9 s, 589.2 s), so the published column
is conservative.

Two honest notes for this section.  (1) Output paths: pass `--out`
under `experiments/results/best5/...` explicitly — the defaults of
`compare_topols.py`, `ladder_ours.py` and `compare_tqec.py` predate the
`best5/` layout, and `dj16c_ours.py` now defaults to
`dj16c_ours_d3.jsonl` while the committed d = 3 file is
`dj16c_ours.jsonl`.  (2) The per-program coverage ledger
`results/best5/topols_compare/coverage45.jsonl` (the lay./circ. columns
and the 40/16/29 coverage claims) was recorded by a session driver that
was not kept; the committed `perprogram_table.py` consumes it, and its
counts have been re-verified against the export roster.

## Table 3 (tab:tclass, the nine non-Clifford programs)

The four LER columns (PyMatching / MWPF at d = 3 and d = 5) come from
the per-point samples archived under
`experiments/results/tclass/ler_points/` — see its README for the
cell-by-cell mapping, the simon_n6 d = 5 MWPF re-sample that supersedes
a lost-seed point, and the multiply_n13 cells sampled on the archived
normative circuits.  The cost columns (allocated volume, qubit-cycles,
execution time) for EIGHT of the nine programs are the `full`-config
d = 3 metrics in
`experiments/results/best5/ablation54/fullmetrics/<prog>__full.json`
(V1_volume_blocks / V2_qubit_rounds / T1_rounds).  multiply_n13 is the
exception: that campaign runs without auto_rotate and its fullmetrics
record is a BentLayoutError, so its cost row (2704.3 / 115195 / 542) is
not archived — recompile it with `auto_rotate=True` (the determinism
exception stated in the introduction) to reproduce those metrics
exactly.  The compile-time column is wall-clock and load-dependent; the
archived compile stage (`ler_points/compile_stage.jsonl`) records
comparable but not identical times for the eight programs, and its
multiply_n13 rows are compile errors for the same auto_rotate reason
(the published 1171 s came from the auto_rotate compile).  The original
LER runner was `experiments/tclass_ler.py` (its `points_core.jsonl`
output was never committed); the archived per-point samples in
`ler_points/` are the shipped record.

## Tables 4 / 15 / 7 (ablations)

The shipped campaign lives in
`experiments/results/best5/ablation54/` (54 programs attempted, 45
Clifford + 9 non-Clifford; 44 compile in every table configuration).
Its `scripts/build_ablation_table.py`, run inside that directory,
reproduces the main table exactly; `scripts/rebuild_appendix_ablation.py`
emits the appendix variant.  Sums cover the 44 common-compiling
programs; Table 7 restricts them to the consumable-patch programs.

To re-run the campaign from scratch, the drivers are the archived
`ablation54/scripts/compile.py` (54 programs, nine configs),
`compile_fullmetrics.py` and `sample.py` — each carries a `<workdir>`
placeholder session path at the top; point it at this repository first.
The older trio below re-runs only the CLIFFORD tier and reproduces the
earlier `results/best5/ablation/` round, NOT the shipped campaign
(`ablation.py` covers the 45 Clifford programs; `ablation_ler.py`
samples six hardcoded cases, not the 70-point panel;
`ablation_table.py` omits the reselect_only row):

```bash
python experiments/ablation.py --outdir experiments/results/best5/ablation
python experiments/ablation_ler.py
python experiments/ablation_table.py --outdir experiments/results/best5/ablation \
    --allow-mixed     # committed data spans several commits
```

The earlier `results/best5/ablation/` round's cross-config sums do not
match Table 4 — but its `full.jsonl` run is the input Table 11 renders
from, so it is shipped data, not dead history.

## Tables 5 / 9 (formula validation)

```bash
python experiments/formula_deviation.py \
    --outdir experiments/results/best5/formula_dev_uniform \
    --d 3 --p 0.001 0.0005 0.0002          # then --d 5 --p 0.001 0.0005
python experiments/o3ls_composition.py     # per-layer O3LS scoring
```

Note: `o3ls_composition.py` reads its measured LERs from a `DEV`
constant that points at `.../best5/formula_dev/deviation.jsonl`; when
re-running against fresh uniform data, point that constant at the
`formula_dev_uniform` file.

`deviation.jsonl` carries the measured LERs and both block-budget
scores (calibrated and published constants); `o3ls_composition.jsonl`
carries the per-layer predictions.  The tables' medians and $N$
columns re-derive from these files.  The shipped records live under
`experiments/results/best5/formula_dev_uniform/`, whose
`reproduce_tables.py` (standard library only) prints both tables
exactly as published; its README states the per-table qualifying-set
filters.

## Figure 9 and Table 16 (distance scaling)

Panel (a), DJ-16, and the appendix distance-scaling table (per-program
LER at d = 3..11; the per-segment R intervals quoted in the appendix
prose come from `dscaling_segments.py`) cover the six
Clifford programs where joint measurements remain
(`dscaling_segments.PROGRAMS`).  Their points live under
`experiments/results/best5/dscaling/` (`points*.jsonl`; deep rows
supersede), every file with a provenance row and per-point seeds:

```bash
python experiments/dscaling_static.py        # static arm, d = 3..7
python experiments/dscaling_d9.py            # d = 9 points, both configs
python experiments/dscaling_d11.py           # d = 11 points
python experiments/dscaling_ext911.py        # d = 9, 11 extension programs
python experiments/dscaling_deepen.py        # re-sample under-target points
python experiments/dscaling_segments.py      # per-segment R intervals
python experiments/dscaling_table.py         # appendix table body
python experiments/plot_dscaling.py          # the two-panel figure
```

Two honest notes.  (1) The six earliest dynamic-arm files
(`points.jsonl`, `points_ext.jsonl`, `points_ext2.jsonl` and their
`_deep` companions) predate the committed runners: their session
drivers were not kept.  Each file carries a provenance row, and the
committed readers reproduce the shipped table byte-for-byte from them.
(2) `dscaling_deepen.py` reads its circuits from the
`DSCALING_CIRC_DIR` environment variable (the original session export
directory no longer exists) — point it at your own exported circuits
before re-running.

Panel (b), adder_n4 under the Y-state approximation, reads
`experiments/results/tclass/points_adder_panel.jsonl`.  That file was
assembled from development-session shard runs predating the
parameterized sampler, so its rows carry no seeds or commit (its
leading note row states this; the d = 11 rows aggregate 24 x 3M-shot
shards).  Regenerate it with fresh seeds via

```bash
python experiments/tclass_dscaling.py --program adder_n4
```

which is statistically compatible rather than byte-identical (d = 3
spot-check: full 0.0251 vs 0.0238, static 0.0406 vs 0.038, both
within sampling error).  The built-in shot caps give full statistical
strength at d <= 9; the shipped d = 11 points aggregate 72M shots, so
pass ``--cap 11=80000000`` (hours of sampling) to match their
precision — the default 4M cap stops there at only ~10-15 failures.  The same entry point reproduces the
toffoli_n3 companion file; the qec_en_n5 companion was produced by its
own script (`tclass_dscaling_qec.py`, a slightly different protocol).
Both companion files carry full provenance.

## Table 10 (shape coverage) and the distance evidence

The construction-shape inventory and its distance probes:

```bash
python experiments/shape_inventory.py            # shapes per benchmark (full, d=3)
python experiments/shape_probes.py               # graphlike distance: d=5 sweep,
                                                 # corridor branch, parallel window
python experiments/seam_case_probes.py           # the two Table-1 seam cases no
                                                 # benchmark exercises, d=3 and d=5
```

Each script refuses a dirty working tree (pass `--allow-dirty` for
throwaway runs) and appends to its jsonl under
`experiments/results/best5/` with a provenance row.

The d = 3 half of the graphlike-distance claim (the 14-program sweep)
is evidenced by `results/best5/roadtest.jsonl`, written by
`experiments/roadtest.py`.  Honest note: that file predates the
provenance convention (no provenance row) and was recorded with
liveness off under both placement modes — a broader smoke
configuration than the full pipeline; the d = 5 evidence
(`shape_probes.jsonl`) uses the full pipeline.

Table 11 (per-program) renders with `experiments/perprogram_table.py`
from the ablation `full` run and `coverage45.jsonl`.  Verification
gate 4 (the re-selection rewrite preserves the measured operators) is
evidenced by `results/best5/reduction/record.jsonl`, written by
`experiments/verify_reduction.py`.

## Table 6 (tab:tclass-scale, the ten larger non-Clifford programs)

Data: `results/tclass/scale_metrics.jsonl` (the 4-hour round) and
`scale_metrics_12h.jsonl` (the 12-hour round, including the qram_n20
compile).  Honest note: the 12-hour driver was a session variant of
`experiments/tclass_scale.py` (which ships with a 4-hour timeout) and
was not kept; the archived rows carry provenance and the table values
have been re-verified against them.

## Table 13 (lsqecc baseline replay)

The [liblsqecc](https://github.com/latticesurgery-com/liblsqecc)
slicer's plan, replayed on the CircLS backend so its qubit-rounds and
LER are measurable at the circuit level (`experiments/lsqecc_bridge.py`;
the replay contract and its verification gates are the module
docstring).

Build the slicer (no sudo; conda supplies the missing headers):

```bash
git clone https://github.com/latticesurgery-com/liblsqecc && cd liblsqecc
git submodule update --init --recursive
conda install -y -c conda-forge postgresql
cmake -B build -DSQLite3_INCLUDE_DIR=$HOME/miniconda3/include \
      -DSQLite3_LIBRARY=$HOME/miniconda3/lib/libsqlite3.so \
      -DPostgreSQL_INCLUDE_DIR=$HOME/miniconda3/include \
      -DPostgreSQL_LIBRARY=$HOME/miniconda3/lib/libpq.so
PKG_CONFIG_PATH=$HOME/miniconda3/lib/pkgconfig make -C build lsqecc_slicer
```

(`make lsqecc_slicer` only — the gtest target does not build.)

Run the bridge; `--prog` loads the case straight from the benchsuite
registry, so no QASM files change hands:

```bash
python experiments/lsqecc_bridge.py \
    --slicer <path>/liblsqecc/build/lsqecc_slicer \
    --prog cat_state_n4 --workdir /tmp/lsqecc
```

A run prints its four gates (p=0 silence, observable determinism,
observable count == the plan's deterministic dimension, output-bit
affine equality against the logical reference) and exits non-zero if
any fails.

## Table 14 (DASCOT baseline: layout sweep + replay)

DASCOT ships as the `wisq` package (`pip install wisq`); it exposes no
seed, but its SA routing draws from the global python+numpy RNGs, so
seeded subprocesses are bit-reproducible:

```bash
python experiments/dascot_sweep.py --workdir /tmp/dascot   # 47 programs x 7 seeds
python experiments/dascot_bridge.py --program cat_state_n4 \
    --measure --p 5e-4                    # replay the median-step seed
```

The sweep records their map, schedule and per-CX corridor paths
(self-contained rows in `dascot_layout.jsonl`); the bridge replays the
median-step plan through the same verification gates and measurement
as the lsqecc replay, appending to `dascot_replay.jsonl`.  The paper
reports the per-program median step count over the seed set.

## LaSsynth optimality anchors

LaSsynth (Tan/Niu/Gidney, ISCA'24) is the SAT-exact lattice-surgery
synthesizer shipped inside quantumlib/Stim under
`glue/lattice_surgery` (this run: commit `79ae4f11`).  Environment
(no sudo): a python 3.11 conda env with `z3-solver==4.12.1` and
`setuptools<81`, kissat 4.0.4 built from source, and the glue
directory on `sys.path`.  Then:

```bash
python experiments/lassynth_anchor.py --probe-dir <dir with env/, kissat/, spec_builder.py>
```

For each anchor program the script proves the optimal state-prep
depth (UNSAT at k*-1, SAT at k*, `verify_stabilizers_stimzx` on the
solution) at two footprints, and records the CircLS side under both
the full config and `measure_reduction=False`.  Solutions land in
`experiments/results/best5/lassynth/*.lasre.json`; rows in
`lassynth_anchor.jsonl`.

## Table 8 (optimality-gap audits)

Three exhaustive audits measure the heuristics against their optima
on inputs small enough to enumerate.  Committed results live under
`experiments/results/best5/optgap/`; each script skips any output
file that already exists (delete or move a file to re-run it).

```bash
python experiments/optgap_e1_enum.py   # placement: every ordered placement, forced via placement=
python experiments/optgap_e2_orders.py # schedule: every legal order, injected via scheduler=
python experiments/optgap_e3_router.py # routing: exact vs greedy pool on recorded instances
```

The committed jsonl keep the provenance, baseline and summary rows;
the full per-enumeration row sets (1.3M rows, 54 MB) are archived
separately as `circls_optgap_raw_sweeps.tar.zst` in the data record
<https://doi.org/10.5281/zenodo.22075313>.  To re-run a sweep from scratch, move the
committed file aside — each script skips any output that exists.

E1 replays every ordered placement of each program's patches onto
the mapper's own slot set through the full pipeline (96 workers;
bv_8, dj_8 and teleport_4 each enumerate 9! = 362880 orderings, of
which teleport_4 evaluates 359280 after exclusions).  E2
enumerates every schedule permutation satisfying the
`verify_schedule` contract and injects each through `scheduler=`
(teleport_4 is excluded: production pins `step_scheduling=False`
for that family, so there is no order choice).  E3 monkeypatches
the two corridor solvers in `circls.core.multi_patch_coupler` to record
every routing instance the production dispatch site sees, then
solves each recorded instance with the exact group-Steiner search
and with the greedy pool on identical inputs.  Provenance rows
carry the commit the committed data was produced at.

## Notes

- Most result files are append-only, and analysis takes the last row
  per key; the dscaling, tclass-scale and optgap runners instead write
  their output file whole (each refuses to overwrite an existing one).
- Every run records provenance (commit, package versions) and
  refuses to start from a dirty working tree unless `--allow-dirty`
  is passed.  Honest exception: a few shipped files did run with
  `--allow-dirty` from the development tree and their provenance rows
  say `dirty: true` — notably `lsqecc_replay.jsonl` and
  `dascot_replay.jsonl` (Tables 13 and 14), `lassynth_anchor.jsonl`,
  `ablation/ler.jsonl`, `ler/panel2.jsonl`, and the ablation54
  campaign.  Their rows carry per-point seeds and re-run cleanly.
- Provenance rows record the authors' Python 3.13 environment; the
  pip bound (`<3.13`) tracks nwqec's wheel coverage for fresh
  installs.
- Result rows record provenance (commit hash, package versions).
  The hashes refer to the authors' development history; this
  released snapshot starts fresh and does not carry it.
