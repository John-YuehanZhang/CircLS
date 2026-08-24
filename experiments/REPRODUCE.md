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
  stim circuits with it.
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
   conda run -n tqec python experiments/coverage45_export.py   # exports + coverage
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

Static columns (volumes, qubit-rounds, compile times) are recorded
by the same runs; `static_stats.py` spot-checks three rows standalone.

## Table 3 / 8 / 9 (ablations)

```bash
python experiments/ablation.py --outdir experiments/results/best5/ablation
python experiments/ablation_ler.py            # the twelve-point LER panel
python experiments/ablation_table.py --outdir experiments/results/best5/ablation
```

Sums cover the 35 programs that compile in every configuration;
Table 9 restricts them to the consumable-patch programs.

## Table 4 / 6 (formula validation)

```bash
python experiments/formula_deviation.py \
    --outdir experiments/results/best5/formula_dev \
    --d 3 --p 0.001 0.0005 0.0002          # then --d 5 --p 0.001 0.0005
python experiments/o3ls_composition.py     # per-layer O3LS scoring
```

`deviation.jsonl` carries the measured LERs and both block-budget
scores (calibrated and published constants); `o3ls_composition.jsonl`
carries the per-layer predictions.  The tables' medians and $N$
columns re-derive from these files.  The shipped records live under
`experiments/results/best5/formula_dev_uniform/`, whose
`reproduce_tables.py` (standard library only) prints both tables
exactly as published; its README states the per-table qualifying-set
filters.

## Shape-coverage audit (distance evidence)

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

## lsqecc baseline replay (appendix comparison)

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

## DASCOT baseline (layout sweep + replay)

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

## Optimality-gap audits (appendix, Optimality of the Heuristics)

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
the teleport_4 sweep is the largest at 362880 placements).  E2
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

- Result files are append-only; analysis takes the last row per key.
- Every run records provenance (commit, package versions) and
  refuses to start from a dirty working tree.
- Provenance rows record the authors' Python 3.13 environment; the
  pip bound (`<3.13`) tracks nwqec's wheel coverage for fresh
  installs.
- Result rows record provenance (commit hash, package versions).
  The hashes refer to the authors' development history; this
  released snapshot starts fresh and does not carry it.
