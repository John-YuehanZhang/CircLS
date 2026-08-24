#!/bin/bash
# End-to-end rerun of the best5 result set (five task groups in parallel).
# Logs go to $LOG_DIR (default: experiments/logs).
set -u
cd "$(dirname "$0")/.."
R=experiments/results/best5
SP=${LOG_DIR:-experiments/logs}
mkdir -p $R/ablation $R/ler $R/topols_compare $R/tqec_compare $R/formula_dev $R/reduction "$SP"
: > $SP/b5_master.log

( python experiments/roadtest.py --out $R/roadtest.jsonl > $SP/b5_roadtest.log 2>&1
  python experiments/ablation.py --outdir $R/ablation > $SP/b5_ablation.log 2>&1
  python experiments/ablation_table.py --outdir $R/ablation --tex > $SP/b5_abtable.log 2>&1
  echo "T1 done" >> $SP/b5_master.log ) &

( python experiments/ler.py --cases teleport_4 twistedghz_4 bbpssw_4 dj_8 bv_8 ghz_8 \
    --d 3 5 --p 2e-3 1e-3 5e-4 2e-4 1e-4 --target-errors 100 --max-shots 1000000 \
    --workers 24 --out $R/ler/panel2.jsonl > $SP/b5_ler.log 2>&1
  python experiments/verify_reduction.py --out $R/reduction/record.jsonl > $SP/b5_reduction.log 2>&1
  echo "T2 done" >> $SP/b5_master.log ) &

( python experiments/compare_tqec.py --k 1 2 --p 2e-3 1e-3 5e-4 \
    --out $R/tqec_compare/main.jsonl > $SP/b5_tqec.log 2>&1
  python experiments/compare_topols.py --names ghz_16 bv_16 dj_16 CNOT \
    --out $R/topols_compare/main.jsonl > $SP/b5_topols.log 2>&1
  echo "T3 done" >> $SP/b5_master.log ) &

( for spec in "bv_32 3" "bv_32 5" "bv_64 3" "bv_64 5" "dj_32 3" "dj_32 5" "dj_64 3" "dj_64 5"; do
    set -- $spec
    python experiments/ladder_ours.py --names $1 --d $2 \
      --out "$R/topols_compare/ladder_ours_${1}_d${2}.jsonl" \
      > "$SP/b5_ladder_${1}_d${2}.log" 2>&1 &
  done
  for spec in "bv_100 3" "bv_100 5" "dj_100 3" "dj_100 5"; do
    set -- $spec
    python experiments/ladder_ours.py --names $1 --d $2 --skip-ler \
      --out "$R/topols_compare/ladder_ours_${1}_d${2}.jsonl" \
      > "$SP/b5_ladder_${1}_d${2}.log" 2>&1 &
  done
  wait
  echo "T4 done" >> $SP/b5_master.log ) &

( python experiments/formula_deviation.py --outdir $R/formula_dev --workers 32 --timeout 7200 > $SP/b5_formula.log 2>&1
  python experiments/formula_deviation.py --outdir $R/formula_dev --d 3 --p 2e-4 --workers 16 --timeout 7200 --allow-dirty > $SP/b5_formula_p2e4.log 2>&1
  python experiments/formula_deviation_table.py --outdir $R/formula_dev > $SP/b5_ftable.log 2>&1
  echo "T5 done" >> $SP/b5_master.log ) &

wait
echo "ALL DONE" >> $SP/b5_master.log

# Reproducibility: every jsonl in experiments/results/best5/ starts with a
# provenance row (circls sha, versions, argv).  Tables regenerate via
# ablation_table.py / formula_deviation_table.py with --outdir pointed at
# the corresponding subdirectory; no number is ever hand-copied.
