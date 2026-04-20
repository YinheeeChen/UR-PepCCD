#!/usr/bin/env bash
set -uo pipefail

cd /workspace/guest/cyh/workspace/PepCCD || exit 1

PY=/home/gguser/xiehao/envs/structure_eval/bin/python

for NAME in pepflow_eval_elbo_no_hotspot pepflow_eval_elbo_no_routing; do
  echo "[$(date '+%F %T')] starting ${NAME}"
  "$PY" ./evaluation/structure_eval.py \
    --summary_json "./evaluation/${NAME}/summary.json" \
    --per_target_csv "./evaluation/${NAME}/per_target_metrics.csv" \
    --generated_csv "./evaluation/${NAME}/generated_peptides.csv" \
    --test_csv ./dataset/PepFlow/pepflow_test.csv \
    --raw_pepflow_root /workspace/guest/cyh/Dataset/PepFlow \
    --output_dir "./evaluation/${NAME}" \
    --max_peptides_per_target 1 \
    --progress_log "./evaluation/run_logs/${NAME}_structure_progress.log"
  RC=$?
  echo "[$(date '+%F %T')] finished ${NAME} rc=${RC}"
done
