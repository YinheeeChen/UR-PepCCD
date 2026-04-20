#!/usr/bin/env bash
set -euo pipefail

cd /workspace/guest/cyh/workspace/PepCCD

PY=/home/gguser/xiehao/envs/structure_eval/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

for NAME in \
  pepflow_eval_reward_guided_legacy \
  pepflow_eval_elbo_full \
  pepflow_eval_elbo_no_hotspot \
  pepflow_eval_elbo_no_implicit \
  pepflow_eval_elbo_no_routing
do
  echo "[$(date '+%F %T')] rerunning ${NAME} with top3 structure candidates"
  "$PY" ./evaluation/structure_eval.py \
    --summary_json "./evaluation/${NAME}/summary.json" \
    --per_target_csv "./evaluation/${NAME}/per_target_metrics.csv" \
    --generated_csv "./evaluation/${NAME}/generated_peptides.csv" \
    --test_csv ./dataset/PepFlow/pepflow_test.csv \
    --raw_pepflow_root /workspace/guest/cyh/Dataset/PepFlow \
    --output_dir "./evaluation/${NAME}" \
    --max_peptides_per_target 3 \
    --progress_log "./evaluation/run_logs/${NAME}_structure_top3_progress.log" \
    --force_rerun
done

/home/gf/anaconda3/bin/python ./evaluation/build_master_table.py \
  --summary_json \
    ./evaluation/pepflow_eval_reward_guided_legacy/summary.json \
    ./evaluation/pepflow_eval_elbo_full/summary.json \
    ./evaluation/pepflow_eval_elbo_no_hotspot/summary.json \
    ./evaluation/pepflow_eval_elbo_no_implicit/summary.json \
    ./evaluation/pepflow_eval_elbo_no_routing/summary.json \
  --output_md ./evaluation/master_ablation_table.md

/home/gf/anaconda3/bin/python ./evaluation/build_paper_metrics_table.py \
  --summary_json \
    ./evaluation/pepflow_eval_reward_guided_legacy/summary.json \
    ./evaluation/pepflow_eval_elbo_full/summary.json \
    ./evaluation/pepflow_eval_elbo_no_hotspot/summary.json \
    ./evaluation/pepflow_eval_elbo_no_implicit/summary.json \
    ./evaluation/pepflow_eval_elbo_no_routing/summary.json \
  --output_md ./evaluation/paper_metrics_table.md
