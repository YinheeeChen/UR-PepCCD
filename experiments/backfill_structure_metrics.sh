#!/usr/bin/env bash
set -uo pipefail

ROOT="/workspace/guest/cyh/workspace/PepCCD"
PEPCCD_PYTHON="/home/gguser/xiehao/envs/PepCCD/bin/python"
STRUCTURE_PYTHON="/home/gguser/xiehao/envs/structure_eval/bin/python"
LOG_DIR="$ROOT/evaluation/run_logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"

run_backfill() {
  local eval_dir="$1"
  local name
  name="$(basename "$eval_dir")"
  echo "[$(date '+%F %T')] backfilling $name"
  local progress_log="$LOG_DIR/${name}_structure_progress.log"
  echo "[$(date '+%F %T')] progress log -> $progress_log"
  env CUDA_VISIBLE_DEVICES="${STRUCTURE_GPU:-2}" \
    "$STRUCTURE_PYTHON" "$ROOT/evaluation/structure_eval.py" \
    --summary_json "$eval_dir/summary.json" \
    --per_target_csv "$eval_dir/per_target_metrics.csv" \
    --generated_csv "$eval_dir/generated_peptides.csv" \
    --test_csv "$ROOT/dataset/PepFlow/pepflow_test.csv" \
    --raw_pepflow_root "/workspace/guest/cyh/Dataset/PepFlow" \
    --output_dir "$eval_dir" \
    --max_peptides_per_target 1 \
    --progress_log "$progress_log"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] backfill failed for $name with exit code $rc"
  else
    echo "[$(date '+%F %T')] backfill finished for $name"
  fi
  return 0
}

run_backfill "./evaluation/pepflow_eval_reward_guided_legacy"
run_backfill "./evaluation/pepflow_eval_elbo_full"
run_backfill "./evaluation/pepflow_eval_elbo_no_hotspot"
run_backfill "./evaluation/pepflow_eval_elbo_no_implicit"
run_backfill "./evaluation/pepflow_eval_elbo_no_routing"

if ! "$PEPCCD_PYTHON" evaluation/build_master_table.py \
  --summary_json \
  ./evaluation/pepflow_eval_reward_guided_legacy/summary.json \
  ./evaluation/pepflow_eval_elbo_full/summary.json \
  ./evaluation/pepflow_eval_elbo_no_hotspot/summary.json \
  ./evaluation/pepflow_eval_elbo_no_implicit/summary.json \
  ./evaluation/pepflow_eval_elbo_no_routing/summary.json \
  --output_md ./evaluation/master_ablation_table.md; then
  echo "[$(date '+%F %T')] failed to build master_ablation_table.md"
fi

if ! "$PEPCCD_PYTHON" evaluation/build_paper_metrics_table.py \
  --summary_json \
  ./evaluation/pepflow_eval_reward_guided_legacy/summary.json \
  ./evaluation/pepflow_eval_elbo_full/summary.json \
  ./evaluation/pepflow_eval_elbo_no_hotspot/summary.json \
  ./evaluation/pepflow_eval_elbo_no_implicit/summary.json \
  ./evaluation/pepflow_eval_elbo_no_routing/summary.json \
  --output_md ./evaluation/paper_metrics_table.md; then
  echo "[$(date '+%F %T')] failed to build paper_metrics_table.md"
fi

echo "[$(date '+%F %T')] structure backfill completed"
