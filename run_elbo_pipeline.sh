#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/guest/cyh/workspace/PepCCD"
PYTHON_BIN="/home/gguser/xiehao/envs/PepCCD/bin/python"
PIPELINE_LOG="$ROOT/run_logs/elbo_pipeline.log"
RL_LOG="$ROOT/run_logs/train_rl_elbo.log"
EVAL_LOG="$ROOT/evaluation/run_logs/eval_elbo_run4.log"
TABLE_LOG="$ROOT/evaluation/run_logs/table_elbo_run4.log"
OUT_DIR="$ROOT/evaluation/pepflow_eval_elbo_run4"

mkdir -p "$ROOT/run_logs" "$ROOT/evaluation/run_logs" "$OUT_DIR"

echo "[$(date '+%F %T')] ELBO-RL pipeline started" >> "$PIPELINE_LOG"

cd "$ROOT"
env USE_FLAX=0 TRANSFORMERS_NO_FLAX=1 TRANSFORMERS_NO_JAX=1 PEPCCD_DEVICE=cuda:1 \
  "$PYTHON_BIN" train/rl_finetune_elbo.py > "$RL_LOG" 2>&1
echo "[$(date '+%F %T')] RL finished with status 0" >> "$PIPELINE_LOG"

env USE_FLAX=0 TRANSFORMERS_NO_FLAX=1 TRANSFORMERS_NO_JAX=1 PEPCCD_DEVICE=cuda:1 \
  "$PYTHON_BIN" evaluate_pepflow.py \
  --max_targets 4 \
  --num_samples 4 \
  --batch_size 4 \
  --seq_len 50 \
  --output_dir "$OUT_DIR" \
  --rl_model_path ./checkpoints/UR_PepCCD_MoE/rl_elbo_diffusion.pt \
  --rl_uncertainty_head_path ./checkpoints/UR_PepCCD_MoE/rl_elbo_uncertainty_head.pt > "$EVAL_LOG" 2>&1
echo "[$(date '+%F %T')] Eval finished with status 0" >> "$PIPELINE_LOG"

"$PYTHON_BIN" evaluation/make_results_table.py \
  --summary_json "$OUT_DIR/summary.json" \
  --output_dir "$OUT_DIR" > "$TABLE_LOG" 2>&1
echo "[$(date '+%F %T')] Table finished with status 0" >> "$PIPELINE_LOG"
