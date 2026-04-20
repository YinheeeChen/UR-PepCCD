#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/guest/cyh/workspace/PepCCD"
PYTHON_BIN="/home/gguser/xiehao/envs/PepCCD/bin/python"
LOG_DIR="$ROOT/evaluation/run_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

PID_NO_IMPLICIT="${1:-}"
PID_NO_ROUTING="${2:-}"

echo "[$(date '+%F %T')] monitor started no_implicit_pid=$PID_NO_IMPLICIT no_routing_pid=$PID_NO_ROUTING" >> "$LOG_DIR/remaining_monitor.log"

wait_for_pid() {
  local pid="$1"
  local name="$2"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date '+%F %T')] $name pid=$pid exited" >> "$LOG_DIR/remaining_monitor.log"
}

eval_model() {
  local gpu="$1"
  local tag="$2"
  local model_path="$3"
  local head_path="$4"
  local out_dir="$5"
  echo "[$(date '+%F %T')] evaluating $tag on gpu $gpu" >> "$LOG_DIR/remaining_monitor.log"
  env CUDA_VISIBLE_DEVICES="$gpu" USE_FLAX=0 TRANSFORMERS_NO_FLAX=1 TRANSFORMERS_NO_JAX=1 PEPCCD_DEVICE=cuda:0 \
    "$PYTHON_BIN" evaluate_pepflow.py \
    --max_targets 4 \
    --num_samples 4 \
    --batch_size 4 \
    --seq_len 50 \
    --seed 20260317 \
    --baseline_tag baseline_stage3 \
    --rl_tag "$tag" \
    --rl_model_path "$model_path" \
    --rl_uncertainty_head_path "$head_path" \
    --output_dir "$out_dir" > "$LOG_DIR/${tag}_eval.log" 2>&1
}

wait_for_pid "$PID_NO_IMPLICIT" "elbo_no_implicit"
wait_for_pid "$PID_NO_ROUTING" "elbo_no_routing"

eval_model 3 \
  "elbo_no_implicit" \
  "./checkpoints/UR_PepCCD_MoE/rl_elbo_no_implicit_diffusion.pt" \
  "./checkpoints/UR_PepCCD_MoE/rl_elbo_no_implicit_uncertainty_head.pt" \
  "./evaluation/pepflow_eval_elbo_no_implicit"

eval_model 2 \
  "elbo_no_routing" \
  "./checkpoints/UR_PepCCD_MoE/rl_elbo_no_routing_diffusion.pt" \
  "./checkpoints/UR_PepCCD_MoE/rl_elbo_no_routing_uncertainty_head.pt" \
  "./evaluation/pepflow_eval_elbo_no_routing"

"$PYTHON_BIN" evaluation/build_master_table.py \
  --summary_json \
  ./evaluation/pepflow_eval_reward_guided_legacy/summary.json \
  ./evaluation/pepflow_eval_elbo_full/summary.json \
  ./evaluation/pepflow_eval_elbo_no_hotspot/summary.json \
  ./evaluation/pepflow_eval_elbo_no_implicit/summary.json \
  ./evaluation/pepflow_eval_elbo_no_routing/summary.json \
  --output_md ./evaluation/master_ablation_table.md > "$LOG_DIR/master_table.log" 2>&1

echo "[$(date '+%F %T')] monitor completed" >> "$LOG_DIR/remaining_monitor.log"
