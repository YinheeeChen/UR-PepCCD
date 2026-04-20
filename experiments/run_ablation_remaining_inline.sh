#!/usr/bin/env bash
set -euo pipefail
ROOT="/workspace/guest/cyh/workspace/PepCCD"
PYTHON_BIN="/home/gguser/xiehao/envs/PepCCD/bin/python"
LOG_DIR="$ROOT/evaluation/run_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

eval_model () {
  local tag="$1"
  local model_path="$2"
  local head_path="$3"
  local out_dir="$4"
  echo "[$(date '+%F %T')] evaluating $tag" | tee -a "$LOG_DIR/ablation_remaining_chain.log"
  env USE_FLAX=0 TRANSFORMERS_NO_FLAX=1 TRANSFORMERS_NO_JAX=1 PEPCCD_DEVICE=cuda:1 \
    "$PYTHON_BIN" evaluate_pepflow.py \
    --max_targets 4 --num_samples 4 --batch_size 4 --seq_len 50 --seed 20260317 \
    --baseline_tag baseline_stage3 --rl_tag "$tag" \
    --rl_model_path "$model_path" --rl_uncertainty_head_path "$head_path" \
    --output_dir "$out_dir" > "$LOG_DIR/${tag}_eval.log" 2>&1
}

train_eval () {
  local tag="$1"
  local model_out="$2"
  local head_out="$3"
  local out_dir="$4"
  shift 4
  echo "[$(date '+%F %T')] training $tag" | tee -a "$LOG_DIR/ablation_remaining_chain.log"
  env USE_FLAX=0 TRANSFORMERS_NO_FLAX=1 TRANSFORMERS_NO_JAX=1 PEPCCD_DEVICE=cuda:1 \
    "$PYTHON_BIN" train/rl_finetune_elbo.py \
    --save_model_path "$model_out" --save_uncertainty_head_path "$head_out" \
    "$@" > "$LOG_DIR/${tag}_train.log" 2>&1
  eval_model "$tag" "$model_out" "$head_out" "$out_dir"
}

eval_model "elbo_no_hotspot" "./checkpoints/UR_PepCCD_MoE/rl_elbo_no_hotspot_diffusion.pt" "./checkpoints/UR_PepCCD_MoE/rl_elbo_no_hotspot_uncertainty_head.pt" "./evaluation/pepflow_eval_elbo_no_hotspot"
train_eval "elbo_no_implicit" "./checkpoints/UR_PepCCD_MoE/rl_elbo_no_implicit_diffusion.pt" "./checkpoints/UR_PepCCD_MoE/rl_elbo_no_implicit_uncertainty_head.pt" "./evaluation/pepflow_eval_elbo_no_implicit" --implicit_reward_weight 0.0 --outcome_reward_weight 1.0
train_eval "elbo_no_routing" "./checkpoints/UR_PepCCD_MoE/rl_elbo_no_routing_diffusion.pt" "./checkpoints/UR_PepCCD_MoE/rl_elbo_no_routing_uncertainty_head.pt" "./evaluation/pepflow_eval_elbo_no_routing" --u_min 0.0 --u_max 0.0
"$PYTHON_BIN" evaluation/build_master_table.py \
  --summary_json \
  ./evaluation/pepflow_eval_reward_guided_legacy/summary.json \
  ./evaluation/pepflow_eval_elbo_full/summary.json \
  ./evaluation/pepflow_eval_elbo_no_hotspot/summary.json \
  ./evaluation/pepflow_eval_elbo_no_implicit/summary.json \
  ./evaluation/pepflow_eval_elbo_no_routing/summary.json \
  --output_md ./evaluation/master_ablation_table.md > "$LOG_DIR/master_table.log" 2>&1

echo "[$(date '+%F %T')] remaining ablation suite completed" | tee -a "$LOG_DIR/ablation_remaining_chain.log"
