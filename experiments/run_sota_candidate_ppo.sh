#!/usr/bin/env bash
set -euo pipefail

cd /workspace/guest/cyh/workspace/PepCCD

TRAIN_PY=/home/gguser/xiehao/envs/PepCCD/bin/python
STRUCT_PY=/home/gguser/xiehao/envs/structure_eval/bin/python

FINAL_TAG=ur_pepccd_sota_ppo
CKPT_DIR=./checkpoints/UR_PepCCD_MoE
EVAL_DIR=./evaluation/pepflow_eval_${FINAL_TAG}

PEPCCD_DEVICE=${PEPCCD_DEVICE:-cuda:2} \
"$TRAIN_PY" ./train/rl_finetune_ppo.py \
  --save_model_path "${CKPT_DIR}/rl_${FINAL_TAG}_diffusion.pt" \
  --save_uncertainty_head_path "${CKPT_DIR}/rl_${FINAL_TAG}_uncertainty_head.pt" \
  --charge_weight 0.15 \
  --hotspot_weight 0.0 \
  --bind_floor 0.16 \
  --bind_penalty_weight 0.90 \
  --u_min 0.05 \
  --u_max 0.18 \
  --outcome_reward_weight 0.90 \
  --implicit_reward_weight 0.10 \
  --epochs 4 \
  --batch_size 12 \
  --robust_lr 5e-6 \
  --uncertainty_head_lr 5e-7 \
  --u_score_target 0.18 \
  --u_score_reg_weight 0.03 \
  --ppo_clip_eps 0.20 \
  --ppo_updates 3

PEPCCD_DEVICE=${PEPCCD_DEVICE:-cuda:2} \
"$TRAIN_PY" ./evaluate_pepflow.py \
  --output_dir "${EVAL_DIR}" \
  --baseline_tag baseline_stage3 \
  --rl_tag "${FINAL_TAG}" \
  --rl_model_path "${CKPT_DIR}/rl_${FINAL_TAG}_diffusion.pt" \
  --rl_uncertainty_head_path "${CKPT_DIR}/rl_${FINAL_TAG}_uncertainty_head.pt" \
  --batch_size 8 \
  --num_samples 4 \
  --candidate_batches 4 \
  --rerank_top_k 4 \
  --rerank_bind_weight 0.90 \
  --rerank_charge_weight 0.10 \
  --enable_structure_metrics True \
  --structure_eval_python "${STRUCT_PY}"
