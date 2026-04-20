# PepGUIDE Reproducibility Guide

This document describes how to reproduce the final `PepGUIDE` pipeline in this repository.

## 1. Environment

Use a Python environment with CUDA-enabled PyTorch and the dependencies required by this repo.

Recommended:

- Python 3.10+
- CUDA-compatible PyTorch
- `transformers`
- `biopython`
- `pandas`
- `numpy`
- `tqdm`
- `scikit-learn`
- `blobfile`
- `matplotlib`

If you use an existing environment path in this repo, update commands as needed:

```bash
/home/gguser/xiehao/envs/PepCCD/bin/python --version
```

## 2. Data Preparation

### 2.1 Build sequence pairs from PepFlow folders

```bash
python ./prepare_pepflow.py
```

This creates:

- `./dataset/Fine_Diffusion/pepflow_fine_sequence.jsonl`

### 2.2 Create train/test CSV split

```bash
python ./split_dataset.py
```

This creates:

- `./dataset/PepFlow/pepflow_train.csv`
- `./dataset/PepFlow/pepflow_test.csv`

## 3. Required Checkpoints

Before RL fine-tuning, make sure these checkpoints exist:

- Stage-3 diffusion checkpoint (EMA)
- Stage-3 uncertainty head checkpoint (optional but recommended)
- Alignment encoders:
  - `./checkpoints/Align/best_prot.pth`
  - `./checkpoints/Align/best_pep.pth`
- Tokenizer/model files under:
  - `./checkpoints/ESM`

The final script reads paths from CLI flags in `train/rl_finetune_ppo.py`.

## 4. Final Training + Evaluation (PepGUIDE)

Run the final PRIME-aligned PPO pipeline:

```bash
bash ./experiments/run_sota_candidate_prime_ppo.sh
```

This script performs:

1. PPO fine-tuning with online implicit reward modeling.
2. Sequence-level evaluation for baseline and final model.

Main outputs:

- Fine-tuned checkpoints:
  - `./checkpoints/UR_PepCCD_MoE/rl_ur_pepccd_sota_prime_ppo_diffusion.pt`
  - `./checkpoints/UR_PepCCD_MoE/rl_ur_pepccd_sota_prime_ppo_uncertainty_head.pt`
  - `./checkpoints/UR_PepCCD_MoE/rl_ur_pepccd_sota_prime_ppo_reward_model.pt`
- Evaluation directory:
  - `./evaluation/pepflow_eval_ur_pepccd_sota_prime_ppo/`


## 5. Core Method Files

The final method implementation is primarily in:

- `./train/rl_finetune_ppo.py`
- `./evaluate_pepflow.py`
- `./model/backend.py`

## 6. Reproducibility Tips

- Keep `random_state` fixed where provided.
- Use the same GPU device and checkpoint paths as the training script.
- Do not mix outputs from different runs in the same evaluation folder.
- Run sequence-level evaluation first, then structure evaluation.
