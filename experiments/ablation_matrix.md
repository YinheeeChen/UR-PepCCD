# Ablation Matrix

## Core comparison set

| Model Tag | Training Style | Implicit Reward | Charge Prior | Hotspot Prior | Uncertainty Routing | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline_stage3` | Stage 3 SFT only | No | No | No | Yes | Main baseline |
| `reward_guided_legacy` | Top-k reward-guided RL | No | Yes | Yes | Yes | Historical RL version |
| `elbo_full` | Group-relative ELBO-RL | Yes | Yes | Yes | Yes | Full UR-PepCCD |
| `elbo_no_hotspot` | Group-relative ELBO-RL | Yes | Yes | No | Yes | Test whether hotspot harms binding |
| `elbo_no_implicit` | Group-relative outcome-only RL | No | Yes | Yes | Yes | Test whether ELBO reward helps |
| `elbo_no_routing` | Group-relative ELBO-RL | Yes | Yes | Yes | No | Test whether uncertainty routing helps |

## Evaluation protocol

- Dataset: `dataset/PepFlow/pepflow_test.csv`
- Targets: first 4 targets for quick-turn ablation
- Samples per target: 4
- Seed: `20260317`
- Metrics: binding, diversity, charge, hotspot, validity, uniqueness, novelty, seconds per target
