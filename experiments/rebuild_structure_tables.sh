#!/usr/bin/env bash
set -uo pipefail

cd /workspace/guest/cyh/workspace/PepCCD || exit 1

while true; do
  sleep 60
  if ! pgrep -f "structure_eval.py.*pepflow_eval_elbo_full|structure_eval.py.*pepflow_eval_elbo_no_implicit|structure_eval.py.*pepflow_eval_elbo_no_hotspot|structure_eval.py.*pepflow_eval_elbo_no_routing" >/dev/null 2>&1; then
    /home/gf/anaconda3/bin/python ./evaluation/build_master_table.py > ./evaluation/master_ablation_table.md
    /home/gf/anaconda3/bin/python ./evaluation/build_paper_metrics_table.py > ./evaluation/paper_metrics_table.md
    echo "[$(date '+%F %T')] rebuilt tables"
    break
  fi
done
