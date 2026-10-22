#!/usr/bin/env bash
set -euo pipefail

MAX_JOBS=1
PYTHON="/home/admin/miniconda3/envs/federated_c2bm/bin/python"

commands=(
    "$PYTHON main.py --config-name=test_chexpert_consensus.yaml"
)

running=0

for cmd in "${commands[@]}"; do
  echo -e "\033[34m[START] $cmd\033[0m"
  bash -lc "$cmd" &  # run in background
  sleep 5

  running=$((running + 1))

  # If we reached max parallel jobs, wait for one to finish
  if (( running >= MAX_JOBS )); then
    wait -n
    running=$((running - 1))
  fi
done

# Wait for remaining jobs
wait
echo "✅ All jobs finished."
