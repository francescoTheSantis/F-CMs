#!/usr/bin/env bash
set -euo pipefail

MAX_JOBS=2

commands=(
    "python main.py --config-name=test_asia_d_new"
    "python main.py --config-name=test_asia_s_new"
    "python main.py --config-name=test_asia_s_old"
    "python main.py --config-name=test_alarm_d_new"
    "python main.py --config-name=test_alarm_s_new"
    "python main.py --config-name=test_alarm_s_old"
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
