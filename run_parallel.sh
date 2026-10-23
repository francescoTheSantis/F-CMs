#!/usr/bin/env bash
set -euo pipefail

MAX_JOBS=2

commands=(
    #"python main.py --config-name=test_asia_int_s.yaml"
    #"python main.py --config-name=test_alarm_data_preparation.yaml"
    #"python main.py --config-name=test_asia_int.yaml"
    #"python main.py --config-name=test_insurance_datapreparation.yaml"
    #"python main.py --config-name=test_asia_int_s_2.yaml"
    #"python main.py --config-name=test_asia_int_2.yaml" 
    "python main.py --config-name=test_siim_int_seed15_c.yaml"
    "python main.py --config-name=test_siim_int_seed16_c.yaml"
    "python main.py --config-name=test_siim_int_seed15_d.yaml"
    "python main.py --config-name=test_siim_int_seed16_d.yaml"
    "python main.py --config-name=test_siim_int_seed15_s.yaml"
    "python main.py --config-name=test_siim_int_seed16_s.yaml"
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
