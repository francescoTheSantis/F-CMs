#!/usr/bin/env bash
set -euo pipefail

MAX_JOBS=1

commands=(
    #"python main.py --config-name=test_asia_int_s.yaml"
    #"python main.py --config-name=test_alarm_data_preparation.yaml"
    #"python main.py --config-name=test_asia_int.yaml"
    #"python main.py --config-name=test_insurance_datapreparation.yaml"
    #"python main.py --config-name=test_asia_int_s_2.yaml"
    #"python main.py --config-name=test_asia_int_2.yaml" 
    #"python main.py --config-name=rebuttal_r1_ablconcepts_asia_d_30.yaml"
    #"python main.py --config-name=rebuttal_r1_ablconcepts_asia_s_30.yaml"
    #"python main.py --config-name=rebuttal_r1_ablconcepts_asia_d_60.yaml"
    #"python main.py --config-name=rebuttal_r1_ablconcepts_asia_s_60.yaml"
    #"python main.py --config-name=rebuttal_r1_alarm_ablconcepts_drift_1.yaml"
    #"python main.py --config-name=rebuttal_r1_alarm_ablconcepts_drift_2.yaml"
    #"python main.py --config-name=rebuttal_r1_alarm_ablconcepts_drift_3.yaml"
    #"python main.py --config-name=test_cheXpert_preparedata.yaml"
    #"python main.py --config-name=test_asia_r1_2steps.yaml"
    #"python main.py --config-name=test_asia_r1_2steps_2.yaml"
    #"python main.py --config-name=test_asia_r1_2steps_0.yaml"
    #"python main.py --config-name=test_alarm_r1_2steps_0.yaml"
    #"python main.py --config-name=test_alarm_r1_2steps_1.yaml"
    #"python main.py --config-name=test_alarm_r1_2steps_2.yaml"
    "python main.py --config-name=test_chexpert_r1_2steps_0.yaml"
    "python main.py --config-name=test_chexpert_r1_2steps_1.yaml"
    "python main.py --config-name=test_chexpert_r1_2steps_2.yaml"

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
