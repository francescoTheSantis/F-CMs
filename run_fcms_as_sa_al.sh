#!/bin/bash
#SBATCH --job-name=fcms_as_sa_al  # Job name
# # SBATCH --gres=gpu:2g:1             # Request 1 MIG slice (24 GB VRAM)
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=6           # CPUs for data loading
                                    # RAM is auto-assigned from GPU VRAM
#SBATCH --time=92:00:00             # Max runtime (HH:MM:SS)
#SBATCH --output=logs/%j_%x.out     # Output: logs/<jobid>_<jobname>.out
#SBATCH --error=logs/%j_%x.err      # Errors: logs/<jobid>_<jobname>.err
#SBATCH --container-image=mamba
# #SBATCH --p blas
#
# PyTorch single-GPU training template
# Available containers: pip, mamba, uv, pixi
# Change --container-image above to use a different variant
# Or use --container-image=nvcr.io/... for any Docker/NGC image
# Usage: sbatch pytorch_single_gpu.sh
#
# Make sure to create the logs/ directory first:
#   mkdir -p logs

echo "=== Job Info ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "================"

# Activate your project's virtual environment (created inside the container — see user guide)
cd ~/Federated-C2BM
source .venv/bin/activate
export PATH=~/graphviz-env/bin:$PATH

# Install extra dependencies (uncomment if you have a requirements.txt)
# pip install -r requirements.txt

# Set common PyTorch environment variables
export PYTHONUNBUFFERED=1

# Run training
# python main.py --config-name=test_asia &
# sleep 10  
# python main.py --config-name=test_asia_s &

# wait  # Wait for all background processes to finish

echo "=== Start Sachs (all) ==="
python main.py --config-name=test_sachs &
sleep 30
echo "=== Start Sachs (static) ==="
python main.py --config-name=test_sachs_s &

wait  # Wait for all background processes to finish

# echo "=== Start Alarm (all) ==="
# python main.py --config-name=test_alarm &
# sleep 20
# echo "=== Start Alarm (static) ==="
# python main.py --config-name=test_alarm_s &

wait  # Wait for all background processes to finish

echo "=== Job Finished ==="
echo "End time: $(date)"
