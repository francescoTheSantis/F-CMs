#!/bin/bash
#SBATCH --job-name=fcms_siim  # Job name
# #SBATCH --gres=gpu:1g:1             # Request 1 MIG slice (24 GB VRAM)
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=6          # CPUs for data loading
                                    # RAM is auto-assigned from GPU VRAM
#SBATCH --time=30:00:00             # Max runtime (HH:MM:SS)
#SBATCH --output=logs/%j_%x.out     # Output: logs/<jobid>_<jobname>.out
#SBATCH --error=logs/%j_%x.err      # Errors: logs/<jobid>_<jobname>.err
#SBATCH --container-image=mamba
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
echo "=== Start SIIM (all) ==="
python main.py --config-name=test_siim &
sleep 30
echo "=== Start SIIM (static) ==="
python main.py --config-name=test_siim_static &

wait  # Wait for all background processes to finish

echo "=== Job Finished ==="
echo "End time: $(date)"
