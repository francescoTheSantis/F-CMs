#!/bin/bash
#SBATCH --job-name=pytorch-train
#SBATCH --gres=gpu:1g:1             # blas: 1 MIG slice (24 GB VRAM)
# Alternative GRES for adula (uncomment one, comment out the line above):
#   #SBATCH --gres=shard:4           # adula: 1 GPU-equivalent (48 GB shared VRAM)
#   #SBATCH --gres=gpu:rtx_a6000:1   # adula: 1 exclusive GPU (48 GB VRAM)
# To target a specific node, add: #SBATCH -w blas  (or -w adula)
#SBATCH --cpus-per-task=6           # CPUs for data loading
                                    # RAM is auto-assigned from GPU VRAM
#SBATCH --time=24:00:00             # Max runtime (HH:MM:SS)
#SBATCH --output=logs/%j_%x.out     # Output: logs/<jobid>_<jobname>.out
#SBATCH --error=logs/%j_%x.err      # Errors: logs/<jobid>_<jobname>.err
#SBATCH --container-image=pip
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
# source ~/my_project/.venv/bin/activate

# Install extra dependencies (uncomment if you have a requirements.txt)
# pip install -r requirements.txt

# Set common PyTorch environment variables
export PYTHONUNBUFFERED=1

# Run training
# python train.py \
#     --batch-size 64 \
#     --epochs 100 \
#     --lr 0.001 \
#     --data-dir /data/$USER/dataset \
#     --output-dir /data/$USER/checkpoints

python show_results.py

echo "=== Job Finished ==="
echo "End time: $(date)"
