#!/bin/bash
#SBATCH --job-name=fcms_chexpert_multi_balanced  # Job name
#SBATCH --gres=gpu:1g:1             # Request 1 MIG slice (24 GB VRAM)
# #SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=6          # CPUs for data loading
                                    # RAM is auto-assigned from GPU VRAM
#SBATCH --time=24:00:00             # Max runtime (HH:MM:SS)
#SBATCH --output=logs/%j_%x.out     # Output: logs/<jobid>_<jobname>.out
#SBATCH --error=logs/%j_%x.err      # Errors: logs/<jobid>_<jobname>.err
#SBATCH --container-image=mamba
#SBATCH --partition blas
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
cd /home/fenogd/Federated-C2BM
source .venv/bin/activate
export PATH=~/graphviz-env/bin:$PATH

# Install extra dependencies (uncomment if you have a requirements.txt)
# pip install -r requirements.txt

# Set common PyTorch environment variables
export PYTHONUNBUFFERED=1

# Run training
cd /home/fenogd/Federated-C2BM_ari/Federated-C2BM/

echo "=== Start CheXpert Multi (all - v2) ==="
python main.py --config-name=test_chexpert_multi_balanced_v2 &
sleep 30
echo "=== Start CheXpert Multi (all - v3) ==="
python main.py --config-name=test_chexpert_multi_balanced_v3 &

wait  # Wait for all background processes to finish

echo "=== Start CheXpert Multi (static - v2) ==="
python main.py --config-name=test_chexpert_multi_balanced_static_v2 &
sleep 30
echo "=== Start CheXpert Multi (static - v3) ==="
python main.py --config-name=test_chexpert_multi_balanced_static_v3 &

wait  # Wait for all background processes to finish

echo "=== Start CheXpert Multi (all - 4) ==="
python main.py --config-name=test_chexpert_multi_balanced_v4 &
sleep 30
echo "=== Start CheXpert Multi (static - v4) ==="
python main.py --config-name=test_chexpert_multi_balanced_static_v4 &

wait  # Wait for all background processes to finish

echo "=== Job Finished ==="
echo "End time: $(date)"
