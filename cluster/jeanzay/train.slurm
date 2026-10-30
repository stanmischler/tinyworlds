#!/bin/bash
# Run `python <args>` on one Jean Zay GPU, from the run folder this job is submitted in.
#   sbatch cluster/jeanzay/train.slurm scripts/dynamics/train_dynamics.py --config <dyn yaml> --training_config <training yaml> -- key=val   # A100, up to 20 h
#   sbatch --qos=qos_gpu_a100-dev --time=00:10:00 cluster/jeanzay/train.slurm …           # test queue, 2 h max
#   sbatch -A hsq@h100 -C h100 --qos=qos_gpu_h100-t3 --cpus-per-task=24 cluster/jeanzay/train.slurm …   # H100
#   sbatch --array=0-3 cluster/jeanzay/train.slurm --lines sweep.txt                       # sweep: task i runs line i+1
#   sbatch --dependency=afterany:<jobid> cluster/jeanzay/train.slurm …                    # chain past the 20 h limit
# Options given to sbatch override the defaults below. Logs go to logs/<job name>_<job id>.out (each array task has its own job id).
#SBATCH --job-name=tinyworlds
#SBATCH --account=hsq@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

module purge
case "$SLURM_JOB_PARTITION" in  # each GPU type has its own software stack, and its own venv
    gpu_p5*) ARCH=a100 ;;
    gpu_p6*) ARCH=h100 ;;
esac
module load arch/$ARCH pytorch-gpu/py3/2.8.0
source "$WORK/envs/tinyworlds-$ARCH/bin/activate"

# no internet on GPU nodes: W&B offline (sync from the login node), weights from the caches setup_env.sh filled
export WANDB_MODE=offline MPLBACKEND=Agg PYTHONPATH="$PWD"
export HF_HOME="$WORK/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$WORK/torch_home"
# torch.compile caches, kept across jobs
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/tinyworlds_cache/inductor" TRITON_CACHE_DIR="$SCRATCH/tinyworlds_cache/triton"

if [ "${1:-}" = "--lines" ]; then  # one sweep setting per line of the file
    set -- $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$2")
fi

echo "job $SLURM_JOB_ID${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on $(hostname): python $*"
srun python -u "$@"
