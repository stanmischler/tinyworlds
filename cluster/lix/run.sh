#!/usr/bin/env bash
# Train the action model on one GPU. Run or submit it from the repo root, after cluster/lix/setup_env.sh. A plain script or a Slurm job:
#   bash cluster/lix/run.sh                                   the run of configs/experiments/small_action/weights_and_window.yaml
#   sbatch cluster/lix/run.sh                                 the same under Slurm (add your partition / account options to sbatch)
#   bash cluster/lix/run.sh weights_and_window error_weight=0.5 run_name=half_weights      another value of a knob, under another name
# Relaunching the same command continues the run from its last checkpoint. Outputs: results/<run_name>/small_action/
#   log.jsonl (training every 100 steps, scores every 10k), checkpoints/step_N/, visualizations/. Scores: python cluster/lix/scores.py
#SBATCH --job-name=action_model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
# the repo root: two folders above this script; under sbatch the script is a copy elsewhere, so the folder the job was submitted from
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/experiments/small_action/train.py" ] || ROOT="${SLURM_SUBMIT_DIR:?run this from the repo root}"
cd "$ROOT"
VENV="${VENV:-$PWD/.venv}"
export PYTHONPATH="$PWD" MPLBACKEND=Agg TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
mkdir -p logs results

RUN="${1:-weights_and_window}"; shift || true
C=configs/experiments/small_action
# Weights & Biases only when a key is set (on a node without internet also export WANDB_MODE=offline); the same numbers are always in log.jsonl
[ -n "${WANDB_API_KEY:-}" ] || set -- "$@" use_wandb=false
echo "run $RUN on $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
exec "$VENV/bin/python" -u experiments/small_action/train.py --config "$C/base.yaml,$C/simple_recipe.yaml,$C/$RUN.yaml" -- "$@"
