#!/bin/bash
# score one dynamics run at 32 Euler steps, checkpoints 100000 and 199999 (the last), dev and confirm halves; run from the run folder
#   submit: sbatch -A ehr@a100 -C a100 --qos=qos_gpu-dev --time=2:00:00 cluster/jeanzay/eval_dynamics.sh <run>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
RUN="$1"
module purge; module load arch/a100 pytorch-gpu/py3/2.8.0
source "$WORK/envs/tinyworlds-a100/bin/activate"
export WANDB_MODE=offline MPLBACKEND=Agg PYTHONPATH="$PWD" HF_HOME="$WORK/hf_home" HF_HUB_OFFLINE=1 TORCH_HOME="$WORK/torch_home"
OUT=results/evals/next_frame; mkdir -p "$OUT"
for STEP in ${STEPS:-100000 199999}; do
  for SPLIT in dev confirm; do
    NAME="${RUN}_${STEP}_${SPLIT}_s32"
    [ -f "$OUT/${NAME}_score.json" ] && { echo "done already: $NAME"; continue; }
    python -u scripts/eval/eval_next_frame.py --device cuda --test-h5 data/zelda_test_frames.h5 \
      --video-tokenizer-path results/zelda_more_data_tokenizer/video_tokenizer/checkpoints/video_tokenizer_step_59000 \
      --latent-actions-path results/small_action_actions_oversampling_crop_shift_200000_yt2 \
      --dynamics-path "results/$RUN/dynamics/checkpoints/dynamics_step_$STEP" \
      --link-boxes evaluation/zelda_link_boxes.json --split-file eval_results/action_eval/zelda/set/split.json \
      --split "$SPLIT" --num-steps 32 --save-predictions --out-dir "$OUT" --name "$NAME"
    python -u experiments/small_action/dynamics_score.py --split "$SPLIT" --eval "$RUN=$OUT/$NAME.json" --out "$OUT/${NAME}_score.json"
    echo "SCORED $NAME"
  done
done
