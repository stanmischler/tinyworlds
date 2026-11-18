#!/usr/bin/env bash
# Score the dynamics run "still walks oversampled" once its training has ended: one job on one Jean Zay GPU, the held-out eval of
# cluster/jeanzay/eval_dynamics.py (32 Euler steps on the dev or confirm half, then the split scores). Run it on the login node,
# from the synced run folder:
#   bash cluster/jeanzay/eval_still_walks_oversampled.sh dev                              # this run, A100
#   bash cluster/jeanzay/eval_still_walks_oversampled.sh confirm                          # once, after dev
#   bash cluster/jeanzay/eval_still_walks_oversampled.sh confirm small_action_dynamics    # the run it is compared to: must give
#                                                                                         # still-walk LPIPS .098, PSNR 32.20
#   bash cluster/jeanzay/eval_still_walks_oversampled.sh dev still_walks_oversampled h100 # on an H100
#   TEST_ONLY=1 bash cluster/jeanzay/eval_still_walks_oversampled.sh dev                  # submits nothing: Slurm's check and estimated start
# STEP=<n> scores another checkpoint than the last step of the 100k (99999). JZ_PROJECT=<project> pays (default: ehr).
# Outputs: the RESULT line at the end of logs/eval_<run>_<split>_<job id>.out, and results/evals/next_frame/<run>_<split>_s32.json
# (every number per window), _score.json (the half's scores), _pred.npz (predicted frames), .png.
set -euo pipefail
cd "$(dirname "$0")/../.."
PROJECT="${JZ_PROJECT:-ehr}"
eval "$(idrenv -d "$PROJECT")"  # $WORK of the project, not of the login's default one

SPLIT="${1:-dev}"
RUN="${2:-still_walks_oversampled}"
GPU="${3:-a100}"
STEP="${STEP:-99999}"
case "$SPLIT" in dev|confirm) ;; *) echo "usage: $0 dev|confirm [run] [a100|h100]"; exit 1 ;; esac
case "$GPU" in
    a100) OPTIONS=(-A "$PROJECT@a100" -C a100 --qos=qos_gpu_a100-dev --cpus-per-task=8) ;;  # the queue of jobs under 2 h: starts sooner
    h100) OPTIONS=(-A "$PROJECT@h100" -C h100 --qos=qos_gpu_h100-dev --cpus-per-task=24) ;;
    *) echo "usage: $0 dev|confirm [run] [a100|h100]"; exit 1 ;;
esac

CKPT="results/$RUN/dynamics/checkpoints/dynamics_step_$STEP"
SET=eval_results/action_eval/zelda/set
# everything the job reads must be there: GPU nodes have no internet, and a job that stops on a missing file still waited in the queue
for f in "$CKPT/model_state_dict.pt" "$CKPT/ema_state_dict.pt" "$CKPT/state.pt" data/zelda_test_frames.h5 \
         results/zelda_more_data_tokenizer/video_tokenizer/checkpoints/video_tokenizer_step_59000/model_state_dict.pt \
         results/zelda_more_data_tokenizer/video_tokenizer/checkpoints/video_tokenizer_step_59000/state.pt \
         results/small_action_actions_late_weight_focus_200000_yt2/state.pt \
         results/late_weight_focus/small_action/checkpoints/step_200000/model_state_dict.pt \
         results/late_weight_focus/small_action/checkpoints/step_200000/state.pt \
         evaluation/zelda_link_boxes.json "$SET/split.json" "$SET/pairs.json" "$SET/labels.json" "$SET/link_positions.npz" \
         "$WORK/torch_home/hub/checkpoints/alexnet-owt-7be5be79.pth" "$WORK/torch_home/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth" \
         "$WORK/envs/tinyworlds-$GPU/bin/python"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
NAME="${RUN}_${SPLIT}_s32"
if [ -e "results/evals/next_frame/$NAME.json" ]; then  # a half is scored once per run: confirm is not a tuning set
    echo "already scored: results/evals/next_frame/$NAME.json (move it away to score again)"; exit 1
fi
mkdir -p logs

COMMAND=(cluster/jeanzay/train.slurm cluster/jeanzay/eval_dynamics.py "$RUN" --split "$SPLIT" --step "$STEP")
sbatch ${TEST_ONLY:+--test-only} --job-name="eval_${RUN}_${SPLIT}" --time=01:00:00 "${OPTIONS[@]}" "${COMMAND[@]}"
