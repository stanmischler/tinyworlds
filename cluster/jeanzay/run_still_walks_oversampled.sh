#!/usr/bin/env bash
# Submit the dynamics run "still walks oversampled" (configs/experiments/small_action_dynamics/still_walks_oversampled.yaml, 100k
# steps) on one Jean Zay GPU, as a chain of jobs: each one continues from the last checkpoint (cluster/jeanzay/resume_dynamics.py)
# and starts when the one before ends, whatever its end (20 h limit, crash, done); a job with nothing left to train exits at once.
# Run it on the login node, from the synced run folder (cluster/jeanzay/sync.sh, then setup_env.sh once):
#   bash cluster/jeanzay/run_still_walks_oversampled.sh a100        # 3 jobs of up to 20 h on an A100
#   bash cluster/jeanzay/run_still_walks_oversampled.sh h100        # 2 jobs on an H100 (13.5 h on Modal's: the second is a spare)
#   bash cluster/jeanzay/run_still_walks_oversampled.sh a100 5      # another number of jobs
#   TEST_ONLY=1 bash cluster/jeanzay/run_still_walks_oversampled.sh a100   # submits nothing: Slurm's check and estimated start
# Relaunching it later continues the run. JZ_PROJECT=<project> pays (default: ehr). Outputs: results/still_walks_oversampled/dynamics/,
# logs/still_walks_oversampled_<job id>.out, W&B offline in wandb/ (wandb sync wandb/offline-run-* from the login node).
set -euo pipefail
cd "$(dirname "$0")/../.."
PROJECT="${JZ_PROJECT:-ehr}"
eval "$(idrenv -d "$PROJECT")"  # $WORK of the project, not of the login's default one

GPU="${1:-a100}"
case "$GPU" in
    a100) JOBS="${2:-3}"; OPTIONS=(-A "$PROJECT@a100" -C a100 --qos=qos_gpu_a100-t3 --cpus-per-task=8) ;;
    h100) JOBS="${2:-2}"; OPTIONS=(-A "$PROJECT@h100" -C h100 --qos=qos_gpu_h100-t3 --cpus-per-task=24) ;;
    *) echo "usage: $0 a100|h100 [jobs]"; exit 1 ;;
esac

RUN=still_walks_oversampled
C=configs/experiments/small_action_dynamics
# everything the job reads must be there: GPU nodes have no internet, and a job that stops on a missing file still waited in the queue
for f in data/zelda_train_frames.h5 data/zelda_youtube data/zelda_more_train_small_action_late_weight_focus_200000_yt2.npy \
         data/zelda_five_hours_shifts.npz data/zelda_five_hours_motion.npz \
         results/zelda_more_data_tokenizer/video_tokenizer/checkpoints/video_tokenizer_step_59000/model_state_dict.pt \
         results/small_action_actions_late_weight_focus_200000_yt2/state.pt \
         results/late_weight_focus/small_action/checkpoints/step_200000/model_state_dict.pt "$WORK/envs/tinyworlds-$GPU/bin/python"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
mkdir -p logs

COMMAND=(cluster/jeanzay/train.slurm cluster/jeanzay/resume_dynamics.py "$RUN" --config "$C/$RUN.yaml" --training_config "$C/training.yaml")
if [ -n "${TEST_ONLY:-}" ]; then
    sbatch --test-only --job-name="$RUN" "${OPTIONS[@]}" "${COMMAND[@]}"
    exit
fi
PREVIOUS=
for _ in $(seq "$JOBS"); do
    PREVIOUS=$(sbatch --parsable --job-name="$RUN" "${OPTIONS[@]}" ${PREVIOUS:+--dependency=afterany:$PREVIOUS} "${COMMAND[@]}")
    echo "submitted job $PREVIOUS"
done
