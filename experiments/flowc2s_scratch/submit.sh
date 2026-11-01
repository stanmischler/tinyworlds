#!/usr/bin/env bash
# Submit FlowC2S-from-scratch jobs on Jean Zay (H100), from the run folder (after cluster/jeanzay/sync.sh):
#   bash experiments/flowc2s_scratch/submit.sh sweep              9 runs, stopped at 10k on the 100k schedule, + their evals
#   bash experiments/flowc2s_scratch/submit.sh final <run>        resume <run> from step 9999 to 100k, + its eval
# Each training job is followed (afterok) by eval_next_frame.py on its last checkpoint: 890 held-out windows, CoMo
# actions, 32 Euler steps, Link metrics. Outputs: results/<run>/ and results/evals/<run>/next_frame/step_<N>.json.
set -euo pipefail
CFG=configs/experiments/zelda_flowc2s_scratch
H100=(-A hsq@h100 -C h100 --qos=qos_gpu_h100-t3 --cpus-per-task=24)
TOKENIZER=results/zelda_cont_final_tp/video_tokenizer/checkpoints/video_tokenizer_step_29000
ACTIONS=results/como_actions_v1_full

train() {  # train <time limit> <run> <yaml stem> <overrides...> -> prints the job id
    local time=$1 run=$2 stem=$3; shift 3
    NG_RUN_ROOT_DIR=$PWD/results/$run sbatch --parsable "${H100[@]}" --time=$time -J "$run" cluster/jeanzay/train.slurm \
        scripts/dynamics/train_dynamics.py --config $CFG/$stem.yaml --training_config $CFG/training.yaml -- "$@"
}

evaluate() {  # evaluate <after job id> <run> <step>
    local job=$1 run=$2 step=$3
    sbatch --parsable "${H100[@]}" --time=01:00:00 --dependency=afterok:$job -J "eval_$run" cluster/jeanzay/train.slurm \
        scripts/eval/eval_next_frame.py --test-h5 data/zelda_test_frames.h5 --video-tokenizer-path $TOKENIZER \
        --latent-actions-path $ACTIONS --dynamics-path results/$run/dynamics/checkpoints/dynamics_step_$step \
        --action-mode lam --num-steps 32 --link-boxes evaluation/zelda_link_boxes.json \
        --out-dir results/evals/$run/next_frame --name step_$step
}

case "${1:-}" in
    sweep)
        for s in 01 03 05; do
            for m in "" _motion01 _motion03; do
                run=zelda_flowc2s_s$s$m
                job=$(train 03:00:00 $run flowc2s_s$s$m stop_step=10000)  # ~1.5 h + compile
                echo "$run train $job eval $(evaluate $job $run 9999)"
            done
        done ;;
    final)
        run=$2
        job=$(train 20:00:00 $run ${run#zelda_} checkpoint=results/$run/dynamics/checkpoints/dynamics_step_9999 \
            seed=1)  # new seed: seed 0 would replay the first 10k steps' batches and noise (the weights come from the checkpoint)
        echo "$run train $job eval $(evaluate $job $run 99999)" ;;
    *)
        echo "usage: $0 sweep | final <run>"; exit 1 ;;
esac
