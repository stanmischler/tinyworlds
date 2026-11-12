#!/usr/bin/env bash
# The Bigfish action model, end to end on one GPU (32 GB or more): data, tokenizer checkpoint, 20-step smoke test,
# the 100,000-step run, the action report, and one archive of the results to send back.
#   bash scripts/infra/run_bigfish_action_model.sh             # everything; run it again after a stop: it resumes
#   bash scripts/infra/run_bigfish_action_model.sh --prepare   # data and checkpoint only (a node with internet),
#                                                              # then run it without the flag on the GPU node
# Before: put bigfish_tokenizer_100k.zip (sent separately) in the repo root. Every step is skipped when its output exists.
# PYTHON=<interpreter> picks the python (default: python); USE_WANDB=true logs to W&B (project tinyworlds);
# OVERRIDES="key=value ..." is passed to the training (e.g. a smaller button_check on a smaller GPU).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" MPLBACKEND=Agg
PY="${PYTHON:-python}"
CONFIG=configs/como/zelda.yaml,configs/como/bigfish.yaml
TOKENIZER=results/bigfish_tokenizer_long/video_tokenizer/checkpoints/video_tokenizer_step_99000
RUN=results/bigfish_action_model
REPORT=eval_results/action_report/bigfish_action_model

echo "== 1/6 tokenizer checkpoint"
if [ ! -f "$TOKENIZER/model_state_dict.pt" ]; then
    [ -f bigfish_tokenizer_100k.zip ] || { echo "missing: put bigfish_tokenizer_100k.zip in $PWD and run again"; exit 1; }
    unzip -oq bigfish_tokenizer_100k.zip
fi

echo "== 2/6 data (downloads 8.7 GB from Hugging Face the first time)"
"$PY" -c "import pyarrow, h5py, omegaconf, torch" 2>/dev/null || "$PY" -m pip install -q -r requirements.txt
if [ ! -f data/bigfish_frames.h5 ] || [ ! -f data/bigfish_test_frames.h5 ]; then
    "$PY" scripts/data/convert_procgen.py --game bigfish
    rm -rf data/procgen_raw
fi
[ "${1:-}" = "--prepare" ] && { echo "prepared: data and checkpoint are in place"; exit 0; }

echo "== 3/6 smoke test (20 steps)"
if ! ls "$RUN"/como/checkpoints/como_step_* >/dev/null 2>&1; then  # skipped once the run has a checkpoint
    rm -rf results/bigfish_action_smoke
    "$PY" scripts/actions/train_como.py --config "$CONFIG" -- ${OVERRIDES:-} run_name=bigfish_action_smoke n_updates=20 \
        log_interval=10 val_interval=10 ckpt_interval=10 button_check=256 use_wandb=false
    rm -rf results/bigfish_action_smoke
fi

echo "== 4/6 training (100,000 steps, resumes from the last checkpoint)"
"$PY" scripts/actions/train_como.py --config "$CONFIG" -- use_wandb="${USE_WANDB:-false}" ${OVERRIDES:-}

echo "== 5/6 action report"
LAST=$(ls -d "$RUN"/como/checkpoints/como_step_* | sort -t_ -k5 -n | tail -1)  # the final checkpoint
"$PY" scripts/eval/action_report.py --ckpt "$LAST" --out-dir "$REPORT"

echo "== 6/6 results archive"
# the log, the prediction images, the report, and every checkpoint's weights (without the optimizer state)
tar czf bigfish_action_model_results.tgz "$RUN/como/log.jsonl" "$RUN/como/visualizations" "$REPORT" \
    "$RUN"/como/checkpoints/como_step_*/model_state_dict.pt "$RUN"/como/checkpoints/como_step_*/state.pt
echo "done: send back $PWD/bigfish_action_model_results.tgz ($(du -h bigfish_action_model_results.tgz | cut -f1))"
