#!/usr/bin/env bash
# STA-76 drawing model: data -> train P (plain decoder) and F1 (flow-matching decoder), 50k steps each -> evals -> one
# tarball to send back. One local CUDA GPU, from the repo root after `git pull`. Every stage is resumable: if anything
# stops (crash, reboot, Ctrl-C), run the same command again and it continues from the last checkpoint (every 5k steps).
#
#   experiments/drawing/run_local.sh                  # both runs, one after the other, then the evals
#   experiments/drawing/run_local.sh p                # only P (or: f1)
#   GRAD_ACCUM=8 experiments/drawing/run_local.sh     # smaller GPU: more, smaller chunks (table in experiments/drawing/README.md)
#   PY=/path/to/python experiments/drawing/run_local.sh
#
# Outputs: results/drawing_<arm>/ (checkpoints, log.jsonl, prediction PNGs), eval JSONs, and drawing_results.tgz
# (logs + PNGs + scores, no weights) to send back.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
PY=${PY:-python}
ARMS=${1:-"p f1"}
GRAD_ACCUM=${GRAD_ACCUM:-4}
STEPS=${STEPS:-50000}           # STEPS / CKPT / EXTRA are for the smoke test; leave them alone for the real runs
CKPT=${CKPT:-5000}
EVAL_STEPS=${EVAL_STEPS:-"20000 50000"}
EXTRA=${EXTRA:-}
DEVICE=${DEVICE:-cuda}
RUN_PREFIX=${RUN_PREFIX:-drawing}   # run dirs results/<prefix>_<arm>

# 0. environment
$PY - <<EOF
import sys, torch
assert sys.version_info >= (3, 10), 'Python >= 3.10 needed'
if '$DEVICE' == 'cuda':
    assert torch.cuda.is_available(), 'no CUDA GPU visible'
    p = torch.cuda.get_device_properties(0)
    print(f'GPU {p.name}, {p.total_memory / 2**30:.0f} GiB, grad_accum $GRAD_ACCUM -> chunks of {256 // $GRAD_ACCUM} pairs')
EOF
$PY -c "import omegaconf, h5py, einops, huggingface_hub, lpips" 2>/dev/null || $PY -m pip install -r requirements.txt

# 1. data: the Zelda frames from HuggingFace, split exactly as ours (deterministic, scripts/data/split_dataset.py)
if [ ! -f data/zelda_train_frames.h5 ] || [ ! -f data/zelda_test_frames.h5 ]; then
  [ -f data/zelda_frames.h5 ] || $PY scripts/data/download_assets.py datasets --pattern "zelda_frames.h5"
  $PY scripts/data/split_dataset.py data/zelda_frames.h5
fi

# 2. training (checkpoint: auto -> resumes); the in-training action judge runs at every checkpoint
for arm in $ARMS; do
  echo "=== train $arm"
  $PY scripts/actions/train_como.py --config configs/como/zelda.yaml,configs/como/drawing/base.yaml,configs/como/drawing/$arm.yaml -- \
    run_name=${RUN_PREFIX}_$arm grad_accum=$GRAD_ACCUM n_updates=$STEPS ckpt_interval=$CKPT device=$DEVICE judge_device=$DEVICE $EXTRA
done

# 3. evals (same scripts and sets as J1 / the results table); skipped when the score file already exists
for arm in $ARMS; do
  for step in $EVAL_STEPS; do
    ck=results/${RUN_PREFIX}_$arm/como/checkpoints/como_step_$step
    [ -d "$ck" ] || { echo "no $ck, skipping its evals"; continue; }
    name=${RUN_PREFIX}_${arm}_$step
    out=results/${RUN_PREFIX}_$arm/evals
    fm_steps="1"; [ "$arm" = f1 ] && fm_steps="1 4"
    for s in $fm_steps; do  # next-frame PSNR / SSIM / LPIPS on the 890 held-out windows (+ shuffled action, copy, redraw)
      [ -f "$out/next_frame/${name}_s$s.json" ] || $PY scripts/eval/eval_como_pred.py --ckpt "$ck" --name "${name}_s$s" \
        --fm-steps "$s" --device "$DEVICE" --out-dir "$out/next_frame"
    done
    for k in 16 8; do  # action judge: Link's moves, NMI_adj over 10 k-means seeds
      [ -f "eval_results/lam_judge/${name}_k$k/score.json" ] || $PY scripts/eval/lam_judge.py score --device "$DEVICE" \
        --lam "${name}_k$k=$ck" --k "$k" --kmeans-seeds 10
    done
    [ -f "eval_results/lam_eval/zelda/$name/score.json" ] || $PY scripts/eval/lam_eval.py score --game zelda --device "$DEVICE" \
      --kmeans-seeds 10 --lam "$name=$ck"  # results-table action score
  done
done

# 4. bundle everything to send back (no weights)
files=$(ls -d results/${RUN_PREFIX}_*/como/log.jsonl results/${RUN_PREFIX}_*/como/visualizations results/${RUN_PREFIX}_*/evals \
  eval_results/lam_judge/${RUN_PREFIX}_* eval_results/lam_eval/zelda/${RUN_PREFIX}_* 2>/dev/null || true)
tar czf ${RUN_PREFIX}_results.tgz $files
echo "DONE -> ${RUN_PREFIX}_results.tgz"
