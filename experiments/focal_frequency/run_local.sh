#!/usr/bin/env bash
# Focal log-frequency loss on the Zelda flow-matching dynamics: data + frozen weights -> "frequency loss only" run
# (f-loss alone, 60k steps) -> switch step and loss scale from where its test curve crosses the v-loss baseline's ->
# "focal frequency loss" run (f-loss -> v-loss, 100k steps) -> evals of both runs and of the baseline -> one tarball to
# send back. One local CUDA GPU, from the repo root after `git pull`. Every stage is resumable: if anything stops, run
# the same command again; finished stages are skipped and training continues from the last checkpoint.
#
#   experiments/focal_frequency/run_local.sh
#   BATCH=32 ACCUM=2 experiments/focal_frequency/run_local.sh   # smaller GPU: same 64-window batch in two chunks
#   PY=/path/to/python experiments/focal_frequency/run_local.sh
#
# Details: experiments/focal_frequency/README.md. Outputs: results/zelda_{frequency_only,focal_frequency}_fm/ (checkpoints,
# test_curve.jsonl, train_log.jsonl, PNGs), results/focal_frequency_evals/ (eval JSONs + PNGs, switch.json + curve plot)
# and focal_frequency_results.tgz (configs, logs, PNGs, scores; no weights) to send back.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
[ "${1:-}" = -h ] || [ "${1:-}" = --help ] && { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
if [ -z "${PY:-}" ]; then  # the repo's .venv if there is one, else python3 / python on PATH
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; elif command -v python3 >/dev/null; then PY=python3; else PY=python; fi
fi
BATCH=${BATCH:-64}               # windows per chunk; BATCH x ACCUM must stay 64 (the baseline's batch)
ACCUM=${ACCUM:-1}
STEPS=${STEPS:-100000}           # STEPS / F_STOP / LOG / TEST_EVERY / LIMIT / EXTRA are for the smoke test; leave them for real runs
F_STOP=${F_STOP:-60000}          # the frequency-loss-only run stops here (lr schedule of STEPS)
LOG=${LOG:-1000}                 # checkpoint every LOG steps
TEST_EVERY=${TEST_EVERY:-2000}   # test curve every TEST_EVERY steps (must match the baseline curve's 2000 for the crossing)
LIMIT=${LIMIT:-}                 # >0: final evals on the first LIMIT windows only (smoke)
SMOKE=${SMOKE:-}                 # 1: allow a batch other than 64 (smoke test only)
EXTRA=${EXTRA:-}                 # extra key=value overrides for both training runs
DEVICE=${DEVICE:-cuda}
ASSETS_REPO=${ASSETS_REPO-stanmischler/tinyworlds-focal-assets}  # set it empty to skip the download
CFG=configs/experiments/zelda_focal_frequency
EVALS=results/focal_frequency_evals
mkdir -p "$EVALS"
[ $((BATCH * ACCUM)) -eq 64 ] || [ -n "$SMOKE" ] || { echo "BATCH x ACCUM must be 64, got $BATCH x $ACCUM"; exit 1; }

# 0. environment
$PY - <<EOF
import sys, torch
assert sys.version_info >= (3, 10), 'Python >= 3.10 needed'
if '$DEVICE' == 'cuda':
    assert torch.cuda.is_available(), 'no CUDA GPU visible'
    p = torch.cuda.get_device_properties(0)
    print(f'GPU {p.name}, {p.total_memory / 2**30:.0f} GiB')
EOF
# missing packages: install them, but never let pip replace an existing (CUDA) torch build
if ! $PY -c "import omegaconf, h5py, einops, lpips, wandb, huggingface_hub, matplotlib, tqdm" 2>/dev/null; then
  if $PY -c "import torch" 2>/dev/null; then
    req=$(mktemp); grep -viE '^(torch|torchvision)([<>=!~ ]|$)' requirements.txt > "$req"
    $PY -m pip install -r "$req"
  else
    $PY -m pip install -r requirements.txt
  fi
fi
$PY -c "import torch; v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); assert v >= (2, 8), f'torch >= 2.8 needed, found {torch.__version__}'"
export WANDB_MODE=${WANDB_MODE:-offline}  # stage scripts log to W&B; offline unless you `wandb login` and set WANDB_MODE=online

# 1. data: the Zelda frames from HuggingFace, split exactly as ours (deterministic, scripts/data/split_dataset.py), and
# the frozen weights (tokenizer, CoMo actions + their train .npy, the baseline dynamics for its eval) from the assets repo
if [ ! -f data/zelda_train_frames.h5 ] || [ ! -f data/zelda_test_frames.h5 ]; then
  [ -f data/zelda_frames.h5 ] || $PY scripts/data/download_assets.py datasets --pattern "zelda_frames.h5"
  $PY scripts/data/split_dataset.py data/zelda_frames.h5
fi
if [ -n "$ASSETS_REPO" ]; then  # ASSETS_REPO= (empty) when the weights are already in place
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$ASSETS_REPO', local_dir='.')"  # skips present files
fi
TOK=results/zelda_cont_final_tp/video_tokenizer/checkpoints/video_tokenizer_step_29000
ACTIONS=results/como_actions_v1_full
BASELINE=results/zelda_longer_fm/dynamics/checkpoints/dynamics_step_99999

latest() {  # newest complete dynamics checkpoint dir of a run (a save cut off midway has no EMA), or nothing
  local d
  for d in $(ls -d results/$1/dynamics/checkpoints/dynamics_step_* 2>/dev/null | sort -V -r); do
    [ -f "$d/state.pt" ] && [ -f "$d/ema_state_dict.pt" ] && { echo "$d"; return; }
  done
  true
}

train() {  # run stage_yaml last_step [overrides...]; skipped when the last checkpoint is the final step
  local run=$1 syaml=$2 final=$3; shift 3
  local last; last=$(latest $run)
  if [ -n "$last" ] && [ "${last##*_step_}" -ge "$final" ]; then echo "=== $run done ($last)"; return; fi
  echo "=== train $run${last:+ (resume from $last)}"
  NG_RUN_ROOT_DIR="$PWD/results/$run" $PY scripts/dynamics/train_dynamics.py --config "$syaml" --training_config $CFG/training.yaml -- \
    n_updates=$STEPS log_interval=$LOG test_eval_interval=$TEST_EVERY batch_size_per_gpu=$BATCH gradient_accumulation_steps=$ACCUM \
    ${last:+checkpoint=$last} "$@" $EXTRA
}

# 2. frequency loss only: f-loss alone, on the 100k lr schedule, stopped at F_STOP
train zelda_frequency_only_fm $CFG/frequency_only.yaml $((F_STOP - 1)) stop_step=$F_STOP

# 3. switch step = where its test PSNR curve crosses the baseline's; loss scale = v-loss / f-loss there
$PY experiments/focal_frequency/pick_switch.py results/zelda_frequency_only_fm/dynamics "$EVALS/switch.json" "$EVALS/curves.png"
read -r SWITCH SCALE RULE < <($PY -c "import json; d = json.load(open('$EVALS/switch.json')); print(d['f_switch_step'], float(d['f_loss_scale']), d['rule'])")
case $RULE in no*) echo "WARNING: $RULE -> switch at $SWITCH (see $EVALS/curves.png); the focal run still follows the protocol" ;; esac

# 4. focal frequency loss: lambda * scale * f-loss + (1 - lambda) * v-loss, lambda from 1 to 0 around the switch step
train zelda_focal_frequency_fm $CFG/focal_frequency.yaml $((STEPS - 1)) f_switch_step=$SWITCH f_loss_scale=$SCALE

# 5. evals: the 890 test windows, 1 and 4 Euler steps (eval_next_frame.py: PSNR / SSIM / LPIPS + spectrum bands)
evaluate() {  # name dynamics_checkpoint
  local s
  for s in 1 4; do
    [ -f "$EVALS/${1}_s$s.json" ] || $PY scripts/eval/eval_next_frame.py --test-h5 data/zelda_test_frames.h5 \
      --video-tokenizer-path $TOK --latent-actions-path $ACTIONS --dynamics-path "$2" --num-steps $s --device "$DEVICE" \
      --out-dir "$EVALS" --name "${1}_s$s" ${LIMIT:+--limit $LIMIT}
  done
}
evaluate longer_flow_matching $BASELINE
evaluate frequency_loss_only "$(latest zelda_frequency_only_fm)"
evaluate focal_frequency_loss "$(latest zelda_focal_frequency_fm)"

# 6. one table + the tarball to send back (no weights)
$PY - <<EOF
import json
rows = [('longer flow matching', 'longer_flow_matching'), ('frequency loss only', 'frequency_loss_only'),
        ('focal frequency loss', 'focal_frequency_loss')]
print(f"{'run':22s} {'steps':>5s} {'PSNR':>6s} {'SSIM':>6s} {'LPIPS':>6s} {'LPIPS@4':>7s} {'mid %':>6s} {'high %':>6s}")
for label, name in rows:
    r1, r4 = (json.load(open(f'$EVALS/{name}_s{s}.json')) for s in (1, 4))
    m, sp = r1['summary']['model'], r1['spectrum']['pred_vs_recon']
    step = r1['config']['dynamics_path'].rsplit('_', 1)[-1]
    print(f"{label:22s} {step:>5s} {m['psnr']:6.2f} {m['ssim']:6.3f} {m['lpips']:6.3f} {r4['summary']['model']['lpips']:7.3f} {sp['mid']:+6.1f} {sp['high']:+6.1f}")
print('mid / high %: power of the predicted frames vs the tokenizer reconstruction at 0.06-0.25 / 0.25-0.5 cycles per pixel')
EOF
tar czf focal_frequency_results.tgz $CFG experiments/focal_frequency "$EVALS" \
  $(ls -d results/zelda_{frequency_only,focal_frequency}_fm/dynamics/{test_curve.jsonl,train_log.jsonl,visualizations} 2>/dev/null)
echo "done: send focal_frequency_results.tgz"
