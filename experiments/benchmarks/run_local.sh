#!/usr/bin/env bash
# Benchmark runs on published datasets: data -> copy baselines -> NanoWM re-scoring -> training -> evals -> one
# tarball to send back. One local CUDA GPU, from the repo root after `git pull`. Every stage is resumable: if anything
# stops, run the same command again; finished stages are skipped and training continues from the last checkpoint.
#
#   experiments/benchmarks/run_local.sh point_maze            # one run
#   experiments/benchmarks/run_local.sh point_maze wall       # several runs, in that order
#   experiments/benchmarks/run_local.sh all                   # point_maze wall obj3d
#   SKIP_NANOWM=1 experiments/benchmarks/run_local.sh wall    # skip re-scoring NanoWM's checkpoints
#   PY=/path/to/python experiments/benchmarks/run_local.sh all
#
# Runs (configs/experiments/benchmarks/, details in experiments/benchmarks/README.md):
#   point_maze (point_maze_flow)    KL tokenizer + flow-matching dynamics, real actions, 256 px     -> NanoWM protocol (1 -> 3 frames)
#   wall (wall_flow)                same on Wall                                                 -> NanoWM protocol
#   obj3d (obj3d_como_actions)      KL tokenizer, then CoMo on its frozen features, then dynamics on CoMo's continuous z, 128 px
#                                   -> LPWM protocol (6 -> 44 frames)
# Outputs: results/<run>/ (checkpoints, PNGs), eval JSONs + PNGs in results/benchmark_evals/, and benchmark_results.tgz
# (configs, logs, PNGs, scores; no weights) to send back.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
[ $# -gt 0 ] || usage 1
RUNS=""
for a in "$@"; do
  case $a in
    -h|--help) usage ;;
    all) RUNS="$RUNS point_maze_flow wall_flow obj3d_como_actions" ;;
    point_maze|point_maze_flow) RUNS="$RUNS point_maze_flow" ;;
    wall|wall_flow) RUNS="$RUNS wall_flow" ;;
    obj3d|obj3d_como_actions) RUNS="$RUNS obj3d_como_actions" ;;
    *) echo "unknown run '$a' (choose from: point_maze wall obj3d all)"; exit 1 ;;
  esac
done
echo "runs:$RUNS"
if [ -z "${PY:-}" ]; then  # the repo's .venv if there is one, else python3 / python on PATH
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; elif command -v python3 >/dev/null; then PY=python3; else PY=python; fi
fi
STEPS=${STEPS:-30000}           # STEPS / COMO_STEPS / LOG / LIMIT / EXTRA are for the smoke test; leave them for real runs
COMO_STEPS=${COMO_STEPS:-20000}
LOG=${LOG:-1000}                # checkpoint every LOG steps; the last one is STEPS - LOG
LIMIT=${LIMIT:-}                # >0: evals on the first LIMIT clips only (smoke)
EXTRA=${EXTRA:-}                # extra key=value overrides for the tokenizer / dynamics stages
COMO_EXTRA=${COMO_EXTRA:-}      # extra key=value overrides for CoMo
ACTIONS_LIMIT=${ACTIONS_LIMIT:-} # >0: CoMo z for the first N frames of each split only (smoke)
DEVICE=${DEVICE:-cuda}
SKIP_NANOWM=${SKIP_NANOWM:-}
CFG=configs/experiments/benchmarks
EVALS=results/benchmark_evals
mkdir -p "$EVALS"
lim() { [ -n "$LIMIT" ] && echo "--limit $LIMIT --no-fid" || true; }
lim_lpwm() { [ -n "$LIMIT" ] && echo "--limit $LIMIT" || true; }

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
if ! $PY -c "import omegaconf, h5py, einops, piqa, lpips, pytorch_fid, cv2, wandb, huggingface_hub, scipy, matplotlib, tqdm" 2>/dev/null; then
  if $PY -c "import torch" 2>/dev/null; then
    req=$(mktemp); grep -viE '^(torch|torchvision)([<>=!~ ]|$)' requirements.txt > "$req"
    $PY -m pip install -r "$req"
  else
    $PY -m pip install -r requirements.txt
  fi
fi
$PY -c "import torch; v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); assert v >= (2, 8), f'torch >= 2.8 needed, found {torch.__version__}'"
export WANDB_MODE=${WANDB_MODE:-offline}  # stage scripts log to W&B; offline unless you `wandb login` and set WANDB_MODE=online

# 1. data (downloads: DINO-WM Point Maze 0.7 GB + Wall 1.7 GB from OSF, OBJ3D 7.4 GB from HuggingFace), only for the
# requested runs
envs=""
for run in $RUNS; do case $run in point_maze_flow) envs="$envs point_maze" ;; wall_flow) envs="$envs wall" ;; obj3d_*) envs="$envs obj3d" ;; esac; done
has() { [[ " $envs " == *" $1 "* ]]; }
has point_maze && { [ -f data/point_maze_frames.h5 ] && [ -f data/point_maze_val_frames.h5 ] || $PY scripts/data/convert_dinowm.py --env point_maze; }
has wall && { [ -f data/wall_frames.h5 ] && [ -f data/wall_val_frames.h5 ] || $PY scripts/data/convert_dinowm.py --env wall; }
has obj3d && { [ -f data/obj3d_train_frames.h5 ] && [ -f data/obj3d_val_frames.h5 ] && [ -f data/obj3d_test_frames.h5 ] || $PY scripts/data/convert_obj3d.py; }

# 2. copy-last baselines (no model)
for env in point_maze wall; do
  has $env || continue
  [ -f "$EVALS/copy_last_$env.json" ] || $PY scripts/eval/eval_nanowm.py --env $env --copy-only --device "$DEVICE" \
    --out-dir "$EVALS" --name "copy_last_$env" $(lim)
done
if has obj3d; then
  [ -f "$EVALS/copy_last_obj3d.json" ] || $PY scripts/eval/eval_lpwm.py --copy-only --device "$DEVICE" --out-dir "$EVALS" \
    --name copy_last_obj3d $(lim_lpwm)
fi

# 3. NanoWM's released checkpoints on the same clips (their code, pinned commit, own venv; ~15-30 min each)
if [ -z "$SKIP_NANOWM" ]; then
  for env in point_maze wall; do
    has $env || continue
    [ -f "results/nanowm_${env}_rescore/predictions_f16.npz" ] || \
      DATASET_DIR="$PWD/data/dinowm_raw" bash experiments/pusht_nanowm/nanowm_rescore_local.sh $env "results/nanowm_${env}_rescore"
  done
fi

latest() {  # newest complete checkpoint dir of a stage (a save cut off midway has no state.pt / EMA), or nothing
  local d need=state.pt
  [ "$2" = dynamics ] && need=ema_state_dict.pt
  for d in $(ls -d results/$1/$2/checkpoints/$2_step_* 2>/dev/null | sort -V -r); do
    [ -f "$d/state.pt" ] && [ -f "$d/$need" ] && { echo "$d"; return; }
  done
  true
}
final_step=$((STEPS - LOG))

train_stage() {  # run stage script training_yaml stage_yaml [overrides...]; skipped when the last checkpoint exists
  local run=$1 stage=$2 script=$3 tyaml=$4 syaml=$5; shift 5
  local last; last=$(latest $run $stage)
  if [ -n "$last" ] && [ "${last##*_step_}" -ge "$final_step" ]; then echo "=== $run/$stage done ($last)"; return; fi
  echo "=== train $run/$stage${last:+ (resume from $last)}"
  NG_RUN_ROOT_DIR="$PWD/results/$run" $PY "$script" --config "$syaml" --training_config "$tyaml" -- \
    n_updates=$STEPS log_interval=$LOG ${last:+checkpoint=$last} "$@" $EXTRA
}

tok_and_dyn() {  # run training_yaml dynamics_yaml
  local run=$1 tyaml=$2 dyaml=$3
  train_stage $run video_tokenizer scripts/tokenizer/train_video_tokenizer.py $tyaml $CFG/tokenizer.yaml
  train_stage $run dynamics scripts/dynamics/train_dynamics.py $tyaml $dyaml video_tokenizer_path=$(latest $run video_tokenizer)
}

# 4. training
for run in $RUNS; do
  case $run in
    point_maze_flow) tok_and_dyn $run $CFG/training_point_maze.yaml $CFG/dynamics_real_actions.yaml ;;
    wall_flow)       tok_and_dyn $run $CFG/training_wall.yaml $CFG/dynamics_real_actions.yaml ;;
    obj3d_como_actions)
      train_stage $run video_tokenizer scripts/tokenizer/train_video_tokenizer.py $CFG/training_obj3d_como.yaml $CFG/tokenizer.yaml
      tok=$(latest $run video_tokenizer)
      como=results/$run/como/checkpoints/como_step_$COMO_STEPS
      if [ -f "$como/state.pt" ]; then echo "=== $run/como done ($como)"; else
        echo "=== train $run/como (tokenizer $tok)"
        $PY scripts/actions/train_como.py --config configs/como/zelda.yaml,configs/como/obj3d.yaml -- \
          tokenizer_path=$tok n_updates=$COMO_STEPS ckpt_interval=$((COMO_STEPS < 5000 ? COMO_STEPS : 5000)) checkpoint=auto device=$DEVICE $COMO_EXTRA
      fi
      if [ ! -f results/$run/como_z_full/state.pt ] || [ ! -f results/$run/obj3d_train_como_z.npy ]; then
        $PY scripts/actions/como_actions.py --como-ckpt "$como" --out-dir results/$run/como_z --z-dir results/$run --live --k 0 --gap 1 \
          --start-index 0 --splits obj3d_train,obj3d_test --fit-split obj3d_train --device $DEVICE ${ACTIONS_LIMIT:+--limit $ACTIONS_LIMIT}
      fi
      train_stage $run dynamics scripts/dynamics/train_dynamics.py $CFG/training_obj3d_como.yaml $CFG/dynamics_como.yaml \
        video_tokenizer_path=$tok latent_actions_path=results/$run/como_z_full action_file=results/$run/obj3d_train_como_z.npy ;;
    *) echo "unknown run $run"; exit 1 ;;
  esac
done

# 5. evals (1 Euler step per frame, the Zelda flow-matching default; skipped when the score file exists)
for run in $RUNS; do
  case $run in
    point_maze_flow|wall_flow)
      env=${run%_flow}
      npz=results/nanowm_${env}_rescore/predictions_f16.npz
      [ -f "$EVALS/$run.json" ] || $PY scripts/eval/eval_nanowm.py --env $env --run-dir results/$run --num-steps 1 \
        --device "$DEVICE" --out-dir "$EVALS" --name $run $([ -f "$npz" ] && echo "--nanowm-npz $npz") $(lim) ;;
    obj3d_como_actions)
      [ -f "$EVALS/$run.json" ] || $PY scripts/eval/eval_lpwm.py --video-tokenizer-path "$(latest $run video_tokenizer)" \
        --dynamics-path "$(latest $run dynamics)" --latent-actions-path results/$run/como_z_full --num-steps 1 \
        --device "$DEVICE" --out-dir "$EVALS" --name $run $(lim_lpwm) ;;
  esac
done

# 6. bundle everything to send back (no weights)
files=$(ls -d "$EVALS" results/*/video_tokenizer/visualizations results/*/dynamics/visualizations results/*/como/log.jsonl \
  results/*/como/visualizations results/nanowm_*_rescore/metrics.json $CFG configs/como/obj3d.yaml 2>/dev/null || true)
tar czf benchmark_results.tgz $files
echo "DONE -> benchmark_results.tgz"
