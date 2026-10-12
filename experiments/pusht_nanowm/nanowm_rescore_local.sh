#!/usr/bin/env bash
# Re-run NanoWM's published DINO-WM evaluation on a plain Linux GPU box (no Modal) and dump its predictions in our
# format: the same thing as nanowm_rescore_modal.py (both drive experiments/pusht_nanowm/nanowm_dump.py).
#
# Usage (from the tinyworlds repo root; needs git, curl, an NVIDIA GPU + driver; ~15-30 min on one GPU):
#   bash experiments/pusht_nanowm/nanowm_rescore_local.sh <pusht|point_maze|wall> <out_dir>
# Env overrides: NANOWM_HOME (default ~/.cache/tinyworlds-nanowm: their repo + venv + scratch),
#                DATASET_DIR (default $NANOWM_HOME/data: raw DINO-WM folders, downloaded if missing),
#                CUDA_VISIBLE_DEVICES (the first listed GPU is used), NO_PROBE=1 (skip predictions_f16.npz / probe.json).
# Then score them with ours (copy predictions_f16.npz back to the eval machine):
#   python scripts/eval/eval_nanowm.py --env <env> --run-dir results/<run> --nanowm-npz <out_dir>/predictions_f16.npz
set -euo pipefail

ENV_NAME=${1:?usage: nanowm_rescore_local.sh <pusht|point_maze|wall> <out_dir>}
OUT_DIR=${2:?usage: nanowm_rescore_local.sh <pusht|point_maze|wall> <out_dir>}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NANOWM_HOME=${NANOWM_HOME:-$HOME/.cache/tinyworlds-nanowm}
DATASET_DIR=${DATASET_DIR:-$NANOWM_HOME/data}
NANOWM_DIR="$NANOWM_HOME/nano-world-model"
NANOWM_VENV="$NANOWM_HOME/venv"
COMMIT=$(sed -n 's/^NANOWM_COMMIT = "\(.*\)"/\1/p' "$REPO_ROOT/experiments/pusht_nanowm/nanowm_dump.py")
NANOWM_REPO=$(sed -n 's/^NANOWM_REPO = "\(.*\)"/\1/p' "$REPO_ROOT/experiments/pusht_nanowm/nanowm_dump.py")
mkdir -p "$NANOWM_HOME" "$DATASET_DIR" "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd); DATASET_DIR=$(cd "$DATASET_DIR" && pwd)  # absolute: we cd below

# 1. their code at the pinned commit
if [ ! -d "$NANOWM_DIR/.git" ]; then
    git clone "$NANOWM_REPO" "$NANOWM_DIR"
fi
git -C "$NANOWM_DIR" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$NANOWM_DIR" checkout -q "$COMMIT"
rm -f "$NANOWM_DIR/src/main_probe.py"  # written by the driver; keeps the checkout clean between runs

# 2. their locked environment (python 3.11, torch 2.7.1+cu128, lightning 1.9.5, piqa, lpips, pytorch-fid ...);
#    --frozen installs uv.lock as-is: at this commit `--locked` reports the lock as stale vs pyproject
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/0.12.21/install.sh | sh  # the version the Modal image pins
    export PATH="$HOME/.local/bin:$PATH"
fi
(cd "$NANOWM_DIR" && UV_PROJECT_ENVIRONMENT="$NANOWM_VENV" uv sync --frozen --no-dev --managed-python)
"$NANOWM_VENV/bin/python" -c 'import torch, pytorch_lightning, piqa, lpips, pytorch_fid, decord; print(torch.__version__, pytorch_lightning.__version__, torch.cuda.is_available())'

# 3. data + checkpoint + their eval + dump (their venv python; -m from the repo root finds our driver module)
PROBE_FLAG=""
[ "${NO_PROBE:-0}" = "1" ] && PROBE_FLAG="--no-probe"
cd "$REPO_ROOT"
HF_HUB_ENABLE_HF_TRANSFER=0 WANDB_MODE=disabled \
    "$NANOWM_VENV/bin/python" -m experiments.pusht_nanowm.nanowm_dump \
    --env "$ENV_NAME" --nanowm-dir "$NANOWM_DIR" --py "$NANOWM_VENV/bin/python" \
    --dataset-dir "$DATASET_DIR" --work "$NANOWM_HOME/work_$ENV_NAME" --out "$OUT_DIR" $PROBE_FLAG
echo "done: $OUT_DIR (metrics.json, predictions.npz, predictions_f16.npz)"
