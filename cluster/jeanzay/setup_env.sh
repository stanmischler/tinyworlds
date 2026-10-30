#!/usr/bin/env bash
# One-off setup on Jean Zay, run on the login node from a synced run folder:
#   ssh jeanzay 'bash -lc "cd \$WORK/runs/<folder> && bash cluster/jeanzay/setup_env.sh"'
# Builds one venv per GPU type, $WORK/envs/tinyworlds-{a100,h100}, on top of IDRIS's PyTorch module, and
# downloads the pretrained weights the code fetches at run time into the shared caches under $WORK. Safe to
# re-run. GPU nodes have no internet, so everything a run needs must be fetched here first.
set -euo pipefail
cd "$(dirname "$0")/../.."

export HF_HOME="$WORK/hf_home" TORCH_HOME="$WORK/torch_home"  # same caches as train.slurm

# A100 and H100 nodes have separate software stacks (arch/a100, arch/h100): a venv built on one
# does not import torch on the other.
for ARCH in a100 h100; do
    VENV="$WORK/envs/tinyworlds-$ARCH"
    module purge
    module load arch/$ARCH pytorch-gpu/py3/2.8.0
    # The module already provides torch, torchvision, numpy, h5py, matplotlib, ... The venv sees them
    # (--system-site-packages) and adds the rest; the constraint stops pip from replacing the module's torch.
    # The version is read from metadata: the H100 stack's torch cannot be imported on the login node.
    [ -d "$VENV" ] || python -m venv --system-site-packages "$VENV"
    "$VENV/bin/python" -m pip install --quiet --no-cache-dir -r requirements.txt \
        -c <("$VENV/bin/python" -c "from importlib.metadata import version; print('torch==' + version('torch'))")
    echo "venv ready: $VENV"
done

# Weights fetched at run time, with the A100 venv (its torch imports on the login node):
#   facebook/vit-mae-large: frozen MAE encoder of the CoMo action model (models/como.py)
#   AlexNet + LPIPS heads, FID Inception: image metrics of the evals
module purge && module load arch/a100 pytorch-gpu/py3/2.8.0
"$WORK/envs/tinyworlds-a100/bin/python" - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download('facebook/vit-mae-large')
import lpips
lpips.LPIPS(net='alex')
from pytorch_fid.inception import fid_inception_v3
fid_inception_v3()
print('weights cached')
EOF
