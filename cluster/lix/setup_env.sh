#!/usr/bin/env bash
# One-off setup for the action-model run on a cluster with internet on the node where this is run. From the repo root:
#   bash cluster/lix/setup_env.sh                        downloads the data files missing from data/ (Hugging Face dataset with approved access: hf auth login first)
#   DATA_SRC=/path/to/the/7/files bash cluster/lix/setup_env.sh     links the data files from another folder into data/ instead
#   VENV=/path/to/your/env bash cluster/lix/setup_env.sh  use an environment you built (e.g. on the cluster's PyTorch): nothing is installed in it
# Without VENV it builds .venv (PYTHON=python3.12 to choose the interpreter) with the package versions the run was developed on
# (cluster/lix/constraints.txt). Then it caches the weights of the LPIPS score, checks the data files against cluster/lix/data.sha256
# and checks that the run's config loads. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}" PYTHONPATH="$PWD"

if [ -z "${VENV:-}" ]; then
    VENV="$PWD/.venv"
    [ -d "$VENV" ] || "${PYTHON:-python3}" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    "$VENV/bin/python" -m pip install --quiet -r requirements.txt -c cluster/lix/constraints.txt
fi
echo "environment ready: $VENV ($("$VENV/bin/python" -c 'import torch; print("torch", torch.__version__, "cuda build", torch.version.cuda)'))"

# the Link-box LPIPS score of each checkpoint reads AlexNet + the LPIPS heads; GPU nodes often have no internet, so fetch them now
"$VENV/bin/python" -c "import os, lpips; lpips.LPIPS(net='alex', verbose=False); print('LPIPS weights cached in', os.environ['TORCH_HOME'])"

mkdir -p data results logs  # logs/: Slurm does not create the folder of its output file
if [ -n "${DATA_SRC:-}" ]; then
    while read -r f; do [ -e "data/$f" ] || ln -sfn "$(cd "$DATA_SRC" && pwd)/$f" "data/$f"; done < cluster/lix/data_files.txt
fi
# files still missing come from the Hugging Face dataset (log in first: hf auth login, with an account approved on the dataset page)
HF_DATA="${HF_DATA:-stanmischler/zelda-action-model-data}"
"$VENV/bin/python" - "$HF_DATA" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
for name in (line.strip() for line in open('cluster/lix/data_files.txt')):
    if name and not os.path.exists(f'data/{name}'):
        print('downloading', name, 'from', sys.argv[1], flush=True)
        hf_hub_download(sys.argv[1], name, repo_type='dataset', local_dir='data')
PY
missing=0
while read -r f; do [ -e "data/$f" ] || { echo "missing data/$f"; missing=1; }; done < cluster/lix/data_files.txt
[ "$missing" = 0 ] || { echo "the data files are not all in data/ (see cluster/lix/README.md)"; exit 1; }
echo "checking the data files (13 GB, a few minutes)"
if command -v sha256sum > /dev/null; then SHA=(sha256sum); else SHA=(shasum -a 256); fi
(cd data && "${SHA[@]}" -c "$ROOT/cluster/lix/data.sha256")  # data/ may be a link to another disk, so no ../

"$VENV/bin/python" cluster/lix/check_setup.py
echo "setup done. Launch: bash cluster/lix/run.sh   (or: sbatch cluster/lix/run.sh)"
