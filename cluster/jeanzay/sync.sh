#!/usr/bin/env bash
# Copy a working tree to Jean Zay, into $WORK/runs/<folder name>, with its inputs linked in.
#   bash cluster/jeanzay/sync.sh                 # this working tree
#   bash cluster/jeanzay/sync.sh ../other-tree   # another one
#   DATA_FILES="a.h5 b.npy" bash cluster/jeanzay/sync.sh   # other data files to upload (default: the Zelda set)
#   INPUTS=<local dir> bash cluster/jeanzay/sync.sh        # also upload checkpoints laid out as results/ (see below)
# Needs `Host jeanzay` in ~/.ssh/config (see cluster/README.md). Run it again after each change: only
# modified files are sent, and the checkpoints, results and logs of runs on Jean Zay are left alone.
#
# Shared inputs, one copy for every run:
#   $WORK/tinyworlds_data/      data files, linked as the run's data/
#   $WORK/tinyworlds_results/   input checkpoints, each top-level entry linked into the run's results/
#                               (results/ itself stays a real folder: new runs write there)
set -euo pipefail

SRC="$(cd "${1:-$(dirname "$0")/../..}" && pwd)"
NAME="$(basename "$SRC")"
WORK="$(ssh jeanzay 'bash -lc "echo \$WORK"')"  # $WORK only exists in a login shell
RUN="$WORK/runs/$NAME"
DATA="$WORK/tinyworlds_data"
INPUT_RESULTS="$WORK/tinyworlds_results"
DATA_FILES="${DATA_FILES:-zelda_train_frames.h5 zelda_test_frames.h5 zelda_train_como_z.npy}"

ssh jeanzay "mkdir -p '$RUN/logs' '$RUN/results' '$DATA' '$INPUT_RESULTS'"

# data files: -L follows the local data/ symlink; files already on Jean Zay are skipped (inputs never change in
# place: a regenerated file gets a new name)
(cd "$SRC/data" && rsync -rltzL --ignore-existing --progress $DATA_FILES "jeanzay:$DATA/")
[ -z "${INPUTS:-}" ] || rsync -rltz --progress "$INPUTS/" "jeanzay:$INPUT_RESULTS/"

rsync -rltz --delete \
    --exclude .git --exclude .venv --exclude __pycache__ --exclude .DS_Store --exclude .claude \
    --exclude /data --exclude /results --exclude /wandb --exclude /logs --exclude /inference_results \
    "$SRC/" "jeanzay:$RUN/"

# link data/ and every shared input checkpoint not already in the run's results/
ssh jeanzay "cd '$RUN' && { [ -L data ] || { rm -rf data && ln -s '$DATA' data; }; } \
    && for d in '$INPUT_RESULTS'/*; do [ -e \"\$d\" ] && { [ -e \"results/\${d##*/}\" ] || ln -s \"\$d\" results/; }; done; true"

echo "synced to jeanzay:$RUN"
