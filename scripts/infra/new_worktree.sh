#!/usr/bin/env bash
# Start a new line of work in its own worktree, branched from main.
#
#   scripts/infra/new_worktree.sh <name> [base]
#
# Creates ../worktrees/<name> on a new branch <name> (from `base`, default: main),
# and wires it up: shared .venv and data/ (symlinks), the CPU smoke config (copy),
# and CLAUDE.md symlinked to the main checkout's copy. results/ stays per-worktree
# on purpose so find_latest_checkpoint only sees that experiment's checkpoints.
set -euo pipefail

name="${1:?usage: scripts/infra/new_worktree.sh <name> [base]}"
base="${2:-main}"

main_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && git rev-parse --show-toplevel)"
wt_root="$(dirname "$main_repo")/worktrees"
wt_dir="$wt_root/$name"

mkdir -p "$wt_root"
git -C "$main_repo" worktree add "$wt_dir" -b "$name" "$base"

ln -s "$main_repo/.venv" "$wt_dir/.venv"
ln -s "$main_repo/data" "$wt_dir/data"

# CLAUDE.md is tracked, but every worktree should read the main checkout's copy so an
# edit there is visible everywhere at once. Swap the checked-out file for a symlink and
# tell git to ignore that change here, so it never shows as modified or gets committed.
rm "$wt_dir/CLAUDE.md"
ln -s "$main_repo/CLAUDE.md" "$wt_dir/CLAUDE.md"
git -C "$wt_dir" update-index --skip-worktree CLAUDE.md
[ -f "$main_repo/configs/dev/dev_training_cpu.yaml" ] && cp "$main_repo/configs/dev/dev_training_cpu.yaml" "$wt_dir/configs/dev/"

cat <<EOF

Worktree ready: $wt_dir  (branch: $name, from $base)

  cd "$wt_dir" && export PYTHONPATH="\$PWD"

When done:  git push -u origin $name   then merge/PR into main, then
            git -C "$main_repo" worktree remove --force "$wt_dir"   # --force: the symlinks count as untracked
EOF
