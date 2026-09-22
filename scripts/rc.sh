#!/usr/bin/env bash
# Start a phone-controllable Claude Code server for a worktree (or the main checkout).
#
#   scripts/rc.sh <worktree-name>   # serves ../worktrees/<name>
#   scripts/rc.sh main              # serves the main checkout
#   scripts/rc.sh --list            # show running servers
#   scripts/rc.sh --stop <name>     # stop one
#
# Each server runs detached in its own tmux session (rc-<name>) wrapped in
# `caffeinate -i`, so it survives closing the terminal and keeps the Mac from
# idle-sleeping. Connect from the phone: Claude app → Code → session
# "tinyworlds/<name>", or `tmux attach -t rc-<name>` and press space for a QR code.
set -euo pipefail

main_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && git rev-parse --show-toplevel)"
wt_root="$(dirname "$main_repo")/worktrees"

case "${1:-}" in
  --list)
    tmux ls 2>/dev/null | grep '^rc-' || echo "no rc servers running"
    exit 0 ;;
  --stop)
    tmux kill-session -t "rc-${2:?usage: scripts/rc.sh --stop <name>}"
    echo "stopped rc-$2"
    exit 0 ;;
  "")
    echo "usage: scripts/rc.sh <worktree-name>|main | --list | --stop <name>" >&2
    exit 1 ;;
esac

name="$1"
if [ "$name" = "main" ]; then dir="$main_repo"; else dir="$wt_root/$name"; fi
[ -d "$dir" ] || { echo "no such worktree: $dir (create it with scripts/new_worktree.sh $name)" >&2; exit 1; }

if tmux has-session -t "rc-$name" 2>/dev/null; then
  echo "rc-$name is already running; attach with: tmux attach -t rc-$name"
  exit 0
fi

# PYTHONPATH must point at the served directory so the agent's commands work unmodified.
tmux new-session -d -s "rc-$name" -c "$dir" \
  "export PYTHONPATH='$dir'; caffeinate -i claude remote-control --name 'tinyworlds/$name'"

cat <<EOF
Started rc-$name serving $dir

  phone:   Claude app → Code → "tinyworlds/$name"
  QR code: tmux attach -t rc-$name   (press space; Ctrl-b d to detach)
  stop:    scripts/rc.sh --stop $name
EOF
