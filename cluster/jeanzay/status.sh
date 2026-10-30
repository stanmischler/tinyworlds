#!/usr/bin/env bash
# What is happening on Jean Zay, in one screen. Run from the Mac:
#   bash cluster/jeanzay/status.sh               # all runs
#   bash cluster/jeanzay/status.sh <folder>      # one synced run folder
# Shows the queue, the jobs of the last 2 days with their state, the last lines of each recent
# log, and the project's GPU-hours.
set -euo pipefail

ssh jeanzay "bash -lc '
echo \"== queue\"
squeue -u \$USER -o \"%.10i %.24j %.8T %.10M %.10l %R\"
echo \"== last 2 days\"
sacct -X -S now-2days -o JobID%-14,JobName%-24,State%-11,Elapsed,ExitCode
for f in \$(ls -t \$WORK/runs/${1:-*}/logs/*.out 2>/dev/null | head -6); do
    echo \"== \${f#\$WORK/runs/}\"
    tail -n 3 \"\$f\"
done
echo \"== GPU-hours\"
idracct | grep -E \"Allocation|Totaux|Utilisation\"
'"
