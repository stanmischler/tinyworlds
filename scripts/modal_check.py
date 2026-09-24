"""One-shot health check for a Modal training run: app state + latest W&B step/loss per stage.

    ./.venv/bin/python scripts/modal_check.py <app_id> [--runs 3] [--project tinyworlds] [--entity <wandb entity>]

Prints one MODAL line (state, task count, stop time) and one WANDB line per recent run
(state, last step, losses, heartbeat). Uses run *summaries* (not history) so the step is the
real last step, not a subsample. Used by the /run-experiment skill's babysit loop.
"""
import argparse
import json
import subprocess

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("app_id", nargs="?", default=None)
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--project", default="tinyworlds")
parser.add_argument("--entity", default=None, help="defaults to the W&B default entity of the logged-in user")
args = parser.parse_args()

out = subprocess.run(["./.venv/bin/modal", "app", "list", "--json"], capture_output=True, text=True)
apps = json.loads(out.stdout) if out.stdout.strip() else []
mine = [a for a in apps if a.get("description") == args.project and (args.app_id is None or a.get("app_id") == args.app_id)]
for a in mine:
    print("MODAL", a["app_id"], a["state"], "tasks", a["tasks"], "created", a["created_at"], "stopped", a["stopped_at"])
if not mine:
    print("MODAL: app not listed (stopped some time ago, or never started)")

api = wandb.Api()
entity = args.entity or api.default_entity
for r in list(api.runs(f"{entity}/{args.project}", order="-created_at", per_page=args.runs))[: args.runs]:
    summ = r.summary
    loss = {k: round(summ[k], 4) for k in summ.keys() if "loss" in k and isinstance(summ[k], (int, float))}
    print("WANDB", r.name, r.state, "step", summ.get("_step"), loss, "heartbeat", r.heartbeat_at)
