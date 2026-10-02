"""Block (up to --max-wait seconds) until the given Modal apps have stopped, then print each app's state and the
checkpoints its run wrote. For agents, whose shell calls time out after 10 min: call it again until it prints DONE.

    ./.venv/bin/python scripts/modal_wait.py ap-xxx=lamloop_i1_a ap-yyy=lamloop_i1_b [--max-wait 540]

Each argument is <app id>=<run name> (the --run-name given to scripts/modal_train.py); the run name is used only to
list results/<run name>/*/checkpoints on the tinyworlds-results volume.
"""
import argparse
import json
import subprocess
import time

MODAL = "./.venv/bin/modal"

p = argparse.ArgumentParser()
p.add_argument("apps", nargs="+")
p.add_argument("--max-wait", type=int, default=540)
p.add_argument("--poll", type=int, default=30)
args = p.parse_args()
apps = dict(a.split("=", 1) for a in args.apps)


def states():
    out = subprocess.run([MODAL, "app", "list", "--json"], capture_output=True, text=True).stdout
    listed = {a["app_id"]: a["state"] for a in (json.loads(out) if out.strip() else [])}
    return {a: listed.get(a, "not listed") for a in apps}


t0 = time.time()
while True:
    st = states()
    alive = [a for a, s in st.items() if s != "stopped"]  # "not listed" = typo'd id or not registered yet: keep waiting
    if not alive or time.time() - t0 > args.max_wait:
        break
    time.sleep(args.poll)

for app, run in apps.items():
    ls = subprocess.run([MODAL, "volume", "ls", "tinyworlds-results", f"{run}/latent_actions/checkpoints"],
                        capture_output=True, text=True)
    ckpts = sorted((l.strip() for l in ls.stdout.splitlines() if "_step_" in l), key=lambda s: int(s.rsplit("_", 1)[1]))
    print(f"{app} {run}: {st[app]}; checkpoints: {ckpts[-3:] if ckpts else 'none'}")
if any(s == "not listed" for s in st.values()):
    print("WARNING: some app ids are not in `modal app list` (check the id)")
print("DONE" if not alive else f"WAITING ({len(alive)} alive after {int(time.time() - t0)} s)")
