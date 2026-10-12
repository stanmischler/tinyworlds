"""Smoke test of experiments/benchmarks/run_local.sh on one Modal GPU (the real runs are done on a teammate's GPUs).

    modal run experiments/benchmarks/modal_benchmarks.py::prepare              # data into the data volume (CPU, ~1 h)
    TINYWORLDS_GPU=H100 modal run --detach experiments/benchmarks/modal_benchmarks.py::smoke

smoke: the real script end to end on CUDA with short stages (STEPS training steps, evals on LIMIT clips), including
NanoWM's re-scoring with their own code, then the same command again to check that every stage is skipped/resumed.
Run dirs are prefixed smoke_ on the results volume (smoke_results/), so real results are never shadowed.
"""

import subprocess

from scripts.infra.modal_train import GPU, REPO_DIR, VOLUMES, app, committing


@app.function(cpu=16, memory=65536, volumes=VOLUMES, timeout=4 * 60 * 60)
def prepare():
    with committing():
        for cmd in (["python", "scripts/data/convert_dinowm.py", "--env", "point_maze"],
                    ["python", "scripts/data/convert_dinowm.py", "--env", "wall"],
                    ["python", "scripts/data/convert_obj3d.py"]):
            subprocess.run(cmd, cwd=REPO_DIR, check=True)


@app.function(gpu=GPU, cpu=16, memory=98304, volumes=VOLUMES, timeout=4 * 60 * 60)
def smoke(steps: int = 200, log: int = 100, como_steps: int = 100, limit: int = 16, nanowm: bool = True):
    # a scratch copy of the repo whose results/ is a fresh dir inside the results volume: smoke run dirs never mix with
    # real ones, and the data volume is shared
    work = "/tmp/bench"
    subprocess.run(f"rm -rf {work} && mkdir -p {work} && cp -r {REPO_DIR}/. {work}/ 2>/dev/null; rm -rf {work}/results {work}/data && "
                   f"mkdir -p {REPO_DIR}/results/smoke_results && ln -s {REPO_DIR}/results/smoke_results {work}/results && "
                   f"ln -s {REPO_DIR}/data {work}/data", shell=True, check=True)
    env = (f"PY=python STEPS={steps} LOG={log} COMO_STEPS={como_steps} LIMIT={limit} "
           f"COMO_EXTRA='use_wandb=false' {'' if nanowm else 'SKIP_NANOWM=1'}")
    with committing():
        for attempt in (1, 2):  # the second call must skip every finished stage
            r = subprocess.run(f"{env} bash experiments/benchmarks/run_local.sh", shell=True, cwd=work)
            print(f"SMOKE attempt {attempt}: exit {r.returncode}", flush=True)
            if r.returncode:
                raise SystemExit(r.returncode)
    print("SMOKE OK", flush=True)
