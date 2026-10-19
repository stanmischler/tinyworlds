"""Smoke test of experiments/focal_frequency/run_local.sh on one Modal GPU (the real runs are done on a teammate's GPU).

    TINYWORLDS_GPU=H100 modal run experiments/focal_frequency/modal_focal_frequency.py::smoke

The real script end to end on CUDA with short stages (f-only run F_STOP steps at batch 64, focal run STEPS steps, evals on
LIMIT windows), then the same command again to check that every stage is skipped. The frozen weights are linked from the
results volume instead of downloaded (ASSETS_REPO empty); run dirs live in a scratch results/ (smoke_focal_frequency/ on
the volume), so real results are never shadowed. GPU memory is sampled every 2 s into gpu_memory.csv there.
"""

import subprocess

from scripts.infra.modal_train import GPU, REPO_DIR, VOLUMES, app, committing

ASSETS = ["zelda_cont_final_tp", "como_actions_v1_full", "como_zelda_v1", "zelda_longer_fm"]


@app.function(gpu=GPU, cpu=8, memory=65536, volumes=VOLUMES, timeout=2 * 60 * 60)
def smoke(steps: int = 40, f_stop: int = 20, log: int = 10, limit: int = 16):
    work, out = "/tmp/focal", f"{REPO_DIR}/results/smoke_focal_frequency"
    links = " && ".join(f"ln -s {REPO_DIR}/results/{a} {out}/{a}" for a in ASSETS)
    subprocess.run(f"rm -rf {work} {out} && mkdir -p {work} {out} && tar -C {REPO_DIR} --exclude=./data --exclude=./results -cf - . "
                   f"| tar -C {work} -xf - && ln -s {out} {work}/results && ln -s {REPO_DIR}/data {work}/data && {links}",
                   shell=True, check=True)
    mem = subprocess.Popen(f"nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 2 > {out}/gpu_memory.csv", shell=True)
    env = f"PY=python ASSETS_REPO= STEPS={steps} F_STOP={f_stop} LOG={log} TEST_EVERY=1000 LIMIT={limit} SMOKE=1"
    with committing():
        for attempt in (1, 2):  # the second call must skip every finished stage
            r = subprocess.run(f"{env} bash experiments/focal_frequency/run_local.sh", shell=True, cwd=work)
            print(f"SMOKE attempt {attempt}: exit {r.returncode}", flush=True)
            if r.returncode:
                raise SystemExit(r.returncode)
        mem.terminate()
        subprocess.run(f"cp {work}/focal_frequency_results.tgz {out}/", shell=True)
    print("SMOKE OK", flush=True)
