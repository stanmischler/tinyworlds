"""STA-76 GPU stress test of experiments/drawing/run_local.sh on one Modal GPU (we do not run the 50k runs; Lucas does).

Per arm (p | f1):
  1. memory probes: 20 training steps + a held-out pass at each grad_accum in `accums` -> peak GPU memory and it/s
     (an OOM is recorded, not fatal)
  2. the real script, end to end on CUDA, with a resume: run_local.sh at STEPS=150, then again at STEPS=300 (continues
     from the step-150 checkpoint), then the evals on the step-300 checkpoint and the tarball.
Outputs on the results volume: stress_<arm>/ and stress_probe_<arm>_a<k>/; the summary is printed at the end.

    TINYWORLDS_GPU=H100 modal run --detach experiments/drawing/modal_drawing.py::stress --arm p
    TINYWORLDS_GPU=H100 modal run --detach experiments/drawing/modal_drawing.py::stress --arm p --accums 16 --no-script  # probe only
"""

import json
import subprocess

from scripts.infra.modal_train import GPU, REPO_DIR, VOLUMES, app, committing

CONFIG = "configs/como/zelda.yaml,configs/como/drawing/base.yaml,configs/como/drawing/{arm}.yaml"


@app.function(gpu=GPU, cpu=8, memory=65536, volumes=VOLUMES, timeout=3 * 60 * 60)
def stress(arm: str, accums: str = "1,2,4,8", script: bool = True):
    summary = {}
    with committing():
        for acc in [int(a) for a in accums.split(",")]:
            run = f"stress_probe_{arm}_a{acc}"
            r = subprocess.run(["python", "scripts/actions/train_como.py", "--config", CONFIG.format(arm=arm), "--",
                                f"run_name={run}", f"grad_accum={acc}", "n_updates=20", "log_interval=10", "val_interval=10",
                                "ckpt_interval=100000", "judge=false", "checkpoint=null"],
                               cwd=REPO_DIR, capture_output=True, text=True)
            recs = [json.loads(l) for l in open(f"{REPO_DIR}/results/{run}/como/log.jsonl")] if r.returncode == 0 else []
            oom = "OutOfMemoryError" in r.stderr or "out of memory" in r.stderr
            summary[f"a{acc}"] = {"ok": r.returncode == 0, "oom": oom, "chunk": 256 // acc,
                                  "mem_gb": round(max((x.get("mem_gb", 0) for x in recs), default=0), 1),
                                  "it_s": round(recs[-1]["it_s"], 3) if recs else None}
            if r.returncode and not oom:
                print(r.stderr[-3000:], flush=True)
            print("PROBE", arm, summary[f"a{acc}"], flush=True)
        env = f"RUN_PREFIX=stress GRAD_ACCUM=4 CKPT=150 EVAL_STEPS=300 PY=python"
        for steps in ((150, 300) if script else ()):  # second call resumes from the step-150 checkpoint
            r = subprocess.run(f"{env} STEPS={steps} bash experiments/drawing/run_local.sh {arm}", shell=True, cwd=REPO_DIR)
            summary[f"script_{steps}"] = r.returncode
    print("SUMMARY", arm, json.dumps(summary), flush=True)
    return summary
