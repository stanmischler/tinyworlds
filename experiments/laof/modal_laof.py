"""Modal functions of the LAOF experiment (optical-flow-supervised latent actions): RAFT flow targets, then LAOF training.
Shares the tinyworlds app, image and volumes of scripts/infra/modal_train.py:
    modal run experiments/laof/modal_laof.py::laof_flow --split test --limit 512"""

import os

from scripts.infra.modal_train import GPU, REPO_DIR, SECRETS, VOLUMES, app, committing, data_volume, results_volume


@app.function(gpu=GPU, cpu=4, memory=32768, volumes=VOLUMES, timeout=6 * 60 * 60)
def laof_flow(split: str = "test", limit: int = 0, batch: int = 32):
    """LAOF flow targets (experiments/laof/laof_flow.py) for data/zelda_<split>_frames.h5 -> data/zelda_<split>_flow_gap4.h5 in the
    data volume (+ a viz PNG in the results volume under laof/). limit > 0 writes a *_smoke file instead.
        modal run experiments/laof/modal_laof.py::laof_flow --split test --limit 512
    """
    import subprocess

    tag = "_smoke" if limit else ""
    os.makedirs(f"{REPO_DIR}/results/laof", exist_ok=True)
    subprocess.run(
        ["python", "experiments/laof/laof_flow.py", "--h5", f"data/zelda_{split}_frames.h5", "--out", f"data/zelda_{split}_flow_gap4{tag}.h5",
         "--batch", str(batch), "--limit", str(limit), "--viz", f"results/laof/flow_{split}{tag}.png"],
        cwd=REPO_DIR,
        check=True,
    )
    data_volume.commit()
    results_volume.commit()


@app.function(gpu=GPU, cpu=4, memory=32768, volumes=VOLUMES, secrets=SECRETS, timeout=24 * 60 * 60)
def train_laof(config: str, overrides: str = "", run_name: str = ""):
    """LAOF latent action model (experiments/laof/train_laof.py); checkpoints, viz and judge.jsonl in results/<run_name>/latent_actions/.
        TINYWORLDS_GPU=H100 modal run --detach experiments/laof/modal_laof.py::train_laof --config configs/experiments/laof/discrete.yaml --run-name laof_discrete
    """
    import subprocess

    with committing():
        subprocess.run(
            ["python", "experiments/laof/train_laof.py", "--config", config, f"run_name={run_name}", *[o for o in overrides.split(",") if o]],
            cwd=REPO_DIR,
            check=True,
        )