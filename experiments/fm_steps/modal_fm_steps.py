"""experiments/fm_steps/diagnose_steps.py on the longer (13M) and bigger (60M) flow-matching Zelda models, one GPU call each (TINYWORLDS_GPU, default A100-80GB).

    modal run experiments/fm_steps/modal_fm_steps.py::run --models longer,bigger [--extra "--n-windows 4 --n-seeds 2"]
    modal volume get tinyworlds-results evals/fm_steps/ eval_results/
"""

import shlex
import subprocess

from scripts.infra.modal_train import GPU, REPO_DIR, VOLUMES, app, committing

TOKENIZER = "results/zelda_cont_final_tp/video_tokenizer/checkpoints/video_tokenizer_step_29000"
ACTIONS = "results/como_actions_v1_full"
DYNAMICS = {"longer": "results/zelda_longer_fm/dynamics/checkpoints/dynamics_step_99999",
            "bigger": "results/zelda_bigger_fm/dynamics/checkpoints/dynamics_step_99999"}


@app.function(gpu=GPU, cpu=4, memory=16384, volumes=VOLUMES, timeout=2 * 60 * 60)
def diagnose(model: str, extra: str = ""):
    with committing():
        subprocess.run(["python", "experiments/fm_steps/diagnose_steps.py", "--device", "cuda", "--test-h5", "data/zelda_test_frames.h5",
                        "--video-tokenizer-path", TOKENIZER, "--latent-actions-path", ACTIONS, "--dynamics-path", DYNAMICS[model],
                        "--out-dir", "results/evals/fm_steps", "--name", f"{model}_fm", *shlex.split(extra)], cwd=REPO_DIR, check=True)


@app.local_entrypoint()
def run(models: str = "longer,bigger", extra: str = ""):
    list(diagnose.starmap([(m, extra) for m in models.split(",")]))
