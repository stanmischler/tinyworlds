"""Small action model runs on Modal: frames and shift tables on the data volume, outputs on the results volume.

    modal run experiments/small_action/modal_small_action.py::shift_table --split zelda_train
    modal volume put tinyworlds-data data/zelda_test_shifts.npz zelda_test_shifts.npz
    modal run experiments/small_action/modal_small_action.py::five_hours_data
    TINYWORLDS_GPU=H100 modal run --detach experiments/small_action/modal_small_action.py::train_small_action --config five_hours_plain
    (then long_late_lpips, ours_heavy_long, late_weight_long, late_weight_focus: each starts from the previous one's checkpoint, init_from)
    TINYWORLDS_GPU=H100 modal run --detach experiments/small_action/modal_small_action.py::train_small_action --config simple_recipe --overrides "run_name=...,error_weight=1.0"
    (simple_recipe.yaml: the one-stage recipe and its knobs: action_dim, still_fraction, contrastive_weight, error_weight, window_fraction)
"""

import os
import subprocess

from scripts.infra.modal_train import GPU, REPO_DIR, SECRETS, VOLUMES, app, committing, data_volume, results_volume

CONFIGS = 'configs/experiments/small_action'


@app.function(gpu=GPU, cpu=4 if GPU == "L4" else 8, memory=65536, volumes=VOLUMES, secrets=SECRETS, timeout=24 * 60 * 60)  # an L4 machine has too few CPUs for 8: the call waits in the queue
def train_small_action(config: str, overrides: str = ""):
    """config: a yaml name in configs/experiments/small_action, or several joined by commas, merged left to right over base.yaml. overrides: "k=v,k=v" (split on commas, so no list values)."""
    os.chdir(REPO_DIR)
    configs = ",".join(f"{CONFIGS}/{name}.yaml" for name in ["base", *config.split(",")])
    with committing():
        subprocess.run(["python", "experiments/small_action/train.py", "--config", configs, "--",
                        *[o for o in overrides.split(",") if o]], check=True)


@app.function(gpu="L4", cpu=4, memory=16384, volumes=VOLUMES, timeout=2 * 60 * 60)
def shift_table(split: str = "zelda_train"):
    """data/<split>_frames.h5 -> data/<split>_shifts.npz on the data volume."""
    os.chdir(REPO_DIR)
    with committing():
        subprocess.run(["python", "experiments/small_action/shift.py", "--h5", f"data/{split}_frames.h5", "--out", f"data/{split}_shifts.npz",
                        "--device", "cuda", "--batch", "256"], check=True)


@app.function(gpu="L4", cpu=4, memory=16384, volumes=VOLUMES, timeout=30 * 60)
def visualise(ckpt: str, out: str):
    """Images of a checkpoint (experiments/small_action/visualise.py): results/<ckpt> -> results/<out>."""
    os.chdir(REPO_DIR)
    with committing():
        subprocess.run(["python", "experiments/small_action/visualise.py", "--ckpt", f"results/{ckpt}", "--out", f"results/{out}", "--device", "cuda"], check=True)


@app.function(gpu=GPU, cpu=4 if GPU == "L4" else 8, memory=16384, volumes=VOLUMES, timeout=30 * 60)
def probe(ckpts: str, subset: str = "all"):
    """ckpts: comma-separated "run", "run@step" or checkpoint dirs under results/."""
    os.chdir(REPO_DIR)
    for ckpt in ckpts.split(","):
        subprocess.run(["python", "experiments/small_action/probe.py", "--ckpt", f"results/{latest_checkpoint(ckpt)}", "--subset", subset], check=True)


def latest_checkpoint(spec):
    # "run" or "run@step" -> its checkpoint under results/; a path with a slash is taken as it is
    if "/" in spec:
        return spec
    run, _, step = spec.partition("@")
    root = f"{run}/small_action/checkpoints"
    done = [int(d.rsplit("_", 1)[1]) for d in os.listdir(f"results/{root}") if os.path.exists(f"results/{root}/{d}/state.pt")]
    return f"{root}/step_{step or max(done)}"


@app.function(gpu=GPU, cpu=4 if GPU == "L4" else 8, memory=32768, volumes=VOLUMES, timeout=60 * 60)
def assess(models: str, subset: str = "all", out: str = "", como: str = "como_zelda_v1/como/checkpoints/como_step_50000"):
    """models: comma-separated "run", "run@step" or "name=<checkpoint under results/>". Scores and crops in results/<out>,
    by default results/small_action_images/assess_<subset>_<first run>."""
    os.chdir(REPO_DIR)
    named = [m.split("=", 1) if "=" in m else [m, m] for m in models.split(",")]
    specs = [x for name, ck in named for x in ("--model", f"{name}=results/{latest_checkpoint(ck)}")]
    out = out or f"small_action_images/assess_{subset}_{named[0][0].replace('@', '_')}"
    with committing():
        subprocess.run(["python", "experiments/small_action/assess.py", "--out", f"results/{out}", "--como", f"results/{como}", "--subset", subset, *specs], check=True)


@app.function(gpu=GPU, cpu=4 if GPU == "L4" else 8, memory=32768, volumes=VOLUMES, timeout=30 * 60)
def full_action(models: str, out: str = "small_action_images/full_action", subset: str = "dev"):
    """The action eval on the dev (or confirm) half with the measured shift + a (experiments/small_action/full_action.py). models: comma-separated "run@step"."""
    os.chdir(REPO_DIR)
    specs = [x for m in models.split(",") for x in ("--model", f"{m}=results/{latest_checkpoint(m)}")]
    with committing():
        subprocess.run(["python", "experiments/small_action/full_action.py", "--out", f"results/{out}", "--subset", subset, *specs], check=True)


@app.function(gpu=GPU, cpu=4 if GPU == "L4" else 8, memory=32768, volumes=VOLUMES, timeout=30 * 60)
def blind_sheet(model: str, out: str):
    """12 random confirm still-walk pairs, seed 0, of one "run@step" (experiments/small_action/blind_sheet.py) -> results/<out>."""
    os.chdir(REPO_DIR)
    with committing():
        subprocess.run(["python", "experiments/small_action/blind_sheet.py", "--ckpt", f"results/{latest_checkpoint(model)}", "--out", f"results/{out}"], check=True)


@app.function(gpu=GPU, cpu=8, memory=65536, volumes=VOLUMES, timeout=3 * 60 * 60)
def five_hours_data():
    """The 5 h capture (experiments/small_action/five_hours.py): one .h5, shift and motion tables on the data volume, a report on the results volume."""
    os.chdir(REPO_DIR)
    subprocess.run(["python", "experiments/small_action/five_hours.py", "--stage", "merge"], check=True)
    data_volume.commit()
    with committing():
        subprocess.run(["python", "experiments/small_action/five_hours.py", "--stage", "tables"], check=True)
    data_volume.commit()


@app.function(gpu=GPU, cpu=8, memory=65536, volumes=VOLUMES, timeout=2 * 60 * 60)
def positive_shifts():
    """The shifts to the InfoNCE positives on the YouTube rows of the 5 h shift table (experiments/small_action/positive_shifts.py), on the data volume."""
    os.chdir(REPO_DIR)
    subprocess.run(["python", "experiments/small_action/positive_shifts.py"], check=True)
    data_volume.commit()


@app.function(gpu=GPU, cpu=4 if GPU == "L4" else 8, memory=32768, volumes=VOLUMES, timeout=30 * 60)
def action_pca(models: str, out: str = "small_action_images/action_pca"):
    """Per-pair a, s, labels on the dev half for PCA pictures (experiments/small_action/action_pca.py). models: comma-separated "run@step"."""
    os.chdir(REPO_DIR)
    specs = [x for m in models.split(",") for x in ("--model", f"{m.replace('@', '_')}=results/{latest_checkpoint(m)}")]
    with committing():
        subprocess.run(["python", "experiments/small_action/action_pca.py", "--out", f"results/{out}", *specs], check=True)


# ---- downstream flow dynamics on our action (s + a): precompute, train, eval. ACTION = "run@step" of a small action model, TAG = run_step
DYN_CONFIGS = 'configs/experiments/small_action_dynamics'
DYN_TOKENIZER = 'results/zelda_more_data_tokenizer/video_tokenizer/checkpoints/video_tokenizer_step_59000'
DEFAULT_ACTION = 'late_weight_focus@200000'
DEFAULT_EXTRA_SKIP = 2  # YouTube pairs (r, r + 2), as the action model's; the yamls' extra_frame_skip. 1 = the first files (pairs r, r + 1)


def action_tag(action, extra_skip):
    return action.replace('@', '_') + (f'_yt{extra_skip}' if extra_skip != 1 else '')


def action_paths(action, extra_skip=DEFAULT_EXTRA_SKIP):
    # "run@step" -> (action dir, per-row action file) written by experiments/small_action/dynamics_actions.py
    tag = action_tag(action, extra_skip)
    return f'results/small_action_actions_{tag}', f'data/zelda_more_train_small_action_{tag}.npy'


@app.function(gpu=GPU, cpu=8, memory=65536, volumes=VOLUMES, timeout=4 * 60 * 60)
def dynamics_actions(action: str = DEFAULT_ACTION, limit: int = 0, extra_skip: int = DEFAULT_EXTRA_SKIP):
    """s + a of every dynamics training pair and held-out pair (experiments/small_action/dynamics_actions.py) from one action model checkpoint.
    extra_skip: rows of a pair on the YouTube files, = the dynamics yamls' extra_frame_skip."""
    os.chdir(REPO_DIR)
    try:
        subprocess.run(["python", "experiments/small_action/dynamics_actions.py", "--action-ckpt", f"results/{latest_checkpoint(action)}",
                        "--tag", action_tag(action, extra_skip), "--extra-skip", str(extra_skip), "--limit", str(limit)], check=True)
    finally:
        data_volume.commit()
        results_volume.commit()


@app.function(gpu=GPU, cpu=8, memory=49152, volumes=VOLUMES, secrets=SECRETS, timeout=24 * 60 * 60)
def train_dynamics(config: str, action: str = DEFAULT_ACTION, run_name: str = "", overrides: str = ""):
    """config: a dynamics yaml in configs/experiments/small_action_dynamics (dyn_lr3e-4_err1, small_action_dynamics, ...), trained through
    full_train.py with that study's training.yaml; the action dir and action file are those of `action`. overrides: "k=v,k=v" on the
    dynamics yaml. Outputs results/<run_name, default config>/dynamics."""
    from omegaconf import OmegaConf
    os.chdir(REPO_DIR)
    import torch
    dyn = OmegaConf.merge(OmegaConf.load(f'{DYN_CONFIGS}/{config}.yaml'), OmegaConf.from_dotlist([o for o in overrides.split(",") if o]))
    extra_skip = int(dyn.get('extra_frame_skip', 1))  # the loader's YouTube step picks the action files computed on the same pairs
    action_dir, action_file = action_paths(action, extra_skip)
    assert os.path.exists(f'{action_dir}/state.pt') and os.path.exists(action_file), \
        f'run dynamics_actions --action {action} --extra-skip {extra_skip} first'
    assert torch.load(f'{action_dir}/state.pt', map_location='cpu', weights_only=False).get('extra_skip', 1) == extra_skip, action_dir
    dyn.latent_actions_path, dyn.action_file = action_dir, action_file
    derived = f'{DYN_CONFIGS}/derived_dynamics.yaml'
    OmegaConf.save(dyn, derived)
    training = OmegaConf.load(f'{DYN_CONFIGS}/training.yaml')
    training.dynamics_config = derived
    OmegaConf.save(training, f'{DYN_CONFIGS}/derived_training.yaml')
    print(OmegaConf.to_yaml(dyn))
    # GPU memory in the log every 2 min (train_dynamics.py logs it only with extra loss terms)
    subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader", "-l", "120"])
    with committing():
        subprocess.run(["python", "scripts/pipeline/full_train.py", "--config", f'{DYN_CONFIGS}/derived_training.yaml'], check=True,
                       env={**os.environ, "NG_RUN_NAME": run_name or config})


@app.function(gpu=GPU, cpu=8, memory=32768, volumes=VOLUMES, timeout=3 * 60 * 60)
def eval_small_dynamics(run: str, action: str = DEFAULT_ACTION, split: str = "dev", step: int = -1, num_steps: int = 32, limit: int = 0,
                        extra_skip: int = DEFAULT_EXTRA_SKIP):
    """32-step held-out eval (scripts/eval/eval_next_frame.py) of results/<run>'s last (or given) dynamics checkpoint, on the dev or confirm
    half of the 890 windows, then its split scores (experiments/small_action/dynamics_score.py) -> results/evals/next_frame/<run>_<split>_s32.json.
    The held-out windows are our capture at skip 4 whatever extra_skip; extra_skip only picks the action dir (mean/std) the run trained with."""
    import torch
    os.chdir(REPO_DIR)
    root = f'results/{run}/dynamics/checkpoints'
    steps = [int(d.rsplit('_', 1)[1]) for d in os.listdir(root) if os.path.exists(f'{root}/{d}/state.pt')]
    ckpt = f'{root}/dynamics_step_{max(steps) if step < 0 else step}'
    trained = torch.load(f'{ckpt}/state.pt', map_location='cpu', weights_only=False)['config']
    trained = trained if isinstance(trained, dict) else vars(trained)
    assert trained.get('latent_actions_path') == action_paths(action, extra_skip)[0], (trained.get('latent_actions_path'), action, extra_skip)
    name = f'{run}_{split}_s{num_steps}' + ('_smoke' if limit else '')
    out = 'results/evals/next_frame'
    with committing():
        subprocess.run(["python", "scripts/eval/eval_next_frame.py", "--device", "cuda", "--test-h5", "data/zelda_test_frames.h5",
                        "--video-tokenizer-path", DYN_TOKENIZER, "--latent-actions-path", action_paths(action, extra_skip)[0], "--dynamics-path", ckpt,
                        "--link-boxes", "evaluation/zelda_link_boxes.json", "--split-file", "eval_results/action_eval/zelda/set/split.json",
                        "--split", split, "--num-steps", str(num_steps), "--save-predictions", "--out-dir", out, "--name", name,
                        *(["--limit", str(limit)] if limit else [])], check=True)
        subprocess.run(["python", "experiments/small_action/dynamics_score.py", "--split", split, "--eval", f"{run}={out}/{name}.json",
                        "--out", f"{out}/{name}_score.json"], check=True)
    print(f"EVAL DONE {ckpt} -> {out}/{name}.json")


@app.function(cpu=4, memory=8192, volumes=VOLUMES, timeout=30 * 60)
def pick_sweep(runs: str, split: str = "dev"):
    """The selection rule of experiments/small_action/dynamics_score.py over evals of eval_small_dynamics. runs: comma-separated run names."""
    if split != "dev":
        raise ValueError(f"pick_sweep selects on dev only, not {split}: score the winner on confirm once with eval_small_dynamics")
    os.chdir(REPO_DIR)
    out = 'results/evals/next_frame'
    specs = [x for r in runs.split(",") for x in ("--eval", f"{r}={out}/{r}_{split}_s32.json")]
    with committing():
        subprocess.run(["python", "experiments/small_action/dynamics_score.py", "--split", split, *specs, "--out", f"{out}/sweep_{split}_pick.json"], check=True)
