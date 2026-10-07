# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal PyTorch reimplementation of DeepMind's Genie: a world model trained on action-less gameplay video. Three models are trained in sequence — a **video tokenizer** (FSQ autoencoder → discrete frame tokens), a **latent action model** (infers a discrete action between consecutive frames, unsupervised), and a **dynamics model** (MaskGIT-style: predicts masked next-frame tokens conditioned on actions). All three share one backbone, the space-time transformer in `models/st_transformer.py`. The README has the conceptual write-up; read it for the math, not for the commands (several README commands are wrong — see Gotchas).

## Environment

- Requires **Python ≥ 3.10** (`str | None` annotations are evaluated at import). A venv at `.venv/` (Python 3.12) exists; use `./.venv/bin/python`.
- No CUDA on this Mac. `TrainingConfig.device` only accepts `CUDA` or `CPU` (enum *names*, uppercase); inference accepts `mps`. `compile`/`amp`/`tf32` must be off for CPU/MPS runs.
- **Always run from the repo root with `PYTHONPATH="$PWD"`.** Dataset paths, `configs/` defaults and checkpoint discovery are all `os.getcwd()`-relative.
- No test suite, no linter config. Verification is: run the smoke pipeline, then inspect the PNGs in `results/<run>/<stage>/visualizations/` and `inference_results/`.

## Commands

```bash
export PYTHONPATH="$PWD"
PY=./.venv/bin/python

# data / pretrained weights (HuggingFace, into data/ and results/)
$PY scripts/data/download_assets.py datasets --pattern "zelda_frames.h5"
$PY scripts/data/download_assets.py models --suite-name sonic

# smoke-test the whole 3-stage pipeline on CPU (~1 min; needs data/zelda_frames.h5)
WANDB_MODE=offline MPLBACKEND=Agg $PY scripts/pipeline/full_train.py --config configs/dev/dev_training_cpu.yaml

# real training (GPU box)
$PY scripts/pipeline/full_train.py --config configs/training.yaml

# one stage only, with overrides (bare key=value after --)
$PY scripts/dynamics/train_dynamics.py --config configs/dynamics.yaml --training_config configs/training.yaml -- n_updates=100 batch_size_per_gpu=8

# inference — random actions, non-interactive (safe to run from an agent)
MPLBACKEND=Agg $PY scripts/inference/run_inference.py --config configs/inference.yaml -- use_latest_checkpoints=true dataset=SONIC use_interactive_mode=false use_actions=true

# training on Modal (auth + `wandb` secret + volumes already set up)
./.venv/bin/modal run scripts/infra/modal_train.py::download --pattern "zelda_frames.h5"
TINYWORLDS_GPU=A100-80GB ./.venv/bin/modal run --detach scripts/infra/modal_train.py --dataset ZELDA
```

Do not launch Modal GPU runs without being asked — they bill the user.

After any local smoke run, **delete its `results/<timestamp>/` directory** (see checkpoint discovery below).

## Repository layout (STA-58) — where things live and where new code goes

- Packages = library: `models/`, `datasets/` (+ `datasets/split.py`), `evaluation/` (image metrics, held-out windows, motion proxies, action-code metrics, LAM decode, Zelda judge set), `utils/`.
- `scripts/<stage>/` = entry points, in pipeline order: `data/` `tokenizer/` `actions/` (LAM, CoMo) `dynamics/` `pipeline/` (full_train) `inference/` `eval/` `infra/` (Modal, worktree, rc).
- `experiments/<study>/` = one-off studies, named in a word or two (**never a ticket number**), each with its own `modal_<name>.py` (never `modal.py`: it shadows the `modal` package) importing `app`, `VOLUMES`, `committing` from `scripts/infra/modal_train.py`.
- `configs/`: base stage yamls + `training.yaml`, `como/`, `dev/`; every past run's configs under `configs/experiments/<run>/`.
- Rules: scripts import only from the packages, never from another script (no `sys.path` hacks); new metric = function in `evaluation/` + thin CLI in `scripts/eval/`; new game = dataset class + one row in `VIDEO_GAMES` (`datasets/data_utils.py`); new model kwargs go in the `*_kwargs(cfg)` helpers of `utils/utils.py` so training and checkpoint loading build the model identically; a study that graduates moves into the packages/`scripts/` and its folder is deleted.
- Never move/rename `utils.config.{DeviceType, DistributedConfig, FSDPMixedPrecisionConfig}`: they are pickled into every `state.pt`.

## Architecture — what you must read across files to understand

**Config layering** (`utils/config.py`): dataclass schema → stage yaml (`configs/<stage>.yaml`) → `configs/training.yaml` overlaid on top (only keys in the stage schema, non-null) → CLI dotlist. So `training.yaml` *beats* the stage yamls; only the CLI beats `training.yaml`. Shared knobs (`dataset`, `frame_size`, `patch_size`, `n_actions`, `latent_dim`/`num_bins`, `amp`/`tf32`/`compile`, `distributed.*`, `optimizer`, MoE flags, `use_wandb`) live in `training.yaml`; per-stage knobs (`n_updates`, `batch_size_per_gpu`, `learning_rate`, `log_interval`, width/depth, `checkpoint:` for resume) live in the stage yamls.

**Pipeline orchestration** (`scripts/pipeline/full_train.py`): creates `results/<timestamp>/`, exports it as `NG_RUN_ROOT_DIR`, runs the three stage scripts (`scripts/{tokenizer,actions,dynamics}/train_*.py`) as subprocesses (via `torchrun` when `nproc_per_node > 1`), then locates the tokenizer/LAM checkpoints with `find_latest_checkpoint` and passes them to dynamics as `video_tokenizer_path=…`/`latent_actions_path=…`. **CLI overrides given to `full_train.py` are NOT forwarded to the stage scripts** — they re-read `training.yaml` from disk. Edit the yaml (or write a derived one, as `scripts/infra/modal_train.py` does).

**Checkpoint format & discovery** (`utils/utils.py`): a checkpoint is a **directory** `…/checkpoints/<stage>_step_N/` holding `model_state_dict.pt`, `optim_state_dict.pt`, `state.pt` (config dict + scheduler + step). The `load_*_from_checkpoint` helpers rebuild the model from the config in `state.pt`, so no yaml is needed to load. `find_latest_checkpoint` globs all of `results/**`, picks the **newest run dir by ctime, then highest step** — a 2-step smoke run will shadow a real one. The HuggingFace pretrained weights are legacy single-file `.pth` (`{'model', 'config', ...}`) and do not load without conversion to the directory format; the Sonic set in `results/20260922_152223_sonic/` is already converted.

**Model wiring** (`models/`): tokenizer `tokenize()` → indices `[B,T,P]`; `fsq.get_latents_from_indices` → latents `[B,T,P,L]`; LAM `encoder`+`quantizer` → action latents; dynamics consumes tokenizer latents + action latents (FiLM via `norms.AdaptiveNormalizer`, which shifts conditioning by one step so a_{t-1} drives z_t) and emits logits over the `num_bins**latent_dim` codebook; `forward_inference` does the iterative unmasking; tokenizer `detokenize()` → pixels. `n_actions` must be a power of 2 (`action_dim = log2(n_actions)`, 2 FSQ bins per dim). Shape suffixes (`B,T,P,E,L,A,C,H,W,Hp,Wp,S`) are defined in the README and used in every annotation — keep them.

**Data** (`datasets/`): each game is a `VideoHDF5Dataset` subclass with **hardcoded** `resolution`/`fps`/`preload_ratio`; the `.h5` cache fixes the actual frame size (zelda/picodoom 128, sonic/pong 64). `frame_size` in `training.yaml` must match the `.h5`, or `patch_embed` shape-errors. Train and val loaders are the **same array** (`disable_test_split=True` everywhere) — val loss is not held-out. Batches are `[B,T,C,H,W]` in `[-1,1]`. Adding a game = mp4 in `data/` + subclass in `datasets.py` + a row in `VIDEO_GAMES` (`data_utils.py`) + dataset entry in `utils/config.py`.

## Gotchas (all verified)

- `use_wandb: false` **crashes** latent-actions and dynamics training at the first `log_interval` step: logging stats are computed inside `if args.use_wandb:` but used outside it (`scripts/actions/train_latent_actions.py`, `scripts/dynamics/train_dynamics.py`, in the `log_interval` block). Use `WANDB_MODE=offline` with `use_wandb: true` instead, or fix the guard.
- README says `-- --dataset=ZELDA` to `full_train.py`; it is a no-op (see orchestration above).
- `NG_NUM_WORKERS` / `NG_PIN_MEMORY` / `NG_PREFETCH_FACTOR` exported by `run_command` are read by nothing; `data_utils.py` hardcodes 2 workers, no pin memory.
- Interactive inference blocks on terminal `input()` per step and `assert`s on bad input; there is no live frame display. Never run it from an agent — use `use_interactive_mode=false use_actions=true`.
- `configs/dev/dev_training.yaml` (upstream's smoke profile) **no longer loads** — it predates the mandatory `distributed` block and targets PICODOOM at 64px. Use `configs/dev/dev_training_cpu.yaml` (verified end-to-end on this Mac) or copy it.

## Worktree lifecycle (how new work is started here)

The main checkout is `Project Vicky/tinyworlds` and stays on `main`. Each new direction gets its own worktree + branch under `Project Vicky/worktrees/`:

```bash
cd "/Users/stanislas/Desktop/Project Vicky/tinyworlds"
scripts/infra/new_worktree.sh <name>            # branch <name> from main → ../worktrees/<name>
scripts/infra/new_worktree.sh <name> <base>     # stack on an unmerged branch instead
cd "../worktrees/<name>" && export PYTHONPATH="$PWD"

# ... commit inside the worktree as usual ...
git push -u origin <name>                 # publish the branch (origin = stanmischler/tinyworlds)

# merge from the main checkout (or open a PR), then retire the worktree
cd "/Users/stanislas/Desktop/Project Vicky/tinyworlds"
git merge <name> && git push
git worktree remove --force ../worktrees/<name>   # --force: the symlinks count as untracked
```

What the script wires up in a worktree, and why:
- `.venv` and `data/` are **symlinks** to the main checkout (no reinstall, no re-download).
- `CLAUDE.md` is a **symlink** to the main checkout's copy, with `git update-index --skip-worktree` so the swap never shows as modified or gets committed. **Edit CLAUDE.md only in the main checkout**, and commit it from there; every worktree sees the edit immediately.
- `configs/dev/dev_training_cpu.yaml` is copied (it is gitignored).
- `results/` is **not** shared — `find_latest_checkpoint` is cwd-relative, so each worktree's inference only sees its own checkpoints.
- `scripts/infra/modal_train.py` mounts `.`, so running it from a worktree trains that branch's code.

`upstream` = `AlmondGod/tinyworlds` (the original project); `git fetch upstream && git merge upstream/main` on `main` pulls their changes.

**Phone control (Remote Control).** User settings have `remoteControlAtStartup: true`, so every interactive session already shows up in the Claude app under Code. For a session that must outlive the terminal (e.g. one driving a Modal run), `scripts/infra/rc.sh <worktree-name>` starts `claude remote-control` for that worktree inside tmux (`rc-<name>`) wrapped in `caffeinate -i`; `scripts/infra/rc.sh --list` / `--stop <name>` manage them. Push notifications are on (`inputNeededNotifEnabled`, `agentPushNotifEnabled`), so when working remotely, send a push when a long task finishes or a decision is needed.

## Contributing conventions (from README)

Keep backwards compatibility, keep code lean, annotate every tensor with the shape key, and include inference visualizations in PRs. The README TODO list is the roadmap (RoPE/AliBi, AdaLN-Zero, MaskGIT schedulers, larger runs).
