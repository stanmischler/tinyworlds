# Benchmarks on published datasets: run them locally

Three runs that put our pipeline on datasets where recent papers publish prediction scores, scored with each paper's
exact protocol. Every run uses the same recipe: a continuous per-frame KL tokenizer (16 channels per token, 30k steps),
then flow-matching dynamics on its latents (30k steps).

| run | data | actions | resolution | protocol | number to beat |
|---|---|---|---|---|---|
| `point_maze` | DINO-WM Point Maze | real (10-D: 5 env steps x 2) | 256 px, 8 px patches | [NanoWM](https://arxiv.org/abs/2605.23993) Tab. 7: 1 real frame -> 3 predicted, 256 val clips | 36.74 PSNR / 0.984 SSIM / 0.019 LPIPS / 9.66 FID |
| `wall` | DINO-WM Wall | real (10-D) | 256 px, 8 px patches | NanoWM Tab. 7 | 34.05 / 0.994 / 0.010 / 2.64 |
| `obj3d` | OBJ3D | CoMo's continuous z, learned from the video (OBJ3D has no actions) | 128 px, 4 px patches | [LPWM](https://arxiv.org/abs/2603.04553) Tab. 8: 6 real frames -> 44 predicted, 200 test episodes | 31.45 / 0.926 / 0.081 (action-free, see below) |

Copy-last-frame baselines, already measured (same clips and scorer):

| | PSNR | SSIM | LPIPS | FID |
|---|---|---|---|---|
| Point Maze, 256 px | 38.44 | 0.996 | 0.012 | 0.56 |
| Wall, 256 px | 30.10 | 0.989 | 0.031 | 1.05 |
| OBJ3D, 128 px | 22.59 | 0.845 | 0.278 | - |

On Point Maze, copying the last frame already beats NanoWM's published numbers: a model has to beat copy-last to mean
anything there. The script also re-scores NanoWM's released Point Maze and Wall checkpoints on the same clips with our
scorer, so the comparison does not rely on their table alone.

`obj3d` learns its actions, since OBJ3D has none: after the tokenizer, CoMo (motion encoder) reads the
frozen tokenizer's hidden features of frames t and t+1 and outputs a continuous action z (8 x 16 numbers, no k-means); the
dynamics model is conditioned on it. At eval time z is computed from the real frames, the future included, so the score
is optimistic next to LPWM's action-free table; the copy-last baseline is the floor.

## Run

One command per choice of runs, from the repo root; it does everything else (data download and conversion, copy-last
baseline, NanoWM re-scoring, training, eval, `benchmark_results.tgz`):

```bash
git pull
experiments/benchmarks/run_local.sh point_maze          # Point Maze only
experiments/benchmarks/run_local.sh wall                # Wall only
experiments/benchmarks/run_local.sh obj3d               # OBJ3D (tokenizer -> CoMo -> dynamics)
experiments/benchmarks/run_local.sh point_maze wall     # any subset, in that order
experiments/benchmarks/run_local.sh all                 # all three
```

- **Needs:** one CUDA GPU, Python >= 3.10 with PyTorch >= 2.8 (CUDA build); the script installs the other packages of
  `requirements.txt` if missing and never replaces your torch. NanoWM's re-scoring builds its own environment from
  their lockfile with `uv` (installed if missing; needs `git`, `curl` and an NVIDIA driver for CUDA 12.8, as their torch
  is 2.7.1+cu128). Skip it with `SKIP_NANOWM=1`.
- **Data:** downloaded and converted into `data/` on the first call: DINO-WM Point Maze (0.7 GB zip) and Wall (1.7 GB) from
  OSF, OBJ3D (7.4 GB) from HuggingFace. The unzipped DINO-WM folders are large (~80 GB): delete `data/dinowm_raw/` at the end.
- **Only what you chose:** each command downloads only its datasets and runs only their baselines, training and evals.
- **Resumable:** run the same command again after any stop. Finished stages are skipped, training continues from the last
  checkpoint (every 1000 steps), finished evals are skipped.
- **Memory:** OBJ3D's training set is 292k frames (14 GB in RAM at 128 px), and CoMo keeps all of them on the GPU
  (14 GB + the model). On a GPU under ~40 GB, add `COMO_EXTRA="max_frames=150000"`.
  OBJ3D dynamics (7 frames x 1024 tokens, batch 64) is the heaviest stage, sized for an 80 GB GPU; on a smaller one add
  `EXTRA="batch_size_per_gpu=32 gradient_accumulation_steps=2"` (same effective batch).
- **W&B:** off by default (`WANDB_MODE=offline`). `wandb login` and `WANDB_MODE=online` to stream the curves.
- **Python:** the repo's `.venv` if there is one, else `python3`; another one with `PY=/path/to/python`.
- **Time on one H100** (from our Zelda runs with the same recipe): ~3 h per tokenizer, ~3.5 h per dynamics model and ~1 h
  for CoMo: about 20 h for the three runs.

Send back `benchmark_results.tgz` (scores, PNGs, configs, CoMo log; no weights).

## What each step runs

```bash
# data
python scripts/data/convert_dinowm.py --env point_maze      # data/point_maze_frames.h5 (256 px, train split) + point_maze_val_frames.h5
python scripts/data/convert_dinowm.py --env wall
python scripts/data/convert_obj3d.py                        # data/obj3d_{train,val,test}_frames.h5 (128 px)

# copy-last baselines
python scripts/eval/eval_nanowm.py --env point_maze --copy-only
python scripts/eval/eval_nanowm.py --env wall --copy-only
python scripts/eval/eval_lpwm.py --copy-only

# NanoWM's released checkpoints, their code, our clips -> predictions_f16.npz
bash experiments/pusht_nanowm/nanowm_rescore_local.sh point_maze results/nanowm_point_maze_rescore
bash experiments/pusht_nanowm/nanowm_rescore_local.sh wall results/nanowm_wall_rescore

# training: tokenizer then dynamics per run (configs/experiments/benchmarks/); obj3d_como_actions trains CoMo in between
#   (configs/como/zelda.yaml + configs/como/obj3d.yaml) and exports CoMo's z for every frame
#   (scripts/actions/como_actions.py --live --k 0)

# evals (1 Euler step per frame)
python scripts/eval/eval_nanowm.py --env point_maze --run-dir results/point_maze_flow --num-steps 1 \
  --nanowm-npz results/nanowm_point_maze_rescore/predictions_f16.npz
python scripts/eval/eval_lpwm.py --video-tokenizer-path results/obj3d_como_actions/video_tokenizer/checkpoints/video_tokenizer_step_29000 \
  --dynamics-path results/obj3d_como_actions/dynamics/checkpoints/dynamics_step_29000 \
  --latent-actions-path results/obj3d_como_actions/como_z_full --num-steps 1
```

## Smoke test on Modal (what we checked before handing over)

```bash
modal run experiments/benchmarks/modal_benchmarks.py::prepare                      # data into the Modal data volume
TINYWORLDS_GPU=H100 modal run --detach experiments/benchmarks/modal_benchmarks.py::smoke   # run_local.sh with 200-step stages, twice
```
