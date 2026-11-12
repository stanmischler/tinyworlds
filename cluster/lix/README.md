# Running the action model on your own cluster

One run of the small action model on the 5 h Zelda data: 200k steps on one GPU, scored every 10k steps.
Everything below is run from the repo root.

## What you need

| | |
|---|---|
| GPU | one GPU with about 55 GB of memory free (A100 80 GB, H100). The frames sit on the GPU (16 GB) and a batch of 256 takes the rest |
| Time | about 20 h on an H100 (3 steps/s), more on an A100 |
| RAM, CPUs | 64 GB, 8 CPUs |
| Disk | 13 GB of data, about 6 GB of checkpoints and pictures |
| Python | 3.10 or later, with internet on the node where the setup runs |

A smaller GPU: add `batch_size=128` to the launch command. It halves the memory of a batch, but it is no longer the same recipe.

## 1. Data

Seven files go into `data/` (listed in `cluster/lix/data_files.txt`, checksums in `cluster/lix/data.sha256`):

| file | size | what it is |
|---|---|---|
| `zelda_five_hours_frames.h5` | 10.6 GB | training frames, 128×128: our capture + 10 YouTube videos |
| `zelda_five_hours_shifts.npz` | 8 MB | measured camera shift of every pair |
| `zelda_five_hours_motion.npz` | 0.6 MB | changed pixels and text-box flags of every pair (focus pool, text filter) |
| `zelda_train_frames.h5`, `zelda_train_shifts.npz` | 2.1 GB | our capture alone, for the train-versus-held-out gap |
| `zelda_test_frames.h5`, `zelda_test_shifts.npz` | 0.25 GB | held-out capture, for the scores |

They are in a Hugging Face dataset whose files are only served to approved accounts:
https://huggingface.co/datasets/stanmischler/zelda-action-model-data . If your account is not approved yet, press "request access"
on that page. Then log in on the cluster:

    pip install huggingface_hub && hf auth login

and the setup below downloads them. If you received the files another way, put them in a folder and pass it as `DATA_SRC` (see below).

The judged pairs used for the scores are in the repo (`eval_results/action_eval/zelda/set/`, `evaluation/zelda_link_boxes.json`).

## 2. Setup (once)

    bash cluster/lix/setup_env.sh

It builds `.venv` with the package versions the run was developed on (`cluster/lix/constraints.txt`), caches the weights of the LPIPS
score, downloads the data files that are missing from `data/`, checks them and checks that the run's config loads. If the data is in another folder:
`DATA_SRC=/that/folder bash cluster/lix/setup_env.sh`.
If the cluster provides its own PyTorch, build the environment yourself (`pip install -r requirements.txt` on top of it) and pass it:
`VENV=/path/to/env bash cluster/lix/setup_env.sh`; nothing is installed in it. Launch with the same `VENV=...`.

## 3. Launch

    bash cluster/lix/run.sh              # on a machine with a GPU
    sbatch cluster/lix/run.sh            # under Slurm, from the repo root; add your partition and account options

Relaunching the same command continues from the last checkpoint, so a time limit shorter than the run is fine (the script asks for 30 h;
an A100 may need more than one job).
With `WANDB_API_KEY` set, the run also logs to Weights & Biases (project `tinyworlds`); on a node without internet, also set
`WANDB_MODE=offline`, or unset the key.

## 4. Follow it

    python cluster/lix/scores.py

prints one line per scored checkpoint: general PSNR, and PSNR and LPIPS on Link's box, camera still and camera moving.
Everything is in `results/weights_and_window/small_action/`:

| | |
|---|---|
| `log.jsonl` | training loss every 100 steps, all scores every 10k steps |
| `checkpoints/step_N/` | weights and optimiser state, every 10k steps |
| `visualizations/step_N/` | Link crops: frame t, target, prediction, prediction with another action |
| `visualizations/pca_step_N.png` | the actions of the camera-still walks, coloured by direction |
| `visualizations/pred_step_N.png` | whole frames: frame t, target, prediction |

Scores are on the dev half of the held-out capture, on whole frames. Training windows and the strip the shift reveals are not scored.

## The run

`configs/experiments/small_action/weights_and_window.yaml`, merged over `simple_recipe.yaml` (the recipe and its knobs) and `base.yaml`.

| knob | key | this run |
|---|---|---|
| action size | `action_dim` | 8 |
| share of each batch from the focus pool | `still_fraction` | 0.5 |
| contrastive weight | `contrastive_weight` | 0.0001 |
| error weights | `error_weight` | 1.0 |
| probability that a pair is cut to a window | `window_fraction` | 0.5 |
| pixel loss | `pixel_loss` | mse |
| edge loss weight | `edge_weight` | 0 (off) |

Another value, under another name:

    bash cluster/lix/run.sh weights_and_window error_weight=0.5 run_name=half_weights

Numbers to compare with (dev half, 200k): the first run of the simple recipe has general PSNR 36.32 and Link-box LPIPS .255 on
camera-still walks; copying frame t gives .275; the shipped 5-stage model .185.
