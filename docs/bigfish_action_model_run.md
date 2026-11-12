# Running the Bigfish action model

One training run: the action model (stage 2) on Procgen Bigfish, 100,000 steps on one GPU. It reads a frozen
tokenizer (stage 1, already trained) and learns one action of 8 numbers per pair of frames. What it is and why:
the pull request "Procgen Bigfish: data, tokenizer, action model".

Needs: one GPU with 32 GB or more (about 17 GB of frames and cached features sit on the GPU before the model; the
smoke test prints the real peak as `mem_gb`), Python 3.10 or newer, and internet for step 2 only.

## One command

Put `bigfish_tokenizer_100k.zip` (sent separately) in the repo root, then:

```bash
git fetch origin && git checkout stanmischler/sta-283-look-for-additional-datasets
bash scripts/infra/run_bigfish_action_model.sh
```

It builds the data, unzips the checkpoint, runs a 20-step smoke test, trains, writes the report and packs everything
to send back into `bigfish_action_model_results.tgz`. Run it again after a stop: finished steps are skipped and the
training resumes. On a cluster whose GPU nodes have no internet, run it once with `--prepare` on a node with
internet (data and checkpoint), then without the flag in the GPU job.

## The same steps by hand

Run everything from the repo root.

```bash
# 1. code and environment
git fetch origin && git checkout stanmischler/sta-283-look-for-additional-datasets
pip install -r requirements.txt
export PYTHONPATH="$PWD"

# 2. data: downloads 8.7 GB from Hugging Face, writes data/bigfish_frames.h5 (1M frames, about 9 GB)
#    and data/bigfish_test_frames.h5 (50,000 frames, 0.4 GB). Needs internet and about 24 GB of RAM.
python scripts/data/convert_procgen.py --game bigfish
rm -rf data/procgen_raw

# 3. tokenizer checkpoint: unzip bigfish_tokenizer_100k.zip (sent separately) at the repo root. It creates
#    results/bigfish_tokenizer_long/video_tokenizer/checkpoints/video_tokenizer_step_99000/
unzip bigfish_tokenizer_100k.zip

# 4. smoke test: 20 steps, about a minute once the data is loaded. Check that it ends with "DONE" and note
#    `it_s` (steps per second) and `mem_gb` on the "step 20" line. Then delete its output.
python scripts/actions/train_como.py --config configs/como/zelda.yaml,configs/como/bigfish.yaml -- \
    run_name=bigfish_action_smoke n_updates=20 log_interval=10 val_interval=10 ckpt_interval=10 button_check=256 use_wandb=false
rm -rf results/bigfish_action_smoke

# 5. the run. Duration = 100,000 / it_s seconds. Relaunching the same command resumes from the last checkpoint
#    (one every 5,000 steps). W&B is on by default (project tinyworlds); add `-- use_wandb=false` to turn it off.
python scripts/actions/train_como.py --config configs/como/zelda.yaml,configs/como/bigfish.yaml

# 6. report on the final checkpoint: probe, confusion matrix, PCA, cluster purity
python scripts/eval/action_report.py --ckpt results/bigfish_action_model/como/checkpoints/como_step_100000 \
    --out-dir eval_results/action_report/bigfish_action_model
```

## On Jean Zay

GPU nodes have no internet, so steps 2 and 3 happen on the login node (see `cluster/README.md` for the setup).

```bash
bash cluster/jeanzay/sync.sh                      # from the laptop: this branch -> $WORK/runs/<folder>
# on the login node, in $WORK/runs/<folder>, with the tinyworlds environment active and pyarrow installed:
python scripts/data/convert_procgen.py --game bigfish && rm -rf data/procgen_raw      # step 2
unzip bigfish_tokenizer_100k.zip                                                       # step 3
sbatch --qos=qos_gpu_a100-dev --time=00:15:00 cluster/jeanzay/train.slurm scripts/actions/train_como.py \
    --config configs/como/zelda.yaml,configs/como/bigfish.yaml -- run_name=bigfish_action_smoke n_updates=20 \
    log_interval=10 val_interval=10 ckpt_interval=10 button_check=256                  # step 4
sbatch cluster/jeanzay/train.slurm scripts/actions/train_como.py \
    --config configs/como/zelda.yaml,configs/como/bigfish.yaml                         # step 5, 20 h max
```

If the run does not finish in 20 h, submit the same command again: it resumes.

## What to look at while it runs

Each line of `results/bigfish_action_model/como/log.jsonl` (also printed) holds, every 1,000 steps:

| Key | Meaning | Healthy |
|---|---|---|
| `check/probe_mlp_acc` | a probe reads the true control (9 classes) from the action, on held-out pairs | well above `check/majority_acc` (about 0.37) and rising |
| `check/shuffle_gap` | how much worse the decoder gets with another pair's action | clearly above 0 |
| `button_acc` vs `check/head_acc` | the button head on its 10,000 labelled pairs vs on held-out pairs | close to each other; a wide gap means the labels are memorised |
| `val/recon` vs `val/copy` | decoder error vs the error of copying frame t | `val/recon` below `val/copy` |

## What to send back

- `results/bigfish_action_model/como/log.jsonl` and `como/visualizations/`
- the checkpoints `como_step_*` (about 100 MB each), or at least the last one and the one with the best
  `check/probe_mlp_acc`
- the folder written by step 6
