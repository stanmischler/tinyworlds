# Drawing model (STA-76): run it locally

One model, trained from scratch on Zelda. An **encoder** turns each frame into a wide continuous code z
(256 tokens x 1024). A CoMo **action model** reads z(t) and z(t+4) - z(t) and outputs an action (8 x 16 numbers).
**One decoder** draws a frame from z(t) and an action. With a learned NULL action it redraws frame t, so it acts as
the tokenizer decoder. With the inferred action it predicts frame t+4, so it acts as the dynamics model.
Two runs differ only in that decoder:

| run | decoder | loss | sampling |
|---|---|---|---|
| `p` | plain, one pass | pixel MSE | - |
| `f1` | flow matching in pixel space (from noise, v-pred, shift 3, EMA 0.999) | velocity MSE | Euler, 1 and 4 steps |

Shared: batch 256, lr 1e-4 linear decay, 50k steps, a checkpoint every 5k, redraw weight 0.5, InfoNCE 0.004.

## Run

```bash
git pull
experiments/drawing/run_local.sh            # data -> train p -> train f1 -> evals -> drawing_results.tgz
```

- **Needs:** one CUDA GPU, Python >= 3.10 with PyTorch (CUDA build). The script installs `requirements.txt` if packages
  are missing.
- **Data:** about 2 GB is downloaded from HuggingFace into `data/`, then split exactly as ours.
- **Resumable:** if it stops for any reason, run the same command again. Training continues from the last checkpoint,
  and finished evals are skipped.
- **One run only:** `experiments/drawing/run_local.sh p` (or `f1`).
- **Smaller GPU:** set `GRAD_ACCUM` from the table below. It splits the batch of 256 into more, smaller chunks. The
  result is the same (the loss is averaged over equal chunks); only speed changes.
- **Other Python:** `PY=/path/to/python experiments/drawing/run_local.sh`.
- **W&B (optional):** `wandb login`, then add `EXTRA="use_wandb=true"`.

### GPU memory -> GRAD_ACCUM (measured on an H100 80 GB, peak during training + held-out pass)

| GPU memory | GRAD_ACCUM | chunk (pairs) | peak p | peak f1 | speed vs 4 |
|---|---|---|---|---|---|
| does not fit | 1 / 2 | 256 / 128 | OOM | OOM | |
| 80 GB | 4 (default) | 64 | 50 GB | 56 GB | 1 |
| 40-48 GB | 8 | 32 | 28 GB | 31 GB | ~0.93 |
| 24 GB | 16 | 16 | 17 GB | 19 GB | ~0.85 |

Time per run on an H100 at GRAD_ACCUM 4: about 0.65 steps/s, so about 20 h for p and 22 h for f1 (50k steps).
Both runs one after the other: about 2 days.

**Reading f1 early:** its held-out numbers use the EMA weights (decay 0.999), which still mostly hold the starting
weights for the first ~3k steps. So f1 shows ~10 dB until then; that is expected. From 5k steps on, the EMA reflects
training.

## What to send back

`drawing_results.tgz`, a few MB: training logs, prediction PNGs every 5k steps, and all scores. No weights.

## What the scores are

All scores are on the held-out Zelda test split, with the same scripts as J1 and the results table:
- `results/drawing_<run>/evals/next_frame/*.json` (`scripts/eval/eval_como_pred.py`, 890 windows): PSNR, SSIM and
  LPIPS of the predicted frame t+4. Modes:
  - `como`: full action;
  - `como_k16`: action snapped to 16 k-means codes;
  - `shuffled`: another window's action;
  - `copy`: frame t, the copy-last baseline;
  - `redraw`: frame t+4 decoded from its own z, the reconstruction ceiling.
- `eval_results/lam_judge/drawing_<run>_<step>_k{16,8}/score.json`: whether the actions match Link's judged moves
  (NMI_adj over 10 k-means seeds).
- `eval_results/lam_eval/zelda/drawing_<run>_<step>/score.json`: the results-table action score.
- `results/drawing_<run>/como/log.jsonl`: training curves (held-out prediction / redraw PSNR every 1k steps).
