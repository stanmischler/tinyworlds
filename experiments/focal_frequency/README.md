# Focal frequency loss: run it locally

The Zelda flow-matching dynamics model, trained with the loss of Degeorge et al. 2026, "Balancing Frequencies and Pixels
in Flow Matching" ([arXiv 2609.02748](https://arxiv.org/abs/2609.02748)). Everything else is the "longer flow matching"
run (34.36 dB): frozen KL c16 temporal tokenizer, frozen CoMo actions (full z), 13M dynamics model (256/512/12), batch 64,
lr 3e-4 cosine over 100k steps, v-prediction, shift 3, EMA 0.999. Only the dynamics model trains, from scratch.

## The loss

Each training step, the dynamics model predicts the velocity v̂ of the noised last-frame latent x_t (noise level s), as
before. Its predicted clean latent is x̂ = x_t + s·v̂. The frozen tokenizer decoder turns [3 context latents, x̂] into the
predicted frame (the gradient flows through the decoder into x̂), and [3 context latents, x1] into the target frame
(the tokenizer's reconstruction, no gradient). Then (`models/frequency_loss.py`):

```
e      = |FFT(pred frame) − FFT(target frame)|          orthonormal 2D DFT per RGB channel, whole 128×128 frame
f-loss = mean over channels and frequencies of  sg(e / max e) · log(1 + e),  divided by max(s, 0.05)²
loss   = λ(step) · scale · f-loss + (1 − λ(step)) · v-loss        λ = sigmoid(−(step − switch) / width)
```

λ goes from 1 to 0: f-loss only at the start, v-loss only at the end, a smooth handover around `switch` (width 5k:
λ = 0.88 at switch − 10k, 0.12 at switch + 10k). `sg` = no gradient through the weight. Once λ < 0.01 the decoder is
skipped. Sampling and eval are unchanged.

## The runs

| run | dynamics loss | steps | what it gives |
|---|---|---|---|
| longer flow matching (exists, its weights are downloaded) | v-loss only | 100k | baseline, and its test curve (`longer_fm_test_curve.csv`) |
| frequency loss only | f-loss only (λ = 1) | 60k of the 100k schedule | its test curve |
| focal frequency loss | λ · scale · f-loss + (1 − λ) · v-loss | 100k | the result |

The switch step and the scale are not tuned by hand. As in the paper, `pick_switch.py` sets the switch at the step
where the two test PSNR curves cross (f-only ahead early, v-only ahead after), and the scale to v-loss / f-loss of the
f-only run around that step, so the two terms are the same size where they are mixed. It writes them to
`results/focal_frequency_evals/switch.json` with a plot of both curves.

## Run

```bash
git pull
experiments/focal_frequency/run_local.sh    # data + weights -> frequency loss only -> switch -> focal frequency loss -> evals
```

- **Needs:** one CUDA GPU, Python >= 3.10 with PyTorch >= 2.8 (CUDA build). The script installs `requirements.txt` if
  packages are missing. Access to the gated HuggingFace repo
  [`stanmischler/tinyworlds-focal-assets`](https://huggingface.co/stanmischler/tinyworlds-focal-assets) (frozen weights;
  request access on its page), then `hf auth login` once on the GPU machine.
- **Downloads:** the Zelda frames (~2 GB, then split exactly as ours) and the frozen weights (~600 MB).
- **Resumable:** if it stops for any reason, run the same command again. Training continues from the last checkpoint
  (every 1k steps), and finished evals are skipped.
- **Smaller GPU:** `BATCH=32 ACCUM=2` (or `16 4`) splits each batch of 64 into chunks; same result, slower.
- **W&B (optional):** `wandb login`, then `WANDB_MODE=online`.

### GPU memory and time (measured on an H100 80 GB)

| GPU memory | setting | peak |
|---|---|---|
| 80 GB | default (`BATCH=64`) | 73.8 GB (measured) |
| 40-48 GB | `BATCH=32 ACCUM=2` | ~37 GB (estimate: half the activations) |
| 24 GB | `BATCH=16 ACCUM=4` | ~19 GB (estimate) |

On the H100: 1.34 s per step while the f-loss is on (the decoder runs forward and backward, in fp32), ~0.5 s once it is
off. Frequency loss only: 60k steps, about 23 h. Focal frequency loss: the decoder stays on until about switch + 23k
steps, so about 26 h for a switch near 30k. Both runs one after the other: about 2 days.

## What to send back

`focal_frequency_results.tgz`, a few MB: the configs, both test curves and loss logs, `switch.json` + the curve plot,
prediction PNGs every 1k steps, and all eval JSONs + PNGs. No weights. The script prints the final table at the end.

## What the scores are

`results/focal_frequency_evals/<run>_s{1,4}.json` (`scripts/eval/eval_next_frame.py`, the 890 held-out test windows,
1 and 4 Euler steps): PSNR, SSIM, LPIPS of the predicted next frame, next to copy-last-frame and the tokenizer
reconstruction. `spectrum.pred_vs_recon`: power of the predicted frames vs the reconstruction in three frequency bands
(low < 0.0625, mid 0.0625-0.25, high 0.25-0.5 cycles per pixel), in %; negative = fine detail lost. This is what the
f-loss is meant to fix.
