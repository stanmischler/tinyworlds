# Action-conditioning ablation (first comparative experiment)

Status: revision 2, after independent review (2026-09-23). Code changes below are implemented and smoke-tested on CPU (3-step runs of each arm, seeded runs bit-identical, `fullgraph=True` compile of the training forward, eval in all three modes, comparison script). Not launched.

## Question

Does conditioning the dynamics model on latent actions help next-frame prediction at all, and can one
model serve both the conditioned and the unconditioned case? Three dynamics-training arms, one shared
upstream (tokenizer + LAM), one eval framework (`scripts/eval/eval_next_frame.py`, extended).

Motivation: the latent action model (LAM) is near-collapsed. On the 500 test windows, code 0 covers 81% of
transitions for the v2 LAM (n_actions 16) and 83% for the v3 LAM (n_actions 8), and code 0 lumps
left-scroll, right-scroll and static transitions together (phase-correlation check, 2026-09-23). The
action signal the dynamics model receives is weak; this experiment measures what it is worth.

## Arms

All three arms train ONLY the dynamics stage, from the same tokenizer and LAM checkpoints, the same dynamics
config and the same seed. The single difference is `action_dropout_prob`.

| Arm | Name    | `action_dropout_prob` | Training-time conditioning                                           |
|-----|---------|-----------------------|----------------------------------------------------------------------|
| 0   | `lam`   | 0.0                   | LAM action latents for every sample (repo behaviour)                 |
| 1   | `none`  | 1.0                   | Null action for every sample (actions framework effectively removed) |
| 2   | `mixed` | 0.5                   | Per sample: LAM latents with prob 0.5, null action otherwise         |

**Null action.** A learned vector `null_action` of shape `[1, 1, A]`, initialised at zero, created only when
`action_dropout_prob > 0` (so existing checkpoints still load). It replaces the whole `[B, T-1, A]` action
sequence of a dropped sample and goes through the existing FiLM path unchanged. Why learned rather than a
fixed zero: FiLM applies SiLU before the linear layer (`models/norms.py`), and SiLU maps the collapsed code 0
(latent all -1) to -0.27 per dim while zero stays 0, so a fixed zero null would sit closer to code 0 than any
two codes are to each other, weakening the within-model contrast. A learned null can move away. (Note: the
first frame receives identity FiLM because gamma/beta are prepended with zeros for the one-step shift; that
is not the same as the null action and does not need to be.)

**Dropout is per sample.** A `[B, 1, 1]` Bernoulli mask is drawn once per batch and applied with
`torch.where`, so the compiled graph is static. The Bernoulli draw and the masking-type draw (below) are made
in every arm, including arm 0, so all arms consume the RNG identically and see the same MaskGIT masks and the
same mask-type choices; the only difference is whether the dropout decision has an effect. The LAM is loaded
and run in every arm (its output is discarded for dropped samples), so the three training runs are identical
except for the knob.

**Interpretation of the user's protocol 1** ("drop the action tokenizer and the actions framework; the DM
takes the frames and unmasks the next one"): arm 1 never sees an informative action, so it is a pure
frame-conditioned next-frame predictor. Not training a LAM at all would give the same model but lose the
shared-upstream control, so the LAM checkpoint is kept and simply never used.

## Shared upstream

`results/2026_09_23_17_38_44/` on the `tinyworlds-results` volume (the v3 run: tokenizer step 7500, LAM
step 2250 with n_actions 8). Local copy: `results/sonic_v3_2026_09_23/`. It is the newest run dir on the
volume that holds these stages, so `full_train.py`'s `find_latest_checkpoint` fallback resolves to it when the
tokenizer and LAM stages are skipped; ablation run dirs contain no tokenizer/LAM stage and cannot shadow it.
Because every arm shares it, the tokenizer reconstruction ceiling and the LAM histogram are identical across
arms and cancel out of every contrast. The resolved paths are printed at the start of the dynamics stage and
must contain `2026_09_23_17_38_44` (check in each app's log).

## Dynamics training config (identical across arms)

- `full_last_frame_mask_prob: 0.5` ("predict the next frame" training): with prob 0.5 a batch is trained like
  inference (clean context, last frame fully masked); otherwise MaskGIT's 50-100% random masking over all
  frames. 0.5 rather than 1.0 because MaskGIT inference unmasks the last frame over 10 steps and the model
  must have seen partially masked last frames. Same value as the running v3, so arm 0 stays comparable to it.
  Implemented as a tensor select (no Python branch, so no graph break). Known mismatch kept for scope: the
  MaskGIT branch still masks context frames, which inference never does; a "random ratio on the last frame
  only" variant is a different experiment.
- Model: embed 256, hidden 512, 8 blocks, 8 heads (v2/v3 widths). Batch 256, lr 3e-4, cosine with 1k warmup,
  `n_updates: 20001` (so a step-20000 checkpoint is written), AdamW, amp + tf32 + compile, `num_workers 8`,
  `pin_memory true`, checkpoint every 1000 steps.
- Budget: v2 (60k) did not beat sonic_short (15k) on the eval despite a much lower loss, so long schedules are
  not where the signal is. 20k steps is ~140 epochs of the 36.8k train frames; the held-out eval and the
  step-10k eval below guard against overfitting. Wall clock ≈ 1.6 h per arm at the 3.5 it/s the running v3
  shows on an H100 (the loader knobs did not raise throughput over v2's 3.4 it/s; the stage is compute
  bound). Three arms ≈ 5 H100 hours sequential, or three parallel apps.
- Seed: a `seed` knob in `DynamicsConfig` (default 0) seeds torch/numpy/random and a `torch.Generator`
  passed to the DataLoader, so every arm starts from the same init and sees the same batch order in every
  epoch (without the generator, `RandomSampler` reseeds from the global RNG at each epoch).
- Checkpoints evaluated: step 20000 (final; LR is ≈0 by then) and step 10000 (is the ranking stable?).

## Evaluation

All numbers come from `scripts/eval/eval_next_frame.py` on `data/sonic_test_frames.h5`: 500 fixed windows
(10 blocks x 50), 3 real context frames, target fully masked, 10 greedy MaskGIT steps, temperature 0, all on
the same device (MPS). Per-window metrics saved to JSON: token accuracy (primary), PSNR, SSIM (secondary), the
copy-baseline values of each, and the LAM code of the target transition (recorded in every mode, even when
not fed to the model). References: copy-last-frame baseline and tokenizer reconstruction ceiling (same for
every arm).

Action modes (`none` is new):

| Mode     | Conditioning fed to the model                                        | Run on arms |
|----------|----------------------------------------------------------------------|-------------|
| `lam`    | LAM codes for all T-1 transitions, incl. the one into the true target | 0, 2        |
| `none`   | the model's `null_action` on ALL T-1 transitions (zeros if the model has none) | 1, 2 |
| `random` | LAM codes on context transitions, seeded random code on the last (existing) | 0, 2   |

`none` must replace every transition, not only the last: that is what dropped samples looked like in
training. Arm 0 in `none` mode and arm 1 in `lam` mode are off-distribution and not part of the design.

Caveat on `lam` mode: the LAM sees the true target frame, so it hands the model a few oracle bits about the
frame being predicted. Contrast 1 below is therefore "oracle latent action vs none", an upper bound on the
value of conditioning, not the value of actions supplied by a player.

Contrasts, each a paired comparison over the same 500 windows (new `scripts/eval/compare_evals.py`):

1. **Does action conditioning help?** arm0-lam vs arm1-none. The headline. Given the collapsed LAM the
   expected effect is small; "no difference" is a legitimate result and means the actions framework currently
   contributes nothing to prediction quality.
2. **Within-model action information.** arm2-lam vs arm2-none. Same weights, only the action changes: a
   cleaner controllability number than the random-action gap (no unfamiliar-code confound).
3. **Cost of dropout on the conditioned path.** arm2-lam vs arm0-lam.
4. **Cost of dropout on the unconditioned path.** arm2-none vs arm1-none.
5. Continuity with earlier runs: arm0-lam vs arm0-random, arm2-lam vs arm2-random.

`compare_evals.py` reports, per metric: mean paired difference; a **cluster bootstrap CI over the 10 test
blocks** (resample blocks with replacement) as the decision interval, because adjacent windows share 2 of 4
frames and the same scene (on the existing v2 lam-vs-random data the naive per-window CI is about half the
width of the block CI); the naive per-window CI as a secondary number; the count of blocks favouring each
side; and the same statistics restricted to windows whose target transition carries a LAM code other than
0 (about 85 of 500), since any action effect lives there and the pooled mean dilutes it ~5x.

Decision rule: a contrast "wins" if the block-bootstrap 95% CI of the paired token-accuracy difference
excludes zero. PSNR/SSIM must not contradict the direction. Five contrasts x three metrics at nominal 95%
with no correction: exploratory, stated as such. Copy baseline (token acc 0.43) remains the bar every arm is
measured against.

Known limitation: one training seed per arm. The block CI captures measurement noise, not training noise
across seeds. If the headline contrast is close, the follow-up is a second seed of arms 0 and 1.

## Sequencing and gate

The relaunched v3 run (app `ap-vI04zkalwf3LhvHyKs0NIR`, started 20:20, 60k steps, batch 256, LAM actions,
`full_last_frame_mask_prob 0.5`, the same tokenizer/LAM) is arm 0 on a longer schedule. Its step-10k
checkpoint lands around 21:20 and step-20k around 22:10. Gate: evaluate v3 step-10k (and 20k when available)
in `lam` mode first (~17 min each on MPS). If token accuracy is still far below the copy baseline (v2: 0.29 vs
0.43), all three arms would lose to copy-last and the ablation should wait for a better base recipe. If the
masking fix lifted it, launch the three arms. v3 also tells whether 20k steps is enough at this width.

## Launch recipe (Modal, after the gate)

```
TINYWORLDS_GPU=H100 ./.venv/bin/modal run --detach scripts/modal_train.py --dataset SONIC_TRAIN \
  --training-config configs/action_ablation/training.yaml \
  --overrides dynamics_config=configs/action_ablation/dynamics_lam.yaml     # then _none, _mixed
```
`dynamics_config` is a `TrainingConfig` field, so the override reaches `full_train.py`; the ablation
`training.yaml` sets `run_video_tokenizer: false`, `run_latent_actions: false`. Use app ids for logs (all
apps are named `tinyworlds`). Pull each run's step-20000 and step-10000 dynamics checkpoints to
`results/action_ablation/<arm>/`, then evaluate with explicit `--video-tokenizer-path / --latent-actions-path
/ --dynamics-path --name <arm>_<step>_<mode>` (no need to copy the shared upstream three times).

## Code changes (done)

1. `models/dynamics.py`: `full_last_frame_mask_prob` (tensor select) and `action_dropout_prob` with the
   learned `null_action`, both drawn unconditionally. Defaults 0.0 = current behaviour.
2. `utils/config.py`: `full_last_frame_mask_prob`, `action_dropout_prob`, `seed` in `DynamicsConfig` ONLY
   (the `training.yaml` overlay copies any non-null shared key over the stage yaml, so a stray line in
   `training.yaml` would silently make the arms identical); `num_workers` / `pin_memory` in all stage configs
   and `TrainingConfig` (harmless to share). `utils/utils.py`: pass the two model knobs through
   `load_dynamics_from_checkpoint`.
3. `scripts/train_dynamics.py`: forward the knobs, seed everything incl. a DataLoader generator, print the
   resolved tokenizer/LAM paths. (The W&B action histogram still shows the LAM codes before dropout.) `datasets/data_utils.py`: `num_workers` /
   `pin_memory` / `generator`. `scripts/modal_train.py`: `cpu=8`.
4. `scripts/eval/eval_next_frame.py`: `--action-mode none`; save per-window `ssim`, `copy_ssim`,
   `copy_token_acc`; record the LAM code in every mode.
5. `scripts/eval/compare_evals.py`: paired block bootstrap, naive CI, blocks-favouring count, code≠0 subset.
6. `configs/action_ablation/`: `training.yaml`, `dynamics_lam.yaml`, `dynamics_none.yaml`,
   `dynamics_mixed.yaml`.
7. Smoke: CPU copy of `configs/dev/dev_training_cpu.yaml` with the knobs, 2 steps per arm, checkpoint
   reload, eval `--limit 8` in each mode; then delete the `results/<timestamp>/` dirs.

## Results (2026-09-24)

Three concurrent H100 apps, 1 h 40 each at 3.5 it/s (lam `ap-G9h4DdLdbhokmf0ffXAntw`, none `ap-PPRuxgQsc5pqtDRSgNqf9K`,
mixed `ap-YADy6QdO7o2CzjFRZTvFm1`). Checkpoints in `results/action_ablation/<arm>/`, evals in `eval_results/`.

| eval (arm / action mode) | step | token acc | PSNR | SSIM |
|---|---|---|---|---|
| lam / true action   | 10000 | 0.389 | 18.46 | 0.580 |
| lam / true action   | 20000 | 0.383 | 18.57 | 0.583 |
| lam / random action | 20000 | 0.330 | 16.23 | 0.505 |
| none / null         | 10000 | 0.389 | 18.46 | 0.578 |
| none / null         | 20000 | 0.378 | 18.36 | 0.572 |
| mixed / true action | 10000 | 0.393 | 18.69 | 0.587 |
| mixed / true action | 20000 | 0.383 | 18.40 | 0.578 |
| mixed / null        | 20000 | 0.381 | 18.34 | 0.576 |
| mixed / random      | 20000 | 0.362 | 17.44 | 0.549 |
| copy-last baseline  |       | 0.432 | 23.98 | 0.629 |
| tokenizer ceiling   |       |       | 24.83 | 0.826 |
| v2 (no masking fix, 60k steps) / true action | 59000 | 0.293 | 14.44 | 0.392 |

Contrasts, paired token-accuracy difference with block-bootstrap 95% CI (`*` = excludes zero):

| contrast | step 20000 | step 10000 |
|---|---|---|
| 1. lam-true minus none-null (does conditioning help) | +0.005 [+0.001, +0.010] * ; code!=0: +0.012 * | +0.000 [-0.004, +0.005] ; code!=0: +0.009 |
| 2. mixed-true minus mixed-null (action value in one model) | +0.002 [+0.001, +0.004] * ; code!=0: +0.010 * | +0.002 [+0.001, +0.004] * ; code!=0: +0.015 * |
| 3. mixed-true minus lam-true (mix cost, conditioned) | 0.000 [-0.005, +0.005] ; PSNR -0.17 dB * | |
| 4. mixed-null minus none-null (mix cost, unconditioned) | +0.003 [-0.001, +0.007] | |
| 5. lam-true minus lam-random | +0.054 * (PSNR +2.3 dB) | |
| 5. mixed-true minus mixed-random | +0.021 * (PSNR +1.0 dB) | |

Reading:
- **The masking fix is the big effect**, not the actions: every arm at 20k steps sits at 0.38 token accuracy versus
  0.29 for v2 at 60k without it, and 18.4-18.6 dB versus 14.4. All arms still lose to copy-last (0.43 / 24.0 dB).
- **Action conditioning is worth very little with this LAM.** The within-model contrast (2) is the clean number:
  +0.002 token accuracy overall, +0.010 to +0.015 on the 83 windows whose target transition carries a non-zero
  LAM code, significant at both checkpoints. The between-arm contrast (1) agrees in sign but is only significant
  at 20k, where `none` had drifted down slightly more; at 10k it is zero. With one seed per arm, contrast 1 is at
  the noise floor. So the honest statement is: the oracle latent action adds about half a percentage point of
  token accuracy, concentrated on the ~17% of transitions the LAM does not map to code 0.
- **Dropout is free.** `mixed` matches `lam` on true actions (token accuracy identical, 0.17 dB lower PSNR) and
  matches `none` on the null action. It is also far more robust to a wrong action: a random code costs `lam` 0.054
  token accuracy and 2.3 dB, `mixed` only 0.021 and 1.0 dB. If one model has to serve interactive play with
  user-supplied codes, `mixed` is the better default.
- **All arms peak before 20k.** Token accuracy is higher at 10k than at 20k in every arm (0.389-0.393 vs
  0.378-0.383) while training loss keeps falling: mild overfitting on ~140 epochs of 36.8k frames. 10k steps
  would have been enough at this width.
- The learned null action of `none` converged to about (-0.74, +0.84, +0.84), i.e. next to a real FSQ code; in
  `mixed` the null and the codes are distinguishable enough for contrast 2 to be positive.

Next levers, in order: the tokenizer ceiling (24.8 dB) is not the bottleneck yet, copy-last is; so the next
question is why a model that sees three clean frames cannot beat "repeat the last one" (candidates: MaskGIT
decoding vs single-shot argmax, the context-masking mismatch noted above, LAM collapse). A second seed of
arms lam and none would settle whether contrast 1 is real.
