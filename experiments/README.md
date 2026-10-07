# experiments/

One folder per study that is not part of the training pipeline: the scripts that ran it, its figures and, if it ran on
Modal, a `modal_<name>.py` that adds its functions to the shared app of `scripts/infra/modal_train.py`
(`modal run experiments/<folder>/modal_<name>.py::<function>`). The configs of its runs are in `configs/experiments/`.
Every script stays runnable (`PYTHONPATH=$PWD python experiments/<folder>/<script>.py --help`).

| folder | question | outcome |
|---|---|---|
| `itc_actions/` | can token correspondence between frames (ITC) give actions without training a LAM? Training-free codes, teacher pseudo-labels, and a frame-pair action encoder distilled from them | ITC localises the player but its codes do not encode motion; the pixel-teacher encoder beats the LAM on the judge set |
| `patch_similarity/` | can tokenizer patch similarity / optimal transport find the moving character, and does an OT plan help the LAM? | the OT plan in the LAM decoder beats copy-last, but no arm encodes Link's motion |
| `laof/` | latent actions supervised by optical flow (LAOF): flow targets, training, flow oracle, final report | best arm judge moves NMI_adj 0.53 vs 0.60 for the WTA-warp LAM (all-labels 0.55 vs 0.49) |
| `pusht_nanowm/` | re-run NanoWM's own Push-T eval on Modal so its numbers are comparable with `scripts/eval/eval_pusht.py` | NanoWM 35.10 dB vs ours 34.54 dB at 128 px under the same protocol |
| `como_dynamics/` | figures: held-out PSNR vs training step of dynamics conditioned on CoMo actions (MAE vs per-frame-tokenizer features) | full CoMo z beats copy-last; per-frame-tokenizer actions trail MAE actions by 0.4-0.5 dB |

Adding a study: a new folder named in a word or two after what it tests (no ticket numbers), shared code imported from the
packages (`from experiments.<folder>.<module> import ...` inside the folder), and one row in the table above.
When a study graduates into the pipeline, move its code into the packages and `scripts/`, and delete the folder.
