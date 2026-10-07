# scripts/

Everything here is an entry point, run from the repo root with `PYTHONPATH=$PWD`. Folders follow the pipeline order.
Scripts import only from the packages (`models`, `datasets`, `evaluation`, `utils`), never from each other; the pipeline
and Modal launchers call them by path.

| folder | script | what it does |
|---|---|---|
| `data/` | `download_assets.py` | download datasets (`.h5`) and pretrained checkpoints from the HuggingFace Hub |
| | `convert_pusht.py` | convert DINO-WM's Push-T episodes into tinyworlds `.h5` files |
| | `split_dataset.py` | deterministic train/test split of a `<game>_frames.h5` (held-out blocks + margins) |
| | `visualize_batch.py` | plot DataLoader batches, to check a dataset |
| `tokenizer/` | `train_video_tokenizer.py` | stage 1: FSQ video tokenizer (frames -> discrete tokens) |
| `actions/` | `train_latent_actions.py` | stage 2: latent action model (infers a discrete action between frames) |
| | `como_features.py` | precompute frozen MAE ViT-L features of every frame, input of CoMo |
| | `tok_features.py` | precompute frozen video-tokenizer features of every frame, alternative CoMo input |
| | `train_como.py` | stage 2 (CoMo variant): motion inverse-dynamics model on the precomputed features |
| | `como_actions.py` | run a trained CoMo over every frame pair and write the action files dynamics training reads |
| `dynamics/` | `train_dynamics.py` | stage 3: dynamics model (MaskGIT or flow matching), conditioned on the actions |
| `pipeline/` | `full_train.py` | run stages 1-3 in sequence (subprocesses, `torchrun` when multi-GPU) |
| `inference/` | `run_inference.py` | autoregressive / interactive play with the three trained models |
| `eval/` | `eval_next_frame.py` | dynamics: held-out one-step next-frame PSNR/SSIM/LPIPS/token accuracy vs copy-last and tokenizer recon |
| | `compare_evals.py` | paired bootstrap difference between two `eval_next_frame` results |
| | `eval_pusht.py` | dynamics on Push-T with NanoWM's protocol |
| | `tokenizer_frame_eval.py` | tokenizer: reconstruction and token stability over time |
| | `tokenizer_psnr_vs_step.py` | tokenizer: reconstruction PSNR across training checkpoints |
| | `eval_lam.py` | action model: code collapse, NMI with motion, decoder shuffle gap |
| | `lam_judge.py` | action model: build / score the judge-labelled held-out Zelda action set |
| | `lam_eval.py` | action model: judge-labelled action groups for any game (zelda, sonic, pong) |
| | `eval_como_pred.py` | CoMo: next-frame PSNR of its own decoder, as a reference for dynamics |
| `infra/` | `modal_train.py` | Modal app (image, volumes) and the GPU functions of the pipeline: train, download, evals, CoMo stages |
| | `modal_check.py` / `modal_wait.py` | health check of a Modal app + its W&B runs / wait until apps stop |
| | `new_worktree.sh` / `rc.sh` | create a git worktree wired to the shared venv and data / run a remote-control session in tmux |
