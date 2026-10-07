# configs/

Config layering (`utils/config.py`): dataclass defaults -> the stage yaml -> `training.yaml` on top (shared keys) -> CLI `key=value`.

- `training.yaml`, `video_tokenizer.yaml`, `latent_actions.yaml`, `dynamics.yaml`, `inference.yaml`: the base configs.
- `como/`: CoMo stage configs (`zelda.yaml`, and `tok/` for CoMo on video-tokenizer features).
- `dev/`: tiny CPU smoke profiles (`dev_training_cpu.yaml` is gitignored and copied in by `scripts/infra/new_worktree.sh`).
- `experiments/<run>/`: the exact configs of each past run (a `training.yaml` plus the stage yamls it points to). Launch one with
  `--training-config configs/experiments/<run>/training.yaml`; start a new run by copying the closest folder.
