# cluster/

## Jean Zay (`jeanzay/`)

IDRIS's national GPU cluster, used through the project `hsq` (A100 and H100 hours, counted with `idracct`).
It is reached only through the CMAP gateway, so `~/.ssh/config` needs:

```
Host cmap
  HostName ssh.cmap.polytechnique.fr
  User <cmap login>
  IdentityFile ~/.ssh/id_ed25519
Host jeanzay
  HostName jean-zay.idris.fr
  User <idris login>
  ProxyJump cmap
```

Jobs go through the Slurm queue, and **GPU nodes have no internet**: data, checkpoints and pretrained
weights must be in place before a job starts. Remote commands need a login shell (`bash -lc`), or `$WORK`,
`module` and `sbatch` are missing.

```bash
bash cluster/jeanzay/sync.sh                 # this working tree → $WORK/runs/<folder>, inputs linked in
ssh jeanzay 'bash -lc "cd \$WORK/runs/<folder> && bash cluster/jeanzay/setup_env.sh"'   # once: venvs + pretrained weights
ssh jeanzay 'bash -lc "cd \$WORK/runs/<folder> && sbatch cluster/jeanzay/train.slurm scripts/dynamics/train_dynamics.py \
    --config <dynamics yaml> --training_config <training yaml> -- key=val"'
bash cluster/jeanzay/status.sh <folder>      # queue, recent jobs, last log lines, GPU-hours
```

Shared inputs, one copy for all runs (`sync.sh` uploads what is missing or changed and links them in):

| on Jean Zay | in a run folder | what |
|---|---|---|
| `$WORK/tinyworlds_data/` | `data/` (symlink) | `.h5` frames, `.npy` actions; `DATA_FILES="…"` picks which |
| `$WORK/tinyworlds_results/<x>` | `results/<x>` (symlink per entry) | input checkpoints; upload with `INPUTS=<local dir laid out like results/>` |
| `$WORK/hf_home`, `$WORK/torch_home` | | HF and torchvision weights, filled by `setup_env.sh` |

`results/` itself is a real folder per run, so new checkpoints land there. `train.slurm` runs `python <args>`
on one GPU (A100 by default, 20 h max) with W&B offline (`wandb sync` from the login node), the HF hub
offline and `PYTHONPATH` set; logs go to `logs/`. Its header lists the variants: test queue
(`--qos=qos_gpu_a100-dev`, 2 h max), H100, sweeps as job arrays (`--lines sweep.txt`) and chained jobs
(`--dependency=afterany:<id>`) to resume past 20 h. Copy results back with
`rsync -rltz jeanzay:/lustre/fswork/projects/rech/hsq/<idris login>/runs/<folder>/results/<run> results/`
(the full path: `$WORK` is unset in rsync's shell).
