"""Train a LAOF latent action model (models/laof.py, STA-39) on frame pairs + precomputed flow (scripts/laof_flow.py).

Recipe = LAOF's released stage-1 script (stage1_idm_flowdecoder.py): AdamW, lr 3e-4 piecewise-linear 0.1x -> 1x over
50 steps -> 0.01x at the end, grad-norm clip 2, batch 128, 50k steps. Pairs are (frame i, frame i + gap) of the train
.h5 (gap 4 = ZeldaDataset frame_skip, the transitions lam_judge scores), skipping the first `load_start_index` rows like
ZeldaDataset. Frames and flow are preloaded onto the GPU (~3.2 GB uint8 + ~4.2 GB fp16 for zelda_train), batches are
sampled there: no dataloader.

Every `ckpt_interval` steps: held-out losses on the test .h5 pairs, a visualization PNG (frame t | t+gap | FDM pred |
flow target | flow pred) and the lam_judge score (moves / all NMI_adj on the judged Zelda set) of the checkpoint;
everything goes to W&B and to <run>/latent_actions/judge.jsonl. Checkpoints are directories in the repo format
(<run>/latent_actions/checkpoints/latent_actions_step_N/{model_state_dict,optim_state_dict,state}.pt, state.pt has
model_type 'laof' + model_kwargs, so utils.load_latent_actions_from_checkpoint and lam_judge.py load them).

Usage (repo root, PYTHONPATH=$PWD; real runs on Modal: scripts/modal_train.py::train_laof):
    python scripts/train_laof.py --config configs/laof/discrete.yaml -- n_updates=20 batch_size=8 limit=512
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval'))
from models.laof import LAOF, camera_mask  # noqa: E402

MODEL_KEYS = ['frame_size', 'action_dim', 'continuous', 'num_codebooks', 'num_latents', 'num_embs', 'commitment', 'decay', 'restart_threshold',
              'kl_weight', 'flow_weight', 'flow_sigma', 'impala_scale', 'unet_base', 'eval_clusters', 'flow_target']


def load_pairs(frames_h5, flow_h5, start, limit, device):
    # -> frames uint8 [N, H, W, C], flow fp16 [N, 2, H, W], camera shift [N, 2] on device, valid pair starts [M]
    #    (row i pairs with i + gap)
    with h5py.File(frames_h5, 'r') as f, h5py.File(flow_h5, 'r') as g:
        end = start + limit if limit else len(f['frames'])
        frames = torch.from_numpy(f['frames'][start:end]).to(device)
        flow = torch.from_numpy(g['flow'][start:end]).to(device)
        shift = torch.from_numpy(g['shift'][start:end]).to(device)
        valid = g['valid'][start:end]
        gap = int(g.attrs['gap'])
    valid[len(valid) - gap:] = False  # with `limit`, the partner of the last rows is cut off
    return frames, (flow, shift), torch.from_numpy(np.nonzero(valid)[0]).to(device), gap


MASK = {}  # camera_mask settings (flow_mask_tol, flow_mask_dilate), set from the config in main()


def batch(frames, flow, idx, gap):
    # idx [B] -> frames [B, 2, C, H, W] in [-1, 1], flow [B, 2, H, W] float (zeroed where the camera shift explains
    # the pixel, if flow_mask_tol > 0)
    flow, shift = flow
    x = torch.stack([frames[idx], frames[idx + gap]], 1).permute(0, 1, 4, 2, 3).float() / 127.5 - 1
    fl = flow[idx].float()
    if MASK.get('tol', 0) > 0:
        fl = fl * camera_mask(x[:, 0], x[:, 1], shift[idx], MASK['tol'], MASK['dilate'])[:, None]
    return x, fl


def lr_at(step, cfg):
    # LAOF's doy.PiecewiseLinearSchedule([0, 50, steps + 1], [0.1 lr, lr, 0.01 lr])
    if step < 50:
        return cfg.lr * (0.1 + 0.9 * step / 50)
    return cfg.lr * (1 + (0.01 - 1) * (step - 50) / max(cfg.n_updates + 1 - 50, 1))


@torch.no_grad()
def evaluate(model, frames, flow, starts, gap, n, bs):
    # held-out losses on n fixed pairs (eval mode: no VQ EMA update, mean action)
    model.eval()
    sel = starts[torch.linspace(0, len(starts) - 1, min(n, len(starts)), device=starts.device).long()]
    logs, k = {}, 0
    for s in range(0, len(sel), bs):
        x, fl = batch(frames, flow, sel[s:s + bs], gap)
        _, l, _ = model(x, fl)
        for key, v in l.items():
            logs[key] = logs.get(key, 0) + float(v) * len(x)
        k += len(x)
    model.train()
    return {f'val/{key}': v / k for key, v in logs.items()}


@torch.no_grad()
def save_viz(model, frames, flow, starts, gap, path, n=8):
    from PIL import Image
    model.eval()
    sel = starts[torch.linspace(0, len(starts) - 1, n, device=starts.device).long()]
    x, fl = batch(frames, flow, sel, gap)
    _, _, out = model(x, fl)
    model.train()
    show_flow = out['flow_pred'] is not None and out['flow_pred'].dim() == 4  # 'global' targets are vectors: frames only
    cols = [x[:, 0] / 2, x[:, 1] / 2, out['pred']] + ([out['flow_target'], out['flow_pred']] if show_flow else [])
    img = torch.cat([torch.cat(list(c), 1) for c in cols], 2)  # [C, n*H, cols*W] in [-0.5, 0.5]
    img = ((img + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


@torch.no_grad()
def judge(model, device):
    # lam_judge score of the current weights (no checkpoint round trip); None if the labelled set is absent
    import lam_judge
    if not os.path.exists(f'{lam_judge.SET}/labels.json'):
        return None
    meta = json.load(open(f'{lam_judge.SET}/transitions.json'))
    labels = json.load(open(f'{lam_judge.SET}/labels.json'))
    wins = lam_judge.test_windows(lam_judge.H5, lam_judge.SEQ - 1, lam_judge.SKIP, lam_judge.STRIDE)
    model.eval()
    torch.manual_seed(0)  # k-means init for continuous actions
    codes_all, n_codes = lam_judge.codes_from_lam(model, wins, device)
    model.train()
    res = lam_judge.judged_metrics(meta, labels, n_codes, codes_all=codes_all)
    usage = np.bincount(codes_all.reshape(-1), minlength=n_codes) / codes_all.size
    p = usage[usage > 0]
    return {'judge/moves_nmi_adj': res['moves_only']['nmi_adj'], 'judge/all_nmi_adj': res['all']['nmi_adj'],
            'judge/moves_purity': res['moves_only']['purity'], 'judge/codes_used': int((usage > 0.01).sum()),
            'judge/entropy_nats': float(-(p * np.log(p)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('overrides', nargs='*', help='key=value')
    a = ap.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_dotlist(a.overrides))
    print(OmegaConf.to_yaml(cfg))
    MASK.update(tol=cfg.flow_mask_tol, dilate=cfg.flow_mask_dilate)
    torch.manual_seed(cfg.seed)
    device = cfg.device if cfg.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    run_dir = os.path.join('results', cfg.run_name or time.strftime('laof_%Y_%m_%d_%H_%M_%S'), 'latent_actions')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    viz_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    frames, flow, starts, gap = load_pairs(cfg.train_h5, cfg.train_flow, cfg.load_start_index, cfg.limit, device)
    vframes, vflow, vstarts, vgap = load_pairs(cfg.val_h5, cfg.val_flow, 0, cfg.limit, device)
    assert gap == vgap == cfg.gap, f'flow gap {gap}/{vgap} != config gap {cfg.gap}'
    print(f'{len(starts)} train pairs, {len(vstarts)} val pairs, gap {gap}, frames {tuple(frames.shape)} on {device}')

    model_kwargs = {k: cfg[k] for k in MODEL_KEYS}
    model = LAOF(**model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step0 = 0
    if cfg.checkpoint:
        model.load_state_dict(torch.load(os.path.join(cfg.checkpoint, 'model_state_dict.pt'), map_location=device))
        opt.load_state_dict(torch.load(os.path.join(cfg.checkpoint, 'optim_state_dict.pt'), map_location=device))
        step0 = torch.load(os.path.join(cfg.checkpoint, 'state.pt'), weights_only=False)['step'] + 1
        print(f'resumed from {cfg.checkpoint} at step {step0}')
    print(f'params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M')

    use_wandb = cfg.use_wandb
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=os.path.basename(os.path.dirname(run_dir)), config=OmegaConf.to_container(cfg))

    t0, last = time.time(), {}
    for step in range(step0, cfg.n_updates + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, cfg)
        idx = starts[torch.randint(len(starts), (cfg.batch_size,), device=device)]
        x, fl = batch(frames, flow, idx, gap)
        loss, logs, _ = model(x, fl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_interval == 0:
            last = {f'train/{k}': float(v) for k, v in logs.items()}
            last.update({'train/grad_norm': float(gn), 'train/lr': opt.param_groups[0]['lr'],
                         'it_per_s': (step - step0 + 1) / (time.time() - t0)})
            print(f'step {step} ' + ' '.join(f'{k.split("/")[-1]} {v:.4g}' for k, v in last.items()), flush=True)
            if use_wandb:
                wandb.log(last, step=step)

        if step > 0 and (step % cfg.ckpt_interval == 0 or step == cfg.n_updates):
            ev = evaluate(model, vframes, vflow, vstarts, gap, cfg.val_pairs, cfg.batch_size)
            jd = judge(model, device) or {}
            ev.update(jd)
            print(f'EVAL step {step} ' + ' '.join(f'{k} {v:.4g}' for k, v in ev.items()), flush=True)
            with open(os.path.join(run_dir, 'judge.jsonl'), 'a') as f:
                f.write(json.dumps({'step': step, **ev}) + '\n')
            save_viz(model, vframes, vflow, vstarts, gap, os.path.join(viz_dir, f'laof_step_{step}.png'))
            d = os.path.join(ckpt_dir, f'latent_actions_step_{step}')
            os.makedirs(d, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(d, 'model_state_dict.pt'))
            torch.save(opt.state_dict(), os.path.join(d, 'optim_state_dict.pt'))
            torch.save({'model_type': 'laof', 'model_kwargs': model_kwargs, 'config': OmegaConf.to_container(cfg),
                        'step': step}, os.path.join(d, 'state.pt'))
            if use_wandb:
                import wandb
                wandb.log(ev, step=step)
                wandb.log({'viz': wandb.Image(os.path.join(viz_dir, f'laof_step_{step}.png'))}, step=step)
    print(f'done in {(time.time() - t0) / 60:.1f} min -> {ckpt_dir}')
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
