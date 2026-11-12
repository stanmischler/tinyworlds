"""Train a small action model (experiments/small_action/model.py) on the 20 min Zelda capture or the 5 h mix (five_hours.py).

Each sample is a pair (t, t+4) inside one stretch of the .h5, with a measured shift from a table made beforehand
(experiments/small_action/shift.py), plus one frame later or earlier than t+4 for the InfoNCE positive (on the 5 h mix a YouTube pair is t, t+2). Frames and shifts sit on the GPU,
so a step is a gather + the model; no dataloader. Pairs among the worst 5% by error after the shift (room cuts, dialog
changes) are not sampled. Every ckpt_interval steps: a checkpoint, the held-out scores (eval.py) and a sheet of predictions.

    python experiments/small_action/train.py --config configs/experiments/small_action/base.yaml,configs/experiments/small_action/small_action_4.yaml
"""

import argparse
import contextlib
import json
import os
import time

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from datasets.split import segments
from evaluation.image_metrics import try_lpips
from experiments.small_action.action_pca import still_walk_picture
from experiments.small_action.assess import link_assessment, visual_sheets
from experiments.small_action.eval import GAP_ROW, TEST_H5, evaluate, model_kwargs, subset_windows, to_pixels
from experiments.small_action.model import WHOLE, WINDOW, SmallActionModel
from experiments.small_action.probe import probe
from experiments.small_action.shift import SIZE, shift_frames

GAP = 4


def trainable_starts(cfg, errors, sources=None):
    # -> (t such that t .. t + gap + 1 stay inside one stretch, past start_index, and the pair is not among the worst by error;
    #     all those t before the drop; the error limit). sources [N], a table of several files: the worst are dropped inside each file, limit = the largest
    with h5py.File(cfg.train_h5, 'r') as f:
        n_frames = len(f['frames'])
    span = GAP + 1
    starts = np.concatenate([np.arange(max(s, cfg.start_index), e - span) for s, e in segments(cfg.train_h5, n_frames) if e - span > max(s, cfg.start_index)])
    if sources is not None:
        limits = {k: np.quantile(errors[starts[sources[starts] == k], GAP_ROW], 1 - cfg.drop_worst) for k in np.unique(sources[starts])}
        kept = np.concatenate([starts[(sources[starts] == k) & (errors[starts, GAP_ROW] <= limit)] for k, limit in limits.items()])
        return np.sort(kept), starts, max(limits.values())
    limit = np.quantile(errors[starts, GAP_ROW], 1 - cfg.drop_worst)
    return starts[errors[starts, GAP_ROW] <= limit], starts, limit


def drawable_starts(cfg, table):
    # table: the shift table (npz) -> (the rows t a batch is drawn from, a row weighing k being k times in it; every usable t before any drop;
    # the error limit; how many distinct pairs are kept)
    several_files = 'pair_gap' in table.files
    starts, all_starts, limit = trainable_starts(cfg, table['errors'], table['source'] if several_files else None)
    if cfg.get('drop_dialog'):  # no pair with a dialog box or menu on screen (flags of experiments/small_action/five_hours.py, in the motion table)
        motion = np.load(cfg.still_motion_table)
        assert 'dialog' in motion.files, f'{cfg.still_motion_table} holds no dialog flags: rebuild it with experiments/small_action/five_hours.py --stage tables'
        print(f'dialog: {motion["dialog"][starts].sum()} of {len(starts)} kept pairs left out', flush=True)
        starts = starts[~motion['dialog'][starts]]
    n_kept = len(starts)
    if several_files:
        weights = table['row_weight'][starts]
        if cfg.get('our_share'):  # the share of draws from our capture (source 0), as near as a whole weight on its rows gets, instead of equal video seconds
            ours = table['source'][starts] == 0
            weights = np.where(ours, round(cfg.our_share / (1 - cfg.our_share) * weights[~ours].sum(dtype=np.int64) / ours.sum()), weights)
            print(f'our capture: {weights[ours].sum() / weights.sum():.4f} of draws, weight {weights[ours][0]} per row', flush=True)
        starts = np.repeat(starts, weights)  # a row weighing k is k times in the draw
    return starts, all_starts, limit, n_kept


def batch_pools(cfg, starts, shifts):
    # -> [(name, pool of pairs, its share of the batch)]; the last pool holds every pair the others left, and takes the rest of the batch.
    #   focus (still_fraction): camera still, by |s| < 1 px alone; still_motion_table narrows it to the pairs whose changed pixels form a
    #     compact patch (experiments/small_action/five_hours.py motion_table; pixels only), most often Link walking
    #   idle (idle_fraction): camera still and fewer than idle_max_px changed pixels: nothing happens. Uniform draws hold many of them;
    #     this sets their share, so they keep teaching "no action" without taking the steps
    still = np.abs(shifts[starts, GAP_ROW]).max(1) < 1
    motion = np.load(cfg.still_motion_table) if cfg.get('still_motion_table') else None
    pools, left = [], np.ones(len(starts), bool)
    if cfg.get('still_fraction') is not None:
        focus = still.copy()
        if motion is not None:
            focus &= (motion['moving_px'][starts] >= cfg.still_min_px) & (motion['moving_px'][starts] <= cfg.still_max_px) & (motion['peak_share'][starts] >= cfg.still_min_share)
        pools.append(('focus', starts[focus], cfg.still_fraction))
        left &= ~focus
    if cfg.get('idle_fraction') is not None:
        assert motion is not None, 'idle_fraction reads the changed pixels of still_motion_table'
        idle = left & still & (motion['moving_px'][starts] < cfg.idle_max_px)
        pools.append(('idle', starts[idle], cfg.idle_fraction))
        left &= ~idle
    pools.append(('others', starts[left], 1 - sum(share for _, _, share in pools)))
    assert all(len(pool) > 0 and share >= 0 for _, pool, share in pools), [(name, len(pool), share) for name, pool, share in pools]
    return pools


def pick_starts(starts, pools, cfg, gen, size):
    # -> frame rows t [size]: uniform, or each pool's share of them, the last pool filling the batch
    draw = lambda pool, n: pool[torch.randint(0, len(pool), (n,), generator=gen).numpy()]  # noqa: E731
    if pools is None:
        return draw(starts, size)
    counts = [round(share * size) for _, _, share in pools[:-1]]
    return np.concatenate([draw(pool, n) for (_, pool, _), n in zip(pools, [*counts, size - sum(counts)])])


def sample_batch(frames, shifts, starts, pools, cfg, gen, device, size, pair_gaps=None):
    # -> frames t, t+n, t+n+delta [size, 3, 128, 128] in [-1, 1] and the shifts from t to the last two [size, 2]; n = 4, or pair_gaps [N]: a pair's stored frames per row
    t = torch.as_tensor(pick_starts(starts, pools, cfg, gen, size), device=device)
    delta = torch.randint(0, 2, (size,), generator=gen).to(device) * 2 - 1  # -1 or +1
    b = t + GAP if pair_gaps is None else t + pair_gaps[t]
    j = b + delta
    s_b = shifts[t, GAP_ROW]
    s_j = shifts[t, delta + 1]  # delta -1 -> row 0 (one frame less than the pair), +1 -> row 2 (one frame more)
    x = [frames[i].permute(0, 3, 1, 2).float() / 127.5 - 1 for i in (t, b, j)]
    return (*x, s_b, s_j)


def window_jitter(s_b, s_j, fraction, gen):
    # -> the jitter (rows, cols) of each pair's window [B, 2] long, or WHOLE where the pair stays whole (model.py window_corner: the window
    # follows the shift, then moves by -jitter). Each pair is cut with probability `fraction`, if it has room. The jitter is uniform in
    # [-h, h] on both axes, h = a value (at most 1 px under the largest) that keeps the window inside the frame and clear of every pixel the shift reveals or
    # blends (the strip, and one more line when the shift is not whole), for the pair, its contrastive positive (s_j) and the reversed pair:
    #   h = min over those shifts and both axes of  min(16 - |round s| - f,  16 + |round s| - ceil|s| - f),   f = 1 if s is not whole
    # One h for both axes, so the spread of the jitter says how fast the camera moves but not in which direction
    margin = (SIZE - WINDOW) // 2
    shifts = torch.stack([s_b, s_j]).abs().cpu()  # [2, B, 2]; the reversed pair has the size of s_b
    whole, fractional = shifts.round(), (shifts != shifts.round()).float()
    h = torch.minimum(margin - whole - fractional, margin + whole - shifts.ceil() - fractional).amin((0, 2)).long()  # [B]
    cut = (torch.rand(len(h), generator=gen) < fraction) & (h >= 0)
    jitter = (torch.rand(len(h), 2, generator=gen) * (2 * h[:, None] + 1).clamp(min=1)).long() - h[:, None].clamp(min=0)
    return torch.where(cut[:, None], jitter, torch.full_like(jitter, WHOLE)).to(s_b.device)


def encoder_gradient_norms(model, out, contrastive_weight):
    # -> {enc_grad_recon: |d recon / d encoder|, enc_grad_nce: |d (weight x InfoNCE) / d encoder|, enc_grad_ratio: the second over the first,
    #     enc_grad_edge: |d (weight x edge loss) / d encoder| when that term is on}
    # on this batch: how hard each term pulls on the action encoder. A ratio far above 1 is the failure where the decoder stops reading the
    # action. Without InfoNCE only the first; no ratio while the reconstruction does not pull yet (the decoder's head starts at zero)
    params = list(model.encoder.parameters())

    def norm(loss):
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.cat([g.flatten() for g in grads if g is not None]).float().norm().item()
    pulls = {'enc_grad_recon': norm(out['recon'])}
    if model.edge_weight > 0:
        pulls['enc_grad_edge'] = norm(model.edge_weight * out['edge'])
    if contrastive_weight > 0:
        pulls['enc_grad_nce'] = norm(contrastive_weight * out['nce'])
        if pulls['enc_grad_recon'] > 0:
            pulls['enc_grad_ratio'] = pulls['enc_grad_nce'] / pulls['enc_grad_recon']
    return pulls


def checkpoint_scored(log_path, step):
    # is there an EVAL record of this step in the run's log?
    with open(log_path) as f:
        return any(rec['step'] == step and 'link_headline' in rec for rec in map(json.loads, f))


@torch.no_grad()
def prediction_sheet(path, model, device, n=8, subset=None):
    # rows: frame t | aligned frame t | prediction | frame t+4 | |prediction - frame t+4| x 3, on evenly spaced held-out windows
    windows = subset_windows(subset)
    picks = windows[:: max(1, len(windows) // n)][:n]
    table = np.load('data/zelda_test_shifts.npz')['shifts'][:, GAP_ROW].astype(np.float32)
    with h5py.File(TEST_H5, 'r') as f:
        x_a = to_pixels(np.stack([f['frames'][s + 8] for _, s in picks]), device)
        x_b = to_pixels(np.stack([f['frames'][s + 12] for _, s in picks]), device)
    s = torch.from_numpy(table[[s + 8 for _, s in picks]]).to(device)
    model.eval()
    pred, _ = model.predict(x_a, x_b, s)
    model.train()
    pred = pred.clamp(-1, 1)
    error = ((pred - x_b).abs() * 3 - 1).clamp(-1, 1)
    rows = torch.cat([torch.cat(c, 2) for c in zip(x_a, shift_frames(x_a, s), pred, x_b, error)], 1)  # [C, nH, 5W]
    Image.fromarray(((rows + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='yaml, or comma-separated yamls merged left to right')
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    cfg = OmegaConf.merge(*[OmegaConf.load(c) for c in a.config.split(',')], OmegaConf.from_dotlist([o for o in a.overrides if o != '--']))
    device = cfg.device
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = f'results/{cfg.run_name}/small_action'
    ckpt_dir, viz_dir = f'{run_dir}/checkpoints', f'{run_dir}/visualizations'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    with h5py.File(cfg.train_h5, 'r') as f:
        frames = torch.from_numpy(f['frames'][:]).to(device)  # uint8 [N, H, W, C]
    table = np.load(cfg.train_shifts)
    shifts = torch.from_numpy(np.nan_to_num(table['shifts'].astype(np.float32))).to(device)  # [N, 3, 2]
    assert len(table['shifts']) == len(frames) and tuple(table['gaps']) == (3, 4, 5)  # the rows' gaps on our capture; 1, 2, 3 on YouTube rows
    several_files = 'pair_gap' in table.files  # experiments/small_action/five_hours.py: pair gap, file and draw weight per row
    assert several_files or not cfg.get('our_share'), 'our_share needs a table of several files'
    pair_gaps = torch.from_numpy(table['pair_gap'].astype(np.int64)).to(device) if several_files else None
    starts, all_starts, limit, n_kept = drawable_starts(cfg, table)
    # InfoNCE reads the shifts to its positives in rows 0 and 2; on the YouTube rows experiments/small_action/positive_shifts.py measures them
    assert not several_files or cfg.contrastive_weight == 0 or not np.isnan(table['shifts'][starts][:, [0, 2]]).any(), \
        f'{cfg.train_shifts} holds no shift for the InfoNCE positives of some drawable rows: run experiments/small_action/positive_shifts.py'
    n_pairs = n_kept, len(all_starts)
    # memorisation is measured on all pairs, dropped ones included, as the held-out windows include theirs;
    # memorisation_h5 set: on that file's pairs, the same pairs as a run trained on it alone
    memory_h5, memory_shifts = cfg.get('memorisation_h5') or cfg.train_h5, cfg.get('memorisation_shifts') or cfg.train_shifts
    if cfg.get('memorisation_h5'):
        all_starts = trainable_starts(OmegaConf.merge(cfg, {'train_h5': memory_h5}), np.load(memory_shifts)['errors'])[1]
    train_pairs = np.sort(np.random.default_rng(0).choice(all_starts, 890, replace=False))
    print(f'{len(frames)} frames; {n_pairs[0]} of {n_pairs[1]} pairs kept, MSE after the shift <= {limit:.5f}', flush=True)
    pools = batch_pools(cfg, starts, table['shifts']) if cfg.get('still_fraction') is not None or cfg.get('idle_fraction') is not None else None
    if pools is not None:
        print('batch: ' + ', '.join(f'{share:.2f} from {len(pool)} {name} pairs' for name, pool, share in pools), flush=True)
        if several_files:  # the sampler reweighs the capture's share, whatever our_share gave the whole draw
            ours = [(table['source'][pool] == 0).mean() for _, pool, _ in pools]
            print(f'our capture: {sum(share * o for (_, _, share), o in zip(pools, ours)):.4f} of draws; per pool ' + ', '.join(f'{o:.4f}' for o in ours), flush=True)
    subset = cfg.get('eval_subset')  # scores during training stay on this subset of the held-out data; None = all

    model = SmallActionModel(**model_kwargs(cfg)).to(device)
    count = lambda m: sum(p.numel() for p in m.parameters()) / 1e6  # noqa: E731
    print(f'params: action encoder {count(model.encoder):.2f}M, decoder {count(model.decoder):.2f}M, shift predictor {count(model.shift_predictor):.2f}M', flush=True)
    # the shift predictor has its own optimiser and clip, so its gradients never scale those of the action encoder and decoder
    action_params = [*model.encoder.parameters(), *model.decoder.parameters()]
    shift_params = list(model.shift_predictor.parameters())
    opt = torch.optim.AdamW(action_params, lr=cfg.learning_rate, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)
    opt_shift = torch.optim.AdamW(shift_params, lr=cfg.learning_rate, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)
    # init_from: weights of another run's checkpoint (results/.../step_N), fresh optimisers, counting on from step N, the learning rate where this config's schedule has it at N
    init_step = 0
    if cfg.get('init_from'):
        model.load_state_dict(torch.load(f'{cfg.init_from}/model_state_dict.pt', map_location=device, weights_only=True))
        init_step = torch.load(f'{cfg.init_from}/state.pt', map_location='cpu', weights_only=False)['step'] + 1
    schedule = lambda s: max(cfg.lr_floor, 1 - (s + init_step) / cfg.n_updates)  # noqa: E731
    sched, sched_shift = (torch.optim.lr_scheduler.LambdaLR(o, schedule) for o in (opt, opt_shift))

    step0 = init_step
    if init_step:
        gen.manual_seed(cfg.seed + init_step)
    done = [int(d.rsplit('_', 1)[1]) for d in os.listdir(ckpt_dir) if os.path.exists(f'{ckpt_dir}/{d}/state.pt')]
    if done:  # relaunching the same command continues the run
        ck = f'{ckpt_dir}/step_{max(done)}'
        model.load_state_dict(torch.load(f'{ck}/model_state_dict.pt', map_location=device, weights_only=True))
        state = torch.load(f'{ck}/state.pt', map_location='cpu', weights_only=False)
        opt.load_state_dict(state['opt']), opt_shift.load_state_dict(state['opt_shift'])
        sched.load_state_dict(state['sched']), sched_shift.load_state_dict(state['sched_shift'])
        step0 = state['step'] + 1
        gen.manual_seed(cfg.seed + step0)
        print(f'resumed at step {step0}', flush=True)

    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if cfg.amp and device.startswith('cuda') else contextlib.nullcontext
    wb = None
    if cfg.use_wandb:
        import wandb
        wb = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=OmegaConf.to_container(cfg), resume='allow', id=cfg.run_name)
    log = open(f'{run_dir}/log.jsonl', 'a')
    lpips_fn = try_lpips(device)
    assert lpips_fn is not None, 'the Link score needs the lpips package and its AlexNet weights (cached under TORCH_HOME; no download on an offline node)'
    lpips_fn.requires_grad_(False)  # the training term and the scorer are the same network (AlexNet LPIPS)
    model.__dict__['lpips_fn'] = lpips_fn

    def score_checkpoint(step):
        # the held-out scores, sheets and pictures of the weights now in the model, as an EVAL line of the log
        flat, action = evaluate(model, device, lpips_fn, cfg.get('eval_limit', 0), train_pairs[:cfg.get('eval_limit', 0) or None], memory_h5, memory_shifts, subset=subset)
        if subset:  # Link's box per group, as assess.py scores it
            model.eval()
            flat.update(link_assessment(model, device, lpips_fn, subset, cfg.get('eval_limit', 0)))
            if not cfg.get('eval_limit'):  # read-outs and the picture need every pair of the subset
                read = probe(action['actions'].astype(np.float64), subset)
                flat.update({'direction_still': read['direction_balanced_acc_still'], 'direction_moving': read['direction_balanced_acc_moving'], 'shift_r2': read['shift_all']})
                flat['pca_direction_still'] = still_walk_picture(action['actions'], subset, f'{viz_dir}/pca_step_{step}.png', f'{cfg.run_name} step {step}')
            if subset != 'confirm':
                visual_sheets(model, device, subset, f'{viz_dir}/step_{step}')
            model.train()
        prediction_sheet(f'{viz_dir}/pred_step_{step}.png', model, device, subset=subset)
        rec = {'step': step, **flat}
        print('EVAL ' + json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + '\n')
        log.flush()
        if wb:
            wb.log({k: v for k, v in flat.items() if v is not None}, step=step)

    if done and not checkpoint_scored(f'{run_dir}/log.jsonl', max(done)):  # the run stopped between saving that checkpoint and scoring it
        score_checkpoint(max(done))
    tick = time.time()
    window_gen = torch.Generator().manual_seed(cfg.seed + step0 + 10 ** 6)  # its own stream: a run with windows draws the same batches as one without
    model.train()
    for step in range(step0, cfg.n_updates + 1):
        opt.zero_grad(set_to_none=True)
        opt_shift.zero_grad(set_to_none=True)
        batch = sample_batch(frames, shifts, starts, pools, cfg, gen, device, cfg.batch_size, pair_gaps)  # x_t, x_b, x_j [B, 3, 128, 128], s_b, s_j [B, 2]
        window = window_jitter(batch[3], batch[4], cfg.window_fraction, window_gen) if cfg.get('window_fraction') else None  # None: every pair whole
        with amp():
            out = model(*batch, window=window)
        pulls = encoder_gradient_norms(model, out, cfg.contrastive_weight) if step % cfg.log_interval == 0 else {}
        (out['loss'] + out['shift_error']).backward()  # the shift predictor reads a.detach(): its loss reaches only its own weights
        grad_norm = torch.nn.utils.clip_grad_norm_(action_params, cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(shift_params, cfg.grad_clip)
        opt.step(), opt_shift.step()
        sched.step(), sched_shift.step()
        if step % cfg.log_interval == 0:
            rec = {'step': step, 'recon': out['recon'].item(), 'nce': out['nce'].item(), 'kl_bits': out['kl'].item() / 0.6931, 'shift_error': out['shift_error'].item(),
                   'grad_norm': grad_norm.item(), 'a_std': out['a'].float().std(0).mean().item(), 'lr': sched.get_last_lr()[0],
                   'it_s': cfg.log_interval / max(time.time() - tick, 1e-6) if step > step0 else 0.0}
            rec.update(pulls)
            if model.edge_weight > 0:
                rec['edge'] = out['edge'].item()
            if window is not None:
                rec['window_share'] = (window[:, 0] != WHOLE).float().mean().item()
            if device.startswith('cuda'):
                rec['mem_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
            tick = time.time()
            print(' '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in rec.items()), flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()
            if wb:
                wb.log(rec, step=step)
        if step > 0 and (step % cfg.ckpt_interval == 0 or step == cfg.n_updates):
            ck = f'{ckpt_dir}/step_{step}'
            os.makedirs(ck, exist_ok=True)
            torch.save(model.state_dict(), f'{ck}/model_state_dict.pt')
            torch.save({'config': OmegaConf.to_container(cfg), 'step': step, 'opt': opt.state_dict(), 'opt_shift': opt_shift.state_dict(),
                        'sched': sched.state_dict(), 'sched_shift': sched_shift.state_dict()}, f'{ck}/state.pt.saving')
            os.replace(f'{ck}/state.pt.saving', f'{ck}/state.pt')  # state.pt appears only once complete: a job killed mid-save resumes from the checkpoint before
            score_checkpoint(step)
            tick = time.time()
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
