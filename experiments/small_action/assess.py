"""Link assessment on the judged pairs: the same pairs for every model, scored only around Link.

A 32 px box follows Link's detected position in frame t+4 (set/link_positions.npz, confident in both frames). Groups: camera still
and walking, camera moving and walking, standing. For each model: box LPIPS (the network training terms may use) and box VGG LPIPS
(independent of it) and box PSNR per group, 95% bootstrap intervals, the paired LPIPS differences to frame t, box PSNR with another
pair's action, and with an action of another label (from another block when there is one; the standing group has one label, so
its donors are the walking pairs). Then the action eval (worse camera group), the direction and
shift read-outs. Baselines: the shifted frame, frame t, and CoMo's decoder. Writes scores.json and one sheet per group of Link crops,
3x enlarged: frame t+4 | shifted frame t | CoMo | each model.

Intervals resample contiguous chunks of BOOT_CHUNK frames, not single pairs, since neighbouring pairs show the same walk.

    python experiments/small_action/assess.py --out results/assessment --como <dir> --subset dev --model "name=<ckpt>" --model ...
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from evaluation.action_eval import codes_of, camera_groups, link_scores
from evaluation.image_metrics import try_lpips
from evaluation.link_box import BOX, LPIPS_SIZE
from experiments.small_action.eval import GAP_ROW, SET, TEST_H5, TEST_SHIFTS, action_scores, load_model, subset_rows, to_pixels
from experiments.small_action.probe import probe
from experiments.small_action.shift import shift_frames
from experiments.small_action.visualise import swap_sheet
from utils.utils import load_latent_actions_from_checkpoint

WALKS = {'U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR'}
MIN_CONFIDENCE = 0.5
SHEET_ROWS, CROP_SCALE = 16, 3
BOOT_CHUNK, N_BOOT = 64, 2000  # frames, resamples
VISUAL_SEED = 0
COMO_VECTORS = 'eval_results/action_eval/zelda/como_20min/vectors.npy'


def judged_groups(subset=None):
    # -> {group: rows of pairs.json in the subset}, keeping pairs with Link detected in both frames
    rows = subset_rows(subset)[0]
    kept = json.load(open(f'{SET}/labels.json'))['kept']
    found = np.load(f'{SET}/link_positions.npz')
    yx, conf = found['yx'], found['conf']
    groups = {'still_walking': [], 'moving_walking': [], 'standing': []}
    for r in rows:
        label = kept.get(str(r['id']))
        if label is None or min(conf[r['a']], conf[r['b']]) < MIN_CONFIDENCE:
            continue
        still = max(abs(r['shift'][0]), abs(r['shift'][1])) <= 1
        if label in WALKS:
            groups['still_walking' if still else 'moving_walking'].append(r)
        elif label == 'STANDING':
            groups['standing'].append(r)
    return groups, yx


def top_left(rows, yx):
    # the box centred on Link in frame t+4, inside the frame: [N, 2]
    return np.clip(np.round(yx[[r['b'] for r in rows]] - BOX / 2).astype(int), 0, 128 - BOX)


def crops(images, corners):
    # images [N, C, H, W], corners [N, 2] -> [N, C, BOX, BOX]
    return torch.stack([images[i, :, r:r + BOX, c:c + BOX] for i, (r, c) in enumerate(corners)])


@torch.no_grad()
def predict(fn, x_a, x_b, s, batch=64):
    return torch.cat([fn(x_a[i:i + batch], x_b[i:i + batch], s[i:i + batch]).clamp(-1, 1).cpu() for i in range(0, len(x_a), batch)])


def box_lpips_values(pred_crops, target_crops, lpips_fn, device):
    # crops in [-1, 1] -> LPIPS per pair [N]
    values = []
    up = lambda x: F.interpolate(x, size=LPIPS_SIZE, mode='bilinear', align_corners=False)  # noqa: E731
    with torch.no_grad():
        for i in range(0, len(pred_crops), 64):
            values.append(lpips_fn(up(pred_crops[i:i + 64]).to(device), up(target_crops[i:i + 64]).to(device)).flatten().cpu())
    return torch.cat(values).numpy()


def box_psnr_values(pred_crops, target_crops):
    # crops in [-1, 1] -> PSNR of pixels in [0, 1], per pair [N]
    mse = (pred_crops - target_crops).pow(2).mean((1, 2, 3)) / 4
    return -10 * torch.log10(mse.clamp(min=1e-10))


def box_psnr(pred_crops, target_crops):
    return box_psnr_values(pred_crops, target_crops).mean().item()


def bootstrap_95(values, chunks, n=N_BOOT, seed=0):
    # values [N], chunks [N] -> [low, high] of the mean, resampling whole chunks
    ids, inverse = np.unique(chunks, return_inverse=True)
    totals, counts = np.bincount(inverse, values, len(ids)), np.bincount(inverse, minlength=len(ids))
    drawn = np.random.default_rng(seed).integers(0, len(ids), (n, len(ids)))  # [n, K]
    means = totals[drawn].sum(1) / counts[drawn].sum(1)
    return [round(float(x), 4) for x in np.percentile(means, [2.5, 97.5])]


def load_group(rows, table, device):
    # -> frame t, frame t+4 [N, 3, 128, 128] in [-1, 1], measured shift [N, 2]
    with h5py.File(TEST_H5, 'r') as f:
        x_a = to_pixels(np.stack([f['frames'][r['a']] for r in rows]), device)
        x_b = to_pixels(np.stack([f['frames'][r['b']] for r in rows]), device)
    return x_a, x_b, torch.from_numpy(table[[r['a'] for r in rows]]).to(device)


@torch.no_grad()
def rolled_predict(model, x_a, x_b, s, batch=64):
    # the prediction when each pair gets the action of the next pair of its group
    actions = torch.cat([model.encoder(shift_frames(x_a[i:i + batch], s[i:i + batch]), x_b[i:i + batch]) for i in range(0, len(x_a), batch)]).roll(1, 0)
    out = []
    for i in range(0, len(x_a), batch):
        aligned = shift_frames(x_a[i:i + batch], s[i:i + batch])
        out.append((aligned + model.decoder(aligned, actions[i:i + batch])).clamp(-1, 1).cpu())
    return torch.cat(out)


def other_label_donors(rows, pool, kept, seed=0):
    # -> for each row, the position in `pool` of a pair with another label, from another block when there is one: [N]
    labels = np.array([kept[str(q['id'])] for q in pool], dtype=object)
    blocks = np.array([q['block'] for q in pool])
    rng = np.random.default_rng(seed)
    donors = []
    for r in rows:
        other = labels != kept[str(r['id'])]
        choices = np.flatnonzero(other & (blocks != r['block']))
        donors.append(rng.choice(choices if len(choices) else np.flatnonzero(other)))
    return np.array(donors)


@torch.no_grad()
def swapped_predict(encode, decode, x_a, s, pool, donors, batch=64):
    # the prediction when each pair gets the action of its donor: pool = (frame t, frame t+4, shift) of the donor pairs, donors [N]
    actions = torch.cat([encode(*(x[i:i + batch] for x in pool)) for i in range(0, len(pool[0]), batch)])[torch.from_numpy(donors).to(x_a.device)]
    return torch.cat([decode(x_a[i:i + batch], s[i:i + batch], actions[i:i + batch]).clamp(-1, 1).cpu() for i in range(0, len(x_a), batch)])


def link_assessment(model, device, lpips_fn, subset, limit=0):
    # during training: {link_<group>_lpips, link_<group>_psnr} of one model, flat for the log
    table = np.load(TEST_SHIFTS)['shifts'][:, GAP_ROW].astype(np.float32)
    groups, yx = judged_groups(subset)
    out = {}
    for group, rows in groups.items():
        rows = rows[:limit or None]
        if not rows:
            continue
        x_a, x_b, s = load_group(rows, table, device)
        corners = top_left(rows, yx)
        pred, target = crops(predict(lambda p, q, r: model.predict(p, q, r)[0], x_a, x_b, s), corners), crops(x_b.cpu(), corners)
        out[f'link_{group}_lpips'] = float(box_lpips_values(pred, target, lpips_fn, device).mean())
        out[f'link_{group}_psnr'] = box_psnr(pred, target)
    return out


def sheet(columns, names, path):
    # columns: {name: crops [N, C, BOX, BOX] in [-1, 1]}, evenly spaced rows
    N = next(iter(columns.values())).shape[0]
    picks = np.linspace(0, N - 1, min(SHEET_ROWS, N)).astype(int)
    size = BOX * CROP_SCALE
    canvas = Image.new('RGB', (size * len(names), 16 + size * len(picks)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(names):
        draw.text((c * size + 2, 2), name[:17], fill=(255, 255, 0))
        for r, i in enumerate(picks):
            crop = ((columns[name][i].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(crop).resize((size, size), Image.NEAREST), (c * size, 16 + r * size))
    canvas.save(path)


def visual_sheets(model, device, subset, out):
    # for looking at, the same pairs for every model: one sheet per group of Link crops, frame t | target | prediction | prediction with
    # another pair's action, on VISUAL_SEED's random pairs, and the swap image
    os.makedirs(out, exist_ok=True)
    table = np.load(TEST_SHIFTS)['shifts'][:, GAP_ROW].astype(np.float32)
    groups, yx = judged_groups(subset)
    for group, rows in groups.items():
        picks = np.sort(np.random.default_rng(VISUAL_SEED).choice(len(rows), min(SHEET_ROWS, len(rows)), replace=False))
        rows = [rows[k] for k in picks]
        x_a, x_b, s = load_group(rows, table, device)
        corners = top_left(rows, yx)
        columns = {'frame t': crops(x_a.cpu(), corners), 'target': crops(x_b.cpu(), corners),
                   'prediction': crops(predict(lambda p, q, r: model.predict(p, q, r)[0], x_a, x_b, s), corners), 'other action': crops(rolled_predict(model, x_a, x_b, s), corners)}
        sheet(columns, list(columns), f'{out}/{group}.png')
    swap_sheet(model, f'{out}/swap.png', device, subset)


def action_summary(vectors, subset):
    # vectors [N, A] of the subset -> action eval (worse camera group) and linear read-outs
    rows = subset_rows(subset)[0]
    judged = json.load(open(f'{SET}/labels.json'))
    labels = np.array([judged['kept'].get(str(r['id']), '') for r in rows], dtype=object)
    measured = np.array([r['shift'] for r in rows], dtype=np.float64)
    link = link_scores(codes_of(torch.from_numpy(vectors)), labels, camera_groups(measured), np.array([r['block'] for r in rows]), 5)
    read = probe(vectors.astype(np.float64), subset)
    return {'headline': link['headline'], 'still': link['still'].get('nmi_adj'), 'moving': link['moving'].get('nmi_adj'),
            'direction_still': read['direction_balanced_acc_still'], 'direction_moving': read['direction_balanced_acc_moving'], 'shift_r2': read['shift_all'], 'world_move_r2_still': read['world_move_still']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--como', required=True)
    p.add_argument('--model', action='append', default=[], help='name=<checkpoint dir>')
    p.add_argument('--device', default='cuda')
    p.add_argument('--subset', default='all', choices=['all', 'dev', 'confirm'])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device, lpips_fn, vgg_fn = a.device, try_lpips(a.device), try_lpips(a.device, 'vgg')
    assert vgg_fn is not None, 'the VGG score needs its weights'
    table = np.load(TEST_SHIFTS)['shifts'][:, GAP_ROW].astype(np.float32)
    como, _ = load_latent_actions_from_checkpoint(a.como, device)
    como.eval()

    def como_predict(x_a, x_b, s):
        z = como.encode(torch.stack([x_a, x_b], 1))  # [B, 1, A]
        return como.como.decoder(x_a, z.reshape(z.shape[0], como.como.idm.n_queries, -1))

    def como_encode(x_a, x_b, s):
        return como.encode(torch.stack([x_a, x_b], 1))  # [B, 1, A]

    def como_decode(x_a, s, z):
        return como.como.decoder(x_a, z.reshape(z.shape[0], como.como.idm.n_queries, -1))

    predictors, models = {'frame t': lambda x_a, x_b, s: x_a, 'shifted frame t': lambda x_a, x_b, s: shift_frames(x_a, s), 'CoMo': como_predict}, {}
    swappable = {'CoMo': (como_encode, como_decode)}  # action from a pair (frame t, frame t+4, shift), prediction from (frame t, shift, action)
    for spec in a.model:
        name, ckpt = spec.split('=', 1)
        models[name], _ = load_model(ckpt, device)
        predictors[name] = lambda x_a, x_b, s, m=models[name]: m.predict(x_a, x_b, s)[0]
        swappable[name] = (lambda x_a, x_b, s, m=models[name]: m.encoder(shift_frames(x_a, s), x_b),
                           lambda x_a, s, z, m=models[name]: (lambda aligned: aligned + m.decoder(aligned, z))(shift_frames(x_a, s)))
    groups, yx = judged_groups(a.subset)
    kept = json.load(open(f'{SET}/labels.json'))['kept']
    results = {'subset': a.subset, 'other_label_donors': {}}
    for group, rows in groups.items():
        x_a, x_b, s = load_group(rows, table, device)
        corners = top_left(rows, yx)
        target = crops(x_b.cpu(), corners)
        chunks = np.array([r['a'] // BOOT_CHUNK for r in rows])
        columns = {'frame t+4': target}
        results[group], lpips_of, vgg_of = {}, {}, {}
        several_labels = len({kept[str(r['id'])] for r in rows}) > 1
        pool_rows = rows if several_labels else [q for g, other in groups.items() if g != group for q in other]
        results['other_label_donors'][group] = 'same group' if several_labels else 'walking groups'
        donors, pool = other_label_donors(rows, pool_rows, kept), load_group(pool_rows, table, device)
        for name, fn in predictors.items():
            columns[name] = crops(predict(fn, x_a, x_b, s), corners)
            lpips_of[name] = box_lpips_values(columns[name], target, lpips_fn, device)
            vgg_of[name] = box_lpips_values(columns[name], target, vgg_fn, device)
            results[group][name] = {'box_lpips': round(float(lpips_of[name].mean()), 4), 'box_lpips_95': bootstrap_95(lpips_of[name], chunks),
                                    'minus_frame_t_95': bootstrap_95(lpips_of[name] - lpips_of['frame t'], chunks), 'box_psnr': round(box_psnr(columns[name], target), 2), 'n': len(rows),
                                    'box_vgg_lpips': round(float(vgg_of[name].mean()), 4), 'box_vgg_lpips_95': bootstrap_95(vgg_of[name], chunks), 'vgg_minus_frame_t_95': bootstrap_95(vgg_of[name] - vgg_of['frame t'], chunks)}
            if name in models:
                other = box_psnr(crops(rolled_predict(models[name], x_a, x_b, s), corners), target)
                results[group][name].update(box_psnr_other_action=round(other, 2), box_psnr_gain=round(results[group][name]['box_psnr'] - other, 2))
            if name in swappable:
                swapped = box_psnr_values(crops(swapped_predict(*swappable[name], x_a, s, pool, donors), corners), target)
                gain = (box_psnr_values(columns[name], target) - swapped).numpy()
                results[group][name].update(box_psnr_other_label=round(swapped.mean().item(), 2), other_label_gain=round(float(gain.mean()), 2), other_label_gain_95=bootstrap_95(gain, chunks))
        names = ['frame t+4'] + [n for n in predictors if n != 'frame t']
        if a.subset != 'confirm':  # no images of the confirm half
            sheet(columns, names, f'{a.out}/{group}.png')
    results['action'] = {'CoMo': action_summary(np.load(COMO_VECTORS)[subset_rows(a.subset)[1]], a.subset)}
    for name, model in models.items():
        if a.subset != 'confirm':
            visual_sheets(model, device, a.subset, f'{a.out}/{name}')
        results['action'][name] = action_summary(action_scores(model, device, subset=a.subset)['actions'], a.subset)
    json.dump(results, open(f'{a.out}/scores.json', 'w'), indent=1)
    print('ASSESS ' + json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
