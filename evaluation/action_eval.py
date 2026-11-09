"""Movement-vs-camera action eval: each 2-frame test transition has a computed camera shift and a judged Link action.

Judges see frame t with the shift removed, so Link's displacement there is his movement in the world, the button pressed,
whatever the camera does. Pairs are grouped by the shift (camera still or moving). An action vector is scored twice:
k-means over all pairs, NMI_adj against the judge labels in each camera group, the worse group being the headline; and how
well the shift is read from it. A code given to one button under one camera and another under the other camera is NOT
penalised: the scores are per group."""

import numpy as np
import torch
from PIL import Image, ImageDraw

from evaluation.action_metrics import kmeans, nmi

DIRECTIONS = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR']
SCORED = DIRECTIONS + ['STANDING', 'TURN']  # judge labels entering the Link score
LABELS = SCORED + ['ATTACK', 'INTERACT', 'UNCLEAR']
UNCLEAR_REASONS = ['TEXT', 'FADE_OR_CUT', 'NO_LINK', 'MENU', 'BAD_ALIGNMENT']
STILL_PX = 1  # a pair is "camera still" when its largest shift component is at most this
N_CODES = 16
ZOOM, ZOOM_SCALE, FRAME_PX = 48, 5, 240
GROUPS = ('still', 'moving')


def camera_groups(shift):
    # shift: [N, 2] -> array [N] of 'still' / 'moving'
    return np.where(np.abs(shift).max(1) <= STILL_PX, 'still', 'moving')


def align(frame, dy, dx):
    # frame t moved by the shift so the scenery sits where it sits in t+1; pixels with no source are grey
    H, W = frame.shape[:2]
    out = np.full_like(frame, 128)
    y0, y1, x0, x1 = max(dy, 0), H + min(dy, 0), max(dx, 0), W + min(dx, 0)
    out[y0:y1, x0:x1] = frame[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def edge_ticks(img, step, length, colour=(255, 255, 0)):
    draw, (W, H) = ImageDraw.Draw(img), img.size
    for p in range(step, W, step):
        draw.line([(p, 0), (p, length)], fill=colour)
        draw.line([(p, H - 1), (p, H - 1 - length)], fill=colour)
    for p in range(step, H, step):
        draw.line([(0, p), (length, p)], fill=colour)
        draw.line([(W - 1, p), (W - 1 - length, p)], fill=colour)
    return img


def zoom(frame, cy, cx):
    H, W = frame.shape[:2]
    y0, x0 = int(np.clip(round(cy) - ZOOM // 2, 0, H - ZOOM)), int(np.clip(round(cx) - ZOOM // 2, 0, W - ZOOM))
    size = ZOOM * ZOOM_SCALE
    return edge_ticks(Image.fromarray(frame[y0:y0 + ZOOM, x0:x0 + ZOOM]).resize((size, size), Image.NEAREST), 4 * ZOOM_SCALE, 6)


def judge_panel(a, b, shift, link_a, link_b, tag):
    # a, b: uint8 [H, W, C] -> frame t | frame t+1 | t with the shift removed | zoom of it | zoom of t+1 (same region).
    # link_a, link_b: guessed (y, x) of Link, only used to place the zoom
    dy, dx = shift
    aligned = align(a, dy, dx)
    centre = (np.asarray(link_a) + np.asarray([dy, dx]) + np.asarray(link_b)) / 2
    gap, top, size = 6, 18, ZOOM * ZOOM_SCALE
    img = Image.new('RGB', (3 * FRAME_PX + 2 * size + 4 * gap, FRAME_PX + top), 'white')
    for k, frame in enumerate([a, b, aligned]):
        big = Image.fromarray(frame).resize((FRAME_PX, FRAME_PX), Image.NEAREST)
        img.paste(edge_ticks(big, 16 * FRAME_PX // a.shape[0], 8), (k * (FRAME_PX + gap), top))
    x = 3 * (FRAME_PX + gap)
    img.paste(zoom(aligned, *centre), (x, top))
    img.paste(zoom(b, *centre), (x + size + gap, top))
    ImageDraw.Draw(img).text((2, 3), tag, fill='black')
    return img


def judge_sheet(panels, path):
    header = 'frame t | frame t+1 | frame t with the camera shift removed | zoom of aligned t | zoom of t+1 (same region)'
    img = Image.new('RGB', (panels[0].width, sum(p.height for p in panels) + 16), 'white')
    ImageDraw.Draw(img).text((2, 2), header, fill='gray')
    y = 16
    for p in panels:
        img.paste(p, (0, y))
        y += p.height
    img.save(path)


def consensus(a, b):
    # a, b: {id: {label, alignment_ok, confidence, unclear_reason}} -> kept {id: label}, unclear {id: reasons}, no_shift [ids], stats.
    # A pair is kept when both judges give the same action label, one of them is clear and neither flags the alignment.
    # no_shift: either judge saw a cut or fade or a bad alignment, so the computed shift means nothing there.
    assert set(a) == set(b), f'judges differ on {len(set(a) ^ set(b))} panels'
    for v in list(a.values()) + list(b.values()):
        assert v['label'] in LABELS and v.get('unclear_reason') in UNCLEAR_REASONS + [None], v
    kept, unclear, no_shift = {}, {}, []
    n_agree = n_both_unsure = n_bad_alignment = 0
    for i in sorted(a, key=int):
        ja, jb = a[i], b[i]
        reasons = {ja.get('unclear_reason'), jb.get('unclear_reason')} - {None}
        aligned = ja['alignment_ok'] and jb['alignment_ok']
        if 'FADE_OR_CUT' in reasons or not aligned:
            no_shift.append(i)
        if not aligned:
            n_bad_alignment += 1
        elif ja['label'] == jb['label']:
            n_agree += 1
            if ja['label'] == 'UNCLEAR':
                unclear[i] = sorted(reasons)
            elif 'clear' in (ja['confidence'], jb['confidence']):
                kept[i] = ja['label']
            else:
                n_both_unsure += 1
    stats = {'judged': len(a), 'agree': n_agree, 'kept': len(kept), 'kept_scored': sum(l in SCORED for l in kept.values()),
             'unclear': len(unclear), 'both_unsure': n_both_unsure, 'bad_alignment': n_bad_alignment, 'no_shift': len(no_shift)}
    return kept, unclear, no_shift, stats


def scored_subset(labels, group, name, min_pairs):
    # labels: [N] judge label or '' (not kept), group: [N] -> indices of one camera group entering the Link score, pair weights, labels
    # used. Labels under min_pairs leave. STANDING is weighted down to the size of the largest other label, so that separating
    # standing from moving cannot earn most of the score.
    idx = np.array([i for i in range(len(labels)) if group[i] == name and labels[i] in SCORED], dtype=int)
    used = [l for l in SCORED if (labels[idx] == l).sum() >= min_pairs]
    idx = idx[np.isin(labels[idx], used)]
    n_standing = int((labels[idx] == 'STANDING').sum())
    others = [int((labels[idx] == l).sum()) for l in used if l != 'STANDING']
    weights = np.ones(len(idx))
    if n_standing and others:
        weights[labels[idx] == 'STANDING'] = min(max(others) / n_standing, 1)
    return idx, weights, used


def codes_of(vectors, seed=0):
    # vectors: [N, A] float tensor -> k-means codes [N], fit once on all pairs
    centres = kmeans(vectors.float(), N_CODES, seed=seed)
    return torch.cdist(vectors.float(), centres).argmin(1).cpu().numpy()


def weighted_table(codes, label_ids, weights, n_labels):
    table = np.zeros((N_CODES, n_labels))
    np.add.at(table, (codes, label_ids), weights)
    return table


def block_tables(codes, label_ids, weights, blocks, n_labels, rng=None):
    # one code x label table per test block; with rng, codes are first rolled by a random offset inside each block. Rolling keeps
    # the runs of related pairs in a block, so a code that only names the room gets no credit, as shuffling would give it
    tables = {}
    for b in np.unique(blocks):
        at = np.flatnonzero(blocks == b)
        c = codes[at] if rng is None else np.roll(codes[at], rng.integers(len(at)))
        tables[b] = weighted_table(c, label_ids[at], weights[at], n_labels)
    return tables


def adjusted_nmi(codes, label_ids, weights, blocks, n_labels, multiplicity, rng, n_chance):
    # NMI_adj of the blocks taken multiplicity[b] times; chance rolls the codes once per block, so a block drawn twice repeats
    # exactly in both the score and its chance level
    total = lambda tables: sum(multiplicity.get(b, 0) * t for b, t in tables.items())
    table = total(block_tables(codes, label_ids, weights, blocks, n_labels))
    chance = float(np.mean([nmi(total(block_tables(codes, label_ids, weights, blocks, n_labels, rng))) for _ in range(n_chance)]))
    value = nmi(table)
    return (value - chance) / (1 - chance), value, chance, table


def group_inputs(labels, group, blocks, name, min_pairs):
    idx, weights, used = scored_subset(labels, group, name, min_pairs)
    label_ids = np.array([used.index(l) for l in labels[idx]], dtype=int)
    return idx, weights, used, label_ids, blocks[idx]


def link_scores(codes, labels, group, blocks, min_pairs, n_chance=100, seed=0):
    # -> {still, moving: scores + labels used}, headline = worse group (None if a group has under 2 scorable labels)
    rng, out, scores = np.random.default_rng(seed), {}, []
    for name in GROUPS:
        idx, weights, used, label_ids, b = group_inputs(labels, group, blocks, name, min_pairs)
        res = {'n': int(len(idx)), 'n_weighted': round(float(weights.sum()), 1), 'labels': {l: int((label_ids == k).sum()) for k, l in enumerate(used)}}
        if len(used) >= 2:
            adj, value, chance, table = adjusted_nmi(codes[idx], label_ids, weights, b, len(used), {x: 1 for x in np.unique(b)}, rng, n_chance)
            res.update(nmi_adj=round(adj, 4), nmi=round(value, 4), nmi_chance=round(chance, 4),
                       purity=round(float(table.max(1).sum() / table.sum()), 4), majority=round(float(table.sum(0).max() / table.sum()), 4),
                       codes_used=int((table.sum(1) > 0).sum()))
        out[name] = res
        scores.append(res.get('nmi_adj'))
    out['headline'] = None if None in scores else min(scores)
    return out


def block_interval(codes, labels, group, blocks, min_pairs, n_boot=200, n_chance=30, seed=0):
    # 95% interval of each score, resampling whole test blocks (pairs in one block are not independent). Labels and weights stay
    # those of the full set, so every resample scores the same task
    rng = np.random.default_rng(seed)
    ids = np.unique(blocks)
    inputs = {name: group_inputs(labels, group, blocks, name, min_pairs) for name in GROUPS}
    res = {'still': [], 'moving': [], 'headline': []}
    for _ in range(n_boot):
        drawn, counts = np.unique(rng.choice(ids, len(ids)), return_counts=True)
        multiplicity = dict(zip(drawn, counts))
        scores = {}
        for name, (idx, weights, used, label_ids, b) in inputs.items():
            present = np.isin(b, drawn)
            if len(used) >= 2 and len(np.unique(label_ids[present])) >= 2:  # a resample with one label has nothing to score
                scores[name] = adjusted_nmi(codes[idx][present], label_ids[present], weights[present], b[present], len(used), multiplicity, rng, n_chance)[0]
                res[name].append(scores[name])
        if len(scores) == 2:
            res['headline'].append(min(scores.values()))
    return {k: {'low': round(float(np.percentile(v, 2.5)), 3), 'high': round(float(np.percentile(v, 97.5)), 3), 'n_resamples': len(v)} for k, v in res.items() if v}


def ridge_fit(X, y, ridge):
    return np.linalg.solve(X.T @ X + ridge * len(X) * np.eye(X.shape[1]), X.T @ y)


def linear_shift_readout(vectors, shift, blocks, ridges=(1e-3, 1e-2, 1e-1, 1, 10, 100)):
    # vectors [N, A], shift [N, 2] -> predicted shift [N, 2]: ridge regression fit on half the blocks and applied to the other half,
    # both ways, so every pair is predicted by a map that never saw its block. The ridge is chosen inside the fit half, leaving
    # one block out at a time, so a useless big vector falls back on the mean instead of overfitting
    X = (vectors - vectors.mean(0)) / vectors.std(0).clip(min=1e-6)
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    half = np.isin(blocks, np.unique(blocks)[::2])
    pred = np.zeros_like(shift, dtype=np.float64)
    for fit, apply in ((half, ~half), (~half, half)):
        Xf, yf, bf = X[fit], shift[fit], blocks[fit]
        errors = [sum(((Xf[bf == b] @ ridge_fit(Xf[bf != b], yf[bf != b], r) - yf[bf == b]) ** 2).sum() for b in np.unique(bf)) for r in ridges]
        pred[apply] = X[apply] @ ridge_fit(Xf, yf, ridges[int(np.argmin(errors))])
    return pred


def shift_scores(pred, shift, group, valid):
    # relative distance ||pred - true|| / max(||true||, 1 px) per camera group, on the pairs where the computed shift means something
    d = np.linalg.norm(pred - shift, axis=1) / np.maximum(np.linalg.norm(shift, axis=1), 1)
    return {name: {'n': int(((group == name) & valid).sum()), 'median': round(float(np.median(d[(group == name) & valid])), 3),
                   'p90': round(float(np.percentile(d[(group == name) & valid], 90)), 3)} for name in GROUPS}
