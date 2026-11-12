"""Action check: how well a learned action matches the true button, on held-out pairs of a dataset that records buttons
(Procgen, scripts/data/convert_procgen.py). The buttons are read here only; no model is ever trained on them.

Three readings of the same held-out pairs (t, t+1):
  - probe: predict the true button from the action z. A linear and a small MLP classifier are fitted on the first half of
    the pairs and scored on the second half. `majority_acc` is the score of always answering the most frequent button.
  - shuffle gap: decoder error with z taken from a pair that has a different true button, minus the error with the pair's
    own z = how much the decoder uses z.
  - same-button gap: the same with z taken from another pair that has the same true button. If z held nothing but the
    button this would be 0. `beyond_button_share` = same-button gap / shuffle gap is an upper bound on what z carries
    beyond the button, not a measurement of leakage: the same button can have different effects (the player against a
    wall), and other objects' motion (the next frame leaking into z) counts the same way.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Procgen button id (15) -> class (9): the 8 directions and "no move". The 6 letter keys do nothing in Bigfish (measured:
# the player's mean displacement under each is 0), so they count as no move.
NO_MOVE = 4
BIGFISH_CLASSES = [0, 1, 2, 3, NO_MOVE, 5, 6, 7, 8] + [NO_MOVE] * 6
N_CLASSES = 9


def button_classes(buttons):
    # buttons: [N] Procgen button ids -> [N] class ids in 0..8
    return torch.as_tensor(BIGFISH_CLASSES, device=buttons.device)[buttons.long()]


def same_class_partner(labels):
    """labels [N] -> partner [N]: for each index, another index with the same label (the next one of its class, cyclic;
    itself when it is alone in its class)."""
    order = torch.argsort(labels, stable=True)                     # [N] indices grouped by class
    sorted_labels = labels[order]
    group_start = torch.searchsorted(sorted_labels, sorted_labels)  # [N] position of each element's class start
    is_last = torch.ones_like(sorted_labels, dtype=torch.bool)
    is_last[:-1] = sorted_labels[1:] != sorted_labels[:-1]
    partner_sorted = torch.where(is_last, order[group_start], torch.roll(order, -1))
    partner = torch.empty_like(order)
    partner[order] = partner_sorted
    return partner


def different_class_partner(labels, tries=8):
    """labels [N] -> partner [N]: for each index, an index with another label (the nearest of the next `tries` ones)."""
    own = torch.arange(len(labels), device=labels.device)
    partner = own.roll(1)
    for k in range(2, tries + 1):
        partner = torch.where(labels[partner] == labels, own.roll(k), partner)
    return partner


def fit_probe(z_fit, y_fit, z_test, hidden=0, steps=300, seed=0):
    """Classifier z -> class, full-batch Adam on (z_fit, y_fit); returns the predicted classes of z_test [M].
    hidden 0 = linear, else one hidden layer of that width. z is standardised with the fit half's statistics."""
    # the probe's init must not move the training run's random streams: save and restore them (CPU and CUDA)
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    dim = z_fit.shape[1]
    layers = [nn.Linear(dim, N_CLASSES)] if not hidden else [nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, N_CLASSES)]
    probe = nn.Sequential(*layers).to(z_fit.device)
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    mean, std = z_fit.mean(0), z_fit.std(0).clamp_min(1e-6)
    opt = torch.optim.Adam(probe.parameters(), lr=0.02)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(probe((z_fit - mean) / std), y_fit).backward()
            opt.step()
    return probe((z_test - mean) / std).argmax(1)


@torch.no_grad()
def action_check(idm, decoder, f_a, f_b, x_a, x_b, buttons, amp, chunk=1024):
    """idm, decoder: models/como.py MotionIDM and MotionDecoder (eval mode is the caller's job).
    f_a, f_b: frozen features of frames t and t+1 [N, S, D]; x_a, x_b: their pixels in [-1, 1] [N, C, H, W];
    buttons: true Procgen button of each pair [N]. Returns a dict of floats (keys prefixed check/)."""
    labels = button_classes(buttons)                               # [N]
    z = []
    for i in range(0, len(f_a), chunk):
        with amp():
            z.append(idm(f_a[i:i + chunk].float(), f_b[i:i + chunk].float()))
    z = torch.cat(z)                                               # [N, Q, L]

    def decoder_mse(source):
        # mean squared error of the decoder when pair i is given the action of pair source[i]
        total = 0.0
        for i in range(0, len(z), chunk):
            with amp():
                pred = decoder(x_a[i:i + chunk], z[source[i:i + chunk]])
            total += F.mse_loss(pred.float(), x_b[i:i + chunk], reduction='sum').item()
        return total / x_b.numel()

    own = torch.arange(len(z), device=z.device)
    recon = decoder_mse(own)
    shuffle_gap = decoder_mse(different_class_partner(labels)) - recon
    same_button_gap = decoder_mse(same_class_partner(labels)) - recon

    flat = z.float().flatten(1)                                    # [N, A]
    half = len(flat) // 2
    y_test = labels[half:]
    linear = fit_probe(flat[:half], labels[:half], flat[half:])
    mlp = fit_probe(flat[:half], labels[:half], flat[half:], hidden=64)
    moves = y_test != NO_MOVE
    out = {
        'check/probe_linear_acc': (linear == y_test).float().mean().item(),
        'check/probe_mlp_acc': (mlp == y_test).float().mean().item(),
        'check/probe_mlp_move_acc': (mlp[moves] == y_test[moves]).float().mean().item() if moves.any() else 0.0,  # 8 directions only
        'check/majority_acc': torch.bincount(y_test, minlength=N_CLASSES).max().item() / len(y_test),
        'check/shuffle_gap': shuffle_gap,
        'check/same_button_gap': same_button_gap,
    }
    if shuffle_gap > 0.01 * recon:  # undefined while the decoder barely uses z (start of training)
        out['check/beyond_button_share'] = same_button_gap / shuffle_gap
    return out
