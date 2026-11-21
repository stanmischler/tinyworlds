"""Last step of cluster/lix/setup_env.sh, on CPU: the run's config loads, the data and the scoring files are readable and agree with each other,
and the batch pools have the expected sizes."""
import json

import h5py
import numpy as np
from omegaconf import OmegaConf

import experiments.small_action.train as train
from experiments.small_action.eval import GAP_ROW, LINK_BOXES, SET, SPLIT, TEST_H5, model_kwargs

C = 'configs/experiments/small_action'
cfg = OmegaConf.merge(*[OmegaConf.load(f'{C}/{name}.yaml') for name in ('base', 'simple_recipe', 'weights_and_window')])
table, motion = np.load(cfg.train_shifts), np.load(cfg.still_motion_table)
with h5py.File(cfg.train_h5, 'r') as f:
    n_frames = len(f['frames'])
assert n_frames == len(table['shifts']) == len(motion['moving_px']), 'the frames and the two tables do not have the same number of rows'
assert 'dialog' in motion, 'the motion table has no dialog flags'
assert not np.isnan(table['shifts'][:, GAP_ROW]).all()
starts, _, _, n_kept = train.drawable_starts(cfg, table)
pools = train.batch_pools(cfg, starts, table['shifts'])
print(f'{n_frames} frames, {n_kept} pairs kept; pools: ' + ', '.join(f'{name} {len(np.unique(pool))} pairs at {share:.0%} of the batch' for name, pool, share in pools))
assert [name for name, _, _ in pools] == ['focus', 'idle', 'others'] and len(np.unique(pools[0][1])) == 23047, 'the focus pool is not the expected one'
for path in (SPLIT, f'{SET}/pairs.json', f'{SET}/labels.json', LINK_BOXES):
    json.load(open(path))
np.load(f'{SET}/link_positions.npz'), np.load('data/zelda_test_shifts.npz'), np.load(cfg.memorisation_shifts)
for path in (TEST_H5, cfg.memorisation_h5):
    with h5py.File(path, 'r') as f:
        assert f['frames'].shape[1:] == (128, 128, 3), path
print('model settings:', model_kwargs(cfg), '| window_fraction', cfg.window_fraction, '| steps', cfg.n_updates, '| batch', cfg.batch_size)
print('setup check passed')
