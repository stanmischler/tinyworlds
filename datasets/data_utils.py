import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datasets.datasets import PongDataset, SonicDataset, PolePositionDataset, PicoDoomDataset, ZeldaDataset, EpisodeH5Dataset

DEFAULT_NUM_WORKERS = 2
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_PIN_MEMORY = False
DEFAULT_PERSISTENT_WORKERS = True


def _default_video_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def _load_video_dataset_pair(dataset_cls, video_rel_path, h5_rel_path, num_frames, transform=None, fps=30, preload_ratio=1, **kwargs):
    current_folder_path = os.getcwd()
    video_path = current_folder_path + video_rel_path
    preprocessed_path = current_folder_path + h5_rel_path
    transform = _default_video_transform() if transform is None else transform
    train, val = [dataset_cls(
        video_path,
        transform=transform,
        save_path=preprocessed_path,
        train=is_train,
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        **kwargs
    ) for is_train in (True, False)]
    return train, val


# dataset -> (class, mp4, h5, fps that overrides the requested one or None); *_TRAIN = the game minus the held-out
# test blocks (see scripts/data/split_dataset.py), same class and fps
VIDEO_GAMES = {
    'PONG': (PongDataset, '/data/pong.mp4', '/data/pong_frames.h5', None),
    'PONG_TRAIN': (PongDataset, '/data/pong.mp4', '/data/pong_train_frames.h5', None),
    'SONIC': (SonicDataset, '/data/sonic_frames.mp4', '/data/sonic_frames.h5', None),
    'SONIC_TRAIN': (SonicDataset, '/data/sonic_frames.mp4', '/data/sonic_train_frames.h5', None),
    'POLE_POSITION': (PolePositionDataset, '/data/pole_position.mp4', '/data/pole_position_frames.h5', None),
    'PICODOOM': (PicoDoomDataset, '/data/picodoom cleaned.mp4', '/data/picodoom_frames.h5', 30),
    'ZELDA': (ZeldaDataset, '/data/Zelda oot2d 1 Cut.mp4', '/data/zelda_frames.h5', None),
    'ZELDA_TRAIN': (ZeldaDataset, '/data/Zelda oot2d 1 Cut.mp4', '/data/zelda_train_frames.h5', None),
}


# episode .h5 datasets (datasets.EpisodeH5Dataset): fps is fixed by the .h5; the held-out evals read their own files
# (eval_pusht.py / eval_nanowm.py: <env>_val_frames.h5, eval_lpwm.py: obj3d_test_frames.h5)
EPISODE_H5 = {
    'PUSHT': '/data/pusht_frames.h5',            # scripts/data/convert_pusht.py, 10-D actions
    'POINTMAZE': '/data/point_maze_frames.h5',   # scripts/data/convert_dinowm.py --env point_maze, 10-D actions
    'WALL': '/data/wall_frames.h5',              # scripts/data/convert_dinowm.py --env wall, 10-D actions
    'OBJ3D': '/data/obj3d_train_frames.h5',      # scripts/data/convert_obj3d.py, no actions
}
PUSHT_TRAIN_H5 = EPISODE_H5['PUSHT']


def load_episode_h5(dataset, num_frames=4, preload_ratio=1):
    """Train split of an episode .h5 dataset. Val is the same object (as every other game here)."""
    train = EpisodeH5Dataset(os.getcwd() + EPISODE_H5[dataset], num_frames=num_frames, preload_ratio=preload_ratio)
    return train, train


def load_pusht(num_frames=4, fps=None, preload_ratio=1):
    return load_episode_h5('PUSHT', num_frames=num_frames, preload_ratio=preload_ratio)


def dataset_action_dim(dataset):
    """Ground-truth action size of a dataset (None for the action-less datasets)."""
    if dataset in EPISODE_H5:
        import h5py
        with h5py.File(os.getcwd() + EPISODE_H5[dataset], 'r') as f:
            return int(f['actions'].shape[1]) if 'actions' in f else None
    return None


def data_loaders(train_data, val_data, batch_size, distributed=False, rank=0, world_size=1,
                 num_workers=None, pin_memory=None, generator=None):
    # num_workers / pin_memory: None keeps the module defaults (2 workers, no pinning)
    # generator: seeded torch.Generator for a reproducible batch order (the default sampler reseeds every epoch)
    num_workers = DEFAULT_NUM_WORKERS if num_workers is None else int(num_workers)
    pin_memory = DEFAULT_PIN_MEMORY if pin_memory is None else bool(pin_memory)
    persistent_workers = DEFAULT_PERSISTENT_WORKERS and num_workers > 0
    prefetch_factor = DEFAULT_PREFETCH_FACTOR if num_workers > 0 else None
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)

    def loader(data, sampler, **kw):
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False if sampler is not None else True,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True,
            **kw,
        )

    train_loader = loader(train_data, train_sampler, generator=generator)
    val_loader = loader(val_data, val_sampler)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size, num_frames=1, distributed=False, rank=0, world_size=1, fps=15, preload_ratio=1,
                               num_workers=None, pin_memory=None, generator=None):
    if dataset in EPISODE_H5:
        training_data, validation_data = load_episode_h5(dataset, num_frames=num_frames, preload_ratio=preload_ratio)
    elif dataset in VIDEO_GAMES:
        dataset_cls, video_rel_path, h5_rel_path, game_fps = VIDEO_GAMES[dataset]
        training_data, validation_data = _load_video_dataset_pair(
            dataset_cls, video_rel_path, h5_rel_path, num_frames=num_frames,
            fps=fps if game_fps is None else game_fps, preload_ratio=preload_ratio)
    else:
        raise ValueError('Invalid dataset')

    training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size,
        distributed=distributed, rank=rank, world_size=world_size,
        num_workers=num_workers, pin_memory=pin_memory, generator=generator,
    )
    # np.var materialises a float64 copy: subsample to ~10k frames past 100k (Push-T's 23 GB would need ~180 GB);
    # exact (unchanged) for the smaller games
    data = training_data.data
    x_train_var = np.var(data if len(data) <= 100_000 else data[::len(data) // 10_000])

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def visualize_reconstruction(original, reconstruction, save_path=None):
    # original: (B, C, H, W) or (B, T, C, H, W)
    # reconstruction: (B, C, H, W) or (B, T, C, H, W) 

    # move tensors to CPU and convert to float32 for matplotlib compatibility
    original = original.detach().to('cpu', dtype=torch.float32)
    reconstruction = reconstruction.detach().to('cpu', dtype=torch.float32)

    # handle single frames by expanding to sequences
    if original.dim() == 4:  # (B, C, H, W)
        original = original.unsqueeze(1)  # Add sequence dimension
    if reconstruction.dim() == 4:  # (B, C, H, W)
        reconstruction = reconstruction.unsqueeze(1)  # Add sequence dimension

    # take first 4 sequences, each of length 4 (or available length)
    num_sequences = min(4, original.shape[0])
    seq_length = min(4, original.shape[1])

    original = original[:num_sequences, :seq_length]  # (B, T, C, H, W)
    reconstruction = reconstruction[:num_sequences, :seq_length]  # (B, T, C, H, W)

    # create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # for original sequences
    # reshape to (B * T, C, H, W) for make_grid
    orig_flat = original.reshape(-1, *original.shape[2:])  # (B*T, C, H, W)
    grid_orig = make_grid(orig_flat, nrow=seq_length, normalize=True, padding=2).clamp(0, 1)
    ax1.imshow(grid_orig.permute(1, 2, 0).contiguous().numpy())
    ax1.axis('off')
    ax1.set_title(f'Original Sequences (4 sequences × {seq_length} frames)')

    # for reconstructed sequences
    recon_flat = reconstruction.reshape(-1, *reconstruction.shape[2:])  # (B*T, C, H, W)
    grid_recon = make_grid(recon_flat, nrow=seq_length, normalize=True, padding=2).clamp(0, 1)
    ax2.imshow(grid_recon.permute(1, 2, 0).contiguous().numpy())
    ax2.axis('off')
    ax2.set_title(f'Reconstructed Sequences (4 sequences × {seq_length} frames)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
