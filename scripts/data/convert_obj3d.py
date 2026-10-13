"""Convert OBJ3D (G-SWM's 3D-objects dataset, as used by DDLP / LPWM) into tinyworlds .h5 files, one per split.

Source: HF dataset taldatech/OBJ3D (OBJ3D.zip, 7.39 GB; same content as G-SWM's Google Drive
1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm): <split>/<episode int>/test_<step>.png for split in train (2920 eps) / val (200) /
test (200), 100 frames per episode.

Frames are read exactly as LPWM's loader (datasets/obj3d_ds.py Obj3D, ds 'obj3d128' -> res 128 in
datasets/get_dataset.py): episode folders = the int-named entries of <split>/ sorted numerically, frames = glob
test_*.png sorted by the number after '_', then PIL Image.open(path).resize((128, 128)) with PIL's default resample (a
no-op copy when the PNG is already 128x128) and ToTensor()[:3] (the first 3 channels / 255, so uint8 RGB is lossless).

Outputs (in --out-dir), one per split: obj3d_{train,val,test}_frames.h5
  frames          uint8 [N, 128, 128, 3]  every frame of every episode, episodes in LPWM's order, stride 1
  episode_index   int32 [N]               position of the episode in LPWM's sorted list (= LPWM's dataset index in test/val)
  step_index      int32 [N]               frame index within the episode (0 = first frame)
  seq_lengths     int32 [n_ep]            frames per episode
  episode_folder  int32 [n_ep]            the episode's folder name
  attrs: source, split, resize, source_resolution, res

Usage (repo root, PYTHONPATH=$PWD):
    python scripts/data/convert_obj3d.py                                  # downloads + unzips into data/obj3d_raw if missing
    python scripts/data/convert_obj3d.py --limit-episodes 5 --out-dir /tmp/obj3d   # smoke
"""

import argparse
import glob
import os
import os.path as osp
import time
import zipfile
from multiprocessing import Pool

import h5py
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # as LPWM's loader

HF_REPO, HF_FILE = 'taldatech/OBJ3D', 'OBJ3D.zip'
SPLITS = ('train', 'val', 'test')
RES = 128


def find_root(raw_dir):
    """The directory holding train/ val/ test/ (the zip may nest it, e.g. raw_dir/OBJ3D), or None."""
    for d in [raw_dir] + sorted(glob.glob(osp.join(raw_dir, '*'))) + sorted(glob.glob(osp.join(raw_dir, '*', '*'))):
        if all(osp.isdir(osp.join(d, s)) for s in SPLITS):
            return d
    return None


def ensure_raw(raw_dir):
    root = find_root(raw_dir) if osp.isdir(raw_dir) else None
    if root is not None:
        return root
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = osp.join(raw_dir, HF_FILE)
    if not osp.exists(zip_path):
        from huggingface_hub import hf_hub_download
        print(f'downloading {HF_REPO}/{HF_FILE} -> {raw_dir}', flush=True)
        zip_path = hf_hub_download(HF_REPO, HF_FILE, repo_type='dataset', local_dir=raw_dir)
    print(f'unzipping {zip_path}', flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_dir)
    root = find_root(raw_dir)
    assert root is not None, f'no train/val/test dirs under {raw_dir} after unzipping'
    return root


def list_episodes(split_dir):
    """LPWM's Obj3D.__init__: int-named folders sorted, then test_*.png sorted by step number."""
    folders = []
    for name in os.listdir(split_dir):
        try:
            folders.append(int(name))
        except ValueError:
            continue
    folders.sort()
    get_num = lambda x: int(osp.splitext(osp.basename(x))[0].partition('_')[-1])
    return folders, [sorted(glob.glob(osp.join(split_dir, str(f), 'test_*.png')), key=get_num) for f in folders]


def load_episode(paths):
    # -> uint8 [T, RES, RES, 3], source (w, h) of the first frame, its mode
    frames, src = [], None
    for p in paths:
        img = Image.open(p)
        src = src or (img.size, img.mode)
        assert img.mode in ('RGB', 'RGBA'), f'{p}: mode {img.mode}'  # ToTensor()[:3] = RGB only for these modes
        frames.append(np.asarray(img.resize((RES, RES)))[..., :3])
    return np.stack(frames), src


def convert_split(root, split, out_path, limit, workers):
    folders, episodes = list_episodes(osp.join(root, split))
    if limit:
        folders, episodes = folders[:limit], episodes[:limit]
    lengths = [len(e) for e in episodes]
    n = sum(lengths)
    print(f'{split}: {len(episodes)} episodes, {n} frames (lengths {min(lengths)}-{max(lengths)})', flush=True)
    t0, sources = time.time(), set()
    with h5py.File(out_path + '.part', 'w') as f:
        fr = f.create_dataset('frames', (n, RES, RES, 3), dtype=np.uint8, chunks=(min(64, n), RES, RES, 3), compression='lzf')
        ep_d = f.create_dataset('episode_index', (n,), dtype=np.int32)
        st_d = f.create_dataset('step_index', (n,), dtype=np.int32)
        pos = 0
        with Pool(workers) as pool:
            for i, (frames, src) in enumerate(pool.imap(load_episode, episodes, chunksize=4)):
                T = len(frames)
                fr[pos:pos + T] = frames
                ep_d[pos:pos + T] = i
                st_d[pos:pos + T] = np.arange(T)
                pos += T
                sources.add(src)
                if (i + 1) % 200 == 0:
                    print(f'  {split} {i + 1}/{len(episodes)} ({time.time() - t0:.0f}s)', flush=True)
        f.create_dataset('seq_lengths', data=np.array(lengths, dtype=np.int32))
        f.create_dataset('episode_folder', data=np.array(folders, dtype=np.int32))
        f.attrs['source'] = f'OBJ3D (G-SWM) via HF {HF_REPO}/{HF_FILE}'
        f.attrs['split'] = split
        f.attrs['res'] = RES
        f.attrs['source_resolution'] = str(sorted(sources))  # [((w, h), mode)]
        f.attrs['resize'] = "LPWM datasets/obj3d_ds.py: PIL Image.resize((128, 128)) default resample, ToTensor()[:3]"
    os.replace(out_path + '.part', out_path)
    print(f'  wrote {out_path}: frames {(n, RES, RES, 3)}, source {sorted(sources)}, {os.path.getsize(out_path) / 1e9:.2f} GB '
          f'({time.time() - t0:.0f}s)', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-dir', default='data/obj3d_raw', help='where OBJ3D.zip is downloaded and unzipped')
    p.add_argument('--out-dir', default='data')
    p.add_argument('--splits', nargs='+', default=list(SPLITS), choices=SPLITS)
    p.add_argument('--limit-episodes', type=int, help='first N episodes per split (smoke test)')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    args = p.parse_args()
    root = ensure_raw(args.raw_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    for split in args.splits:
        convert_split(root, split, osp.join(args.out_dir, f'obj3d_{split}_frames.h5'), args.limit_episodes, args.workers)


if __name__ == '__main__':
    main()
