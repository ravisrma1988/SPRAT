"""Decode, md5-verify and resize every manifest image once into uint8 .npy arrays.

Train images -> 256x256 (training takes a random 224 crop on the GPU, as v1's Resize(256)+RandomCrop(224));
val/test images -> 224x224 (v1's evaluation Resize(224)). Resizing uses torchvision's PIL path
(bilinear with PIL's antialiasing), identical to v1's transforms.

Usage (run detached on the DGX; ~13.6k images):
  python build_cache.py --manifest .../manifest_sprat_v2_dedup.csv --root /workspace/ravi/SPRAT \
      --out /workspace/ravi/SPRAT/data/cache_v2 --workers 32
"""
import argparse, hashlib, io, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprat_common import EVAL_SIZE, TRAIN_CACHE_SIZE, read_manifest, sha256_file  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--root', required=True, help='manifest paths are relative to this')
p.add_argument('--out', required=True)
p.add_argument('--workers', type=int, default=32)
p.add_argument('--splits', default='train,val,test')
args = p.parse_args()


def load(job):
    path, size, md5 = job
    raw = Path(path).read_bytes()
    if hashlib.md5(raw).hexdigest() != md5:
        return None, path
    with Image.open(io.BytesIO(raw)) as im:
        img = TF.resize(im.convert('RGB'), [size, size])  # handles RGBA / P / CMYK / GIF-in-.png
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy(), path


def replace_all(pairs):
    for tmp, final in pairs:
        os.replace(tmp, final)


if __name__ == '__main__':
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, class_names = read_manifest(args.manifest)
    sha = sha256_file(args.manifest)
    meta_path = out / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(splits={})
    meta.update(class_names=class_names, train_size=TRAIN_CACHE_SIZE, eval_size=EVAL_SIZE,
                resize='torchvision.functional.resize on PIL, bilinear', torchvision=torchvision.__version__)
    staged = []  # (tmp, final) pairs; renamed only after every requested split passed
    for split in args.splits.split(','):
        sr = [r for r in rows if r['split'] == split]
        size = TRAIN_CACHE_SIZE if split == 'train' else EVAL_SIZE
        X = np.empty((len(sr), 3, size, size), dtype=np.uint8)
        bad, t0 = [], time.time()
        jobs = [(str(Path(args.root) / r['path']), size, r['md5']) for r in sr]
        with Pool(args.workers) as pool:
            for i, (arr, path) in enumerate(pool.imap(load, jobs, chunksize=16)):
                if arr is None:
                    bad.append(path)
                else:
                    X[i] = arr
                if i % 2000 == 0:
                    print(f'  {split}: {i}/{len(sr)} ({time.time()-t0:.0f}s)', flush=True)
        if bad:
            sys.exit(f'{len(bad)} files do not match the manifest md5, e.g. {bad[:3]} - nothing was replaced')
        for name, writer in ((f'{split}_x.npy', lambda f: np.save(f, X)),
                             (f'{split}_y.npy', lambda f: np.save(f, np.array([r['class_id'] for r in sr], dtype=np.int64))),
                             (f'{split}_paths.txt', None)):
            tmp = out / (name + '.tmp')
            if writer is None:
                tmp.write_text('\n'.join(r['path'] for r in sr), encoding='utf-8')
            else:
                with open(tmp, 'wb') as fh:
                    writer(fh)
            staged.append((tmp, out / name))
        meta['splits'][split] = dict(n=len(sr), size=size, bytes=int(X.nbytes), seconds=round(time.time() - t0, 1),
                                     manifest=str(args.manifest), manifest_sha256=sha)
        print(f'[CACHE] {split}: {len(sr)} images at {size}px, md5 verified, {X.nbytes/2**30:.2f} GiB', flush=True)
    replace_all(staged)
    tmp = out / 'meta.json.tmp'; tmp.write_text(json.dumps(meta, indent=2)); os.replace(tmp, meta_path)
    print('[DONE]', json.dumps(meta['splits']), flush=True)
