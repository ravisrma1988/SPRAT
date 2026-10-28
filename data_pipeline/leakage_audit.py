"""Leakage / split audit for the SPRAT wheat + rice benchmark.

Checks
  1. Inventory: per-image dataset, split, class, md5, size, format.
  2. Exact duplicates (md5) across splits and across classes (label conflicts).
  3. Near-duplicates via 64-bit DCT perceptual hash, matched against all 8
     dihedral variants (rotations / flips) so augmented copies are caught.
  4. Wheat: train vs valid vs test, using the dataset's own folders.
  5. Rice: no official split exists and the DGX split manifest is not available
     locally, so (a) near-duplicate clusters are found over the whole rice pool and
     (b) the expected train->test leakage of a random stratified 80/10/10 split is
     simulated over many seeds.
  6. Source cues: image size / format distribution per crop.

Usage
  python leakage_audit.py <wheat_root> <rice_root> <out_dir>
"""
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
NEAR_T = 6        # Hamming distance (of 64) treated as a near-duplicate
CAND_T = 10       # looser band reported separately for manual inspection

# ── perceptual hash ─────────────────────────────────────────────────────────
_N = 32
_k = np.arange(_N)
_DCT = np.sqrt(2 / _N) * np.cos(np.pi * (2 * _k[None, :] + 1) * _k[:, None] / (2 * _N))
_DCT[0] /= np.sqrt(2)


def _phash(a):
    d = _DCT @ a @ _DCT.T
    low = d[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int(np.packbits(bits).view('>u8')[0])


def _dihedral(a):
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, np.fliplr(r)]
    return out


def hash_one(path):
    p = Path(path)
    raw = p.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    try:
        with Image.open(p) as im:
            w, h, fmt, mode = im.width, im.height, im.format, im.mode
            g = np.asarray(im.convert('L').resize((_N, _N), Image.LANCZOS), dtype=np.float64)
        hashes = [_phash(v) for v in _dihedral(g)]
        err = ''
    except Exception as e:  # unreadable image is itself worth reporting
        w = h = 0; fmt = mode = ''; hashes = [0] * 8; err = repr(e)
    return dict(md5=md5, bytes=len(raw), w=w, h=h, fmt=fmt, mode=mode, hashes=hashes, err=err)


# ── helpers ─────────────────────────────────────────────────────────────────
def popcount64(x):
    return np.bitwise_count(x)


def min_dist(query_variants, pool):
    """query_variants: (8,) uint64, pool: (M,) uint64 canonical hashes.
    Returns (M,) min Hamming distance over the query's dihedral variants."""
    d = popcount64(query_variants[:, None] ^ pool[None, :])
    return d.min(axis=0)


def collect(wheat_root, rice_root):
    items = []
    for split_dir in sorted(Path(wheat_root).iterdir()):
        if not split_dir.is_dir():
            continue
        split = {'valid': 'val'}.get(split_dir.name, split_dir.name)
        for cls_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    items.append(dict(crop='wheat', split=split, cls_folder=cls_dir.name, path=str(f)))
    rice_base = Path(rice_root)
    for cls_dir in sorted(d for d in rice_base.iterdir() if d.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                items.append(dict(crop='rice', split='pool', cls_folder=cls_dir.name, path=str(f)))
    return items


def canon_class(crop, folder):
    s = folder.lower().replace(' ', '_')
    for suf in ('_test_valid', '_valid', '_test'):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return f'{crop}_{s}'


def main():
    wheat_root, rice_root, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    items = collect(wheat_root, rice_root)
    print(f'[INV] {len(items)} images', flush=True)
    with ProcessPoolExecutor() as ex:
        for it, h in zip(items, ex.map(hash_one, [i['path'] for i in items], chunksize=64)):
            it.update(h)
    for it in items:
        it['cls'] = canon_class(it['crop'], it['cls_folder'])

    errs = [i for i in items if i['err']]
    H = np.array([i['hashes'] for i in items], dtype=np.uint64)  # (N, 8); col 0 = canonical
    canon = H[:, 0]
    idx = {k: np.array([n for n, i in enumerate(items) if (i['crop'], i['split']) == k])
           for k in {(i['crop'], i['split']) for i in items}}

    summary = dict(n_images=len(items), unreadable=len(errs), near_threshold=NEAR_T, cand_threshold=CAND_T)

    # counts per split / class (verifies Table 1)
    counts = Counter((i['crop'], i['split'], i['cls']) for i in items)
    summary['counts'] = {f'{c}|{s}|{k}': v for (c, s, k), v in sorted(counts.items())}

    # ── exact duplicates ──
    by_md5 = defaultdict(list)
    for n, i in enumerate(items):
        by_md5[i['md5']].append(n)
    dup_groups = [g for g in by_md5.values() if len(g) > 1]
    exact_rows = []
    for g in dup_groups:
        splits = {f"{items[n]['crop']}:{items[n]['split']}" for n in g}
        classes = {items[n]['cls'] for n in g}
        exact_rows.append(dict(size=len(g), splits='|'.join(sorted(splits)), classes='|'.join(sorted(classes)),
                               cross_split=len(splits) > 1, cross_class=len(classes) > 1,
                               paths='|'.join(items[n]['path'] for n in g)))
    summary['exact_dup_groups'] = len(dup_groups)
    summary['exact_dup_images_extra'] = sum(len(g) - 1 for g in dup_groups)
    summary['exact_dup_groups_cross_split'] = sum(r['cross_split'] for r in exact_rows)
    summary['exact_dup_groups_cross_class'] = sum(r['cross_class'] for r in exact_rows)
    _write_csv(out_dir / 'exact_duplicates.csv', exact_rows)

    # ── wheat near-duplicates: val/test queries vs train, and test vs val ──
    near_rows = []

    def query(qkey, pkey, tag):
        q_idx, p_idx = idx[qkey], idx[pkey]
        pool = canon[p_idx]
        hit_near = hit_cand = 0
        for qn in q_idx:
            d = min_dist(H[qn], pool)
            if qkey == pkey:
                d[p_idx == qn] = 99
            j = int(d.argmin()); dm = int(d[j])
            if dm <= CAND_T:
                pn = int(p_idx[j])
                near_rows.append(dict(pair=tag, dist=dm, q_cls=items[qn]['cls'], p_cls=items[pn]['cls'],
                                      same_class=items[qn]['cls'] == items[pn]['cls'],
                                      q_path=items[qn]['path'], p_path=items[pn]['path']))
                hit_cand += 1
                hit_near += dm <= NEAR_T
        summary[f'{tag}_queries'] = len(q_idx)
        summary[f'{tag}_near_le{NEAR_T}'] = hit_near
        summary[f'{tag}_cand_le{CAND_T}'] = hit_cand
        print(f'[NEAR] {tag}: {hit_near}/{len(q_idx)} <= {NEAR_T}, {hit_cand} <= {CAND_T}', flush=True)

    query(('wheat', 'test'), ('wheat', 'train'), 'wheat_test_vs_train')
    query(('wheat', 'val'), ('wheat', 'train'), 'wheat_val_vs_train')
    query(('wheat', 'test'), ('wheat', 'val'), 'wheat_test_vs_val')
    query(('wheat', 'train'), ('wheat', 'train'), 'wheat_train_within')
    query(('rice', 'pool'), ('rice', 'pool'), 'rice_within_pool')
    query(('rice', 'pool'), ('wheat', 'train'), 'rice_vs_wheat_train')
    _write_csv(out_dir / 'near_duplicates.csv', sorted(near_rows, key=lambda r: (r['pair'], r['dist'])))

    # ── rice: near-dup graph + simulated random-split leakage ──
    r_idx = idx[('rice', 'pool')]
    rH = H[r_idx]; rc = canon[r_idx]
    nbrs = [np.flatnonzero(min_dist(rH[a], rc) <= NEAR_T) for a in range(len(r_idx))]
    nbrs = [n[n != a] for a, n in enumerate(nbrs)]
    has_dup = np.array([len(n) > 0 for n in nbrs])
    summary['rice_images_with_near_dup'] = int(has_dup.sum())
    cross_cls = sum(any(items[r_idx[b]]['cls'] != items[r_idx[a]]['cls'] for b in n) for a, n in enumerate(nbrs))
    summary['rice_images_with_near_dup_other_class'] = int(cross_cls)

    cls_arr = np.array([items[n]['cls'] for n in r_idx])
    leak = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        split = np.empty(len(r_idx), dtype='<U5')
        for c in np.unique(cls_arr):
            m = np.flatnonzero(cls_arr == c); rng.shuffle(m)
            n_tr = int(round(0.8 * len(m))); n_va = int(round(0.1 * len(m)))
            split[m[:n_tr]] = 'train'; split[m[n_tr:n_tr + n_va]] = 'val'; split[m[n_tr + n_va:]] = 'test'
        te = np.flatnonzero(split == 'test')
        leak.append(np.mean([np.any(split[nbrs[a]] == 'train') for a in te]))
    summary['rice_sim_test_with_train_neardup_mean'] = float(np.mean(leak))
    summary['rice_sim_test_with_train_neardup_p05_p95'] = [float(np.percentile(leak, 5)), float(np.percentile(leak, 95))]

    # ── source cues ──
    def dist(key):
        c = Counter(key(i) for i in items if True)
        return c
    for crop in ('wheat', 'rice'):
        sub = [i for i in items if i['crop'] == crop]
        summary[f'{crop}_formats'] = dict(Counter(i['fmt'] for i in sub).most_common(5))
        summary[f'{crop}_modes'] = dict(Counter(i['mode'] for i in sub).most_common(5))
        summary[f'{crop}_top_sizes'] = {f"{w}x{h}": v for (w, h), v in Counter((i['w'], i['h']) for i in sub).most_common(8)}
        summary[f'{crop}_distinct_sizes'] = len({(i['w'], i['h']) for i in sub})

    inv_cols = ['crop', 'split', 'cls', 'cls_folder', 'path', 'md5', 'bytes', 'w', 'h', 'fmt', 'mode', 'err']
    _write_csv(out_dir / 'inventory.csv', [{k: i[k] for k in inv_cols} for i in items])
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'counts'}, indent=2))


def _write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
