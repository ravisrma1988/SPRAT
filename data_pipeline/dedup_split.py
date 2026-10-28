"""Build a leakage-free, group-aware train/val/test manifest for the wheat + rice benchmark.

1. Hash every image (DCT pHash, 8 dihedral variants) - same method as leakage_audit.py.
2. Union-find: images within Hamming <= NEAR_T under any rotation/flip form one group
   (wheat and rice pooled, so cross-crop copies are caught too).
3. Drop groups whose members carry more than one class label (label conflicts,
   including rice images filed under wheat classes).
4. Keep one representative per group (the group's largest file, i.e. least
   down-sampled) - duplicates add no information and inflate class counts.
5. Stratified split of groups per class: 70/15/15 with a fixed seed.

Original wheat train/valid/test folders are ignored: they are what leaked.

Usage: python dedup_split.py <wheat_root> <rice_root> <out_dir>
"""
import csv, json, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leakage_audit import NEAR_T, canon_class, collect, hash_one  # noqa: E402

SPLIT_SEED = 20261023
FRACS = (0.70, 0.15, 0.15)


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[b] = a


def main():
    wheat_root, rice_root, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    items = collect(wheat_root, rice_root)
    cache = out / 'hashes.npy'
    if cache.exists():
        H = np.load(cache)
        meta = json.loads((out / 'hash_meta.json').read_text())
    else:
        with ProcessPoolExecutor() as ex:
            res = list(ex.map(hash_one, [i['path'] for i in items], chunksize=64))
        H = np.array([r['hashes'] for r in res], dtype=np.uint64)
        meta = [dict(md5=r['md5'], bytes=r['bytes'], w=r['w'], h=r['h']) for r in res]
        np.save(cache, H); (out / 'hash_meta.json').write_text(json.dumps(meta))
    for it, m in zip(items, meta):
        it.update(m); it['cls'] = canon_class(it['crop'], it['cls_folder'])

    n = len(items); canon = H[:, 0]; dsu = DSU(n)
    for a in range(n):  # min over a's 8 variants vs every canonical hash
        d = np.bitwise_count(H[a][:, None] ^ canon[None, :]).min(axis=0)
        for b in np.flatnonzero(d <= NEAR_T):
            if b != a: dsu.union(a, int(b))
        if a % 2000 == 0: print(f'  union {a}/{n}', flush=True)

    groups = defaultdict(list)
    for i in range(n): groups[dsu.find(i)].append(i)
    conflict = {g for g, m in groups.items() if len({items[i]['cls'] for i in m}) > 1}
    clean = {g: m for g, m in groups.items() if g not in conflict}

    # one representative per group: largest pixel count, then largest file
    reps = {g: max(m, key=lambda i: (items[i]['w'] * items[i]['h'], items[i]['bytes'])) for g, m in clean.items()}

    rng = np.random.default_rng(SPLIT_SEED)
    by_cls = defaultdict(list)
    for g, i in reps.items(): by_cls[items[i]['cls']].append(g)
    split_of = {}
    for c in sorted(by_cls):
        gs = sorted(by_cls[c], key=lambda g: items[reps[g]]['path']); rng.shuffle(gs)
        n_tr = int(round(FRACS[0] * len(gs))); n_va = int(round(FRACS[1] * len(gs)))
        for k, g in enumerate(gs):
            split_of[g] = 'train' if k < n_tr else ('val' if k < n_tr + n_va else 'test')

    # train keeps every member of its groups (augmented copies are legitimate training data);
    # val/test keep only the representative so no scene is counted more than once.
    rows = []
    for g in sorted(reps, key=lambda g: items[reps[g]]['path']):
        members = clean[g] if split_of[g] == 'train' else [reps[g]]
        for i in sorted(members, key=lambda i: items[i]['path']):
            it = items[i]
            if split_of[g] == 'train' and it['md5'] in {items[j]['md5'] for j in members if j < i}:
                continue  # byte-identical copy within the group adds nothing
            rows.append(dict(split=split_of[g], cls=it['cls'], crop=it['crop'], path=it['path'], md5=it['md5'],
                             group_id=g, group_size=len(clean[g]), is_rep=int(i == reps[g]),
                             orig_split=it['split']))
    with open(out / 'manifest_dedup.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    drop_rows = [dict(cls=items[i]['cls'], crop=items[i]['crop'], path=items[i]['path'], group_id=g,
                      group_classes='|'.join(sorted({items[j]['cls'] for j in groups[g]})))
                 for g in conflict for i in groups[g]]
    with open(out / 'dropped_label_conflicts.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(drop_rows[0].keys())); w.writeheader(); w.writerows(drop_rows)

    orig = Counter(it['cls'] for it in items)
    uniq = Counter(items[i]['cls'] for i in reps.values())
    per = Counter((r['cls'], r['split']) for r in rows)
    lost_conf = Counter(r['cls'] for r in drop_rows)
    table = [dict(cls=c, original=orig[c], in_conflict_groups=lost_conf.get(c, 0), unique_kept=uniq[c],
                  train=per[(c, 'train')], val=per[(c, 'val')], test=per[(c, 'test')]) for c in sorted(orig)]
    with open(out / 'dedup_class_table.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)

    summ = dict(images=n, groups=len(groups), conflict_groups=len(conflict),
                images_in_conflict_groups=len(drop_rows), unique_scenes_kept=len(reps), manifest_rows=len(rows),
                split_counts=dict(Counter(r['split'] for r in rows)), near_threshold=NEAR_T, split_seed=SPLIT_SEED)
    (out / 'dedup_summary.json').write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    print(f"{'class':34s} {'orig':>6s} {'confl':>6s} {'uniq':>6s} {'tr':>5s} {'va':>5s} {'te':>5s}")
    for t in table:
        print(f"{t['cls']:34s} {t['original']:6d} {t['in_conflict_groups']:6d} {t['unique_kept']:6d} "
              f"{t['train']:5d} {t['val']:5d} {t['test']:5d}")


if __name__ == '__main__':
    main()
