"""Convert dedup/manifest_dedup.csv (local Windows paths) into the canonical DGX manifest.

- Paths become relative to the DGX project root (/workspace/ravi/SPRAT), same convention
  as the frozen data/manifest/combined_manifest.csv.
- class_id / class_name / dataset are taken from the frozen manifest by path, so the
  class index order is unchanged from the original experiments.
- Any path in the DGX audit's cross-label quarantine queue is excluded together with its
  whole duplicate group.
- Writes manifest_sprat_v2_dedup.csv plus a JSON card (SHA-256, counts, provenance).

Usage: python build_canonical_manifest.py <dedup_csv> <frozen_manifest_csv> <quarantine_csv> <out_dir>
"""
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path

LOCAL_ROOTS = {
    'wheat': ('C:\\Users\\admin\\Downloads\\archive\\data\\', 'data/dataset/wheat/data/'),
    'rice': ('C:\\Users\\admin\\Downloads\\fwcj7stb8r-1\\Rice Leaf Disease Images\\',
             'data/dataset/rice/Rice Leaf Disease Images/'),
}

dedup_csv, frozen_csv, quar_csv, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)

frozen = {r['path']: r for r in csv.DictReader(open(frozen_csv, encoding='utf-8'))}
quar = {r['path'] for r in csv.DictReader(open(quar_csv, encoding='utf-8'))}


def to_rel(crop, p):
    pre, new = LOCAL_ROOTS[crop]
    assert p.startswith(pre), p
    return new + p[len(pre):].replace('\\', '/')


rows = []
for r in csv.DictReader(open(dedup_csv, encoding='utf-8')):
    rel = to_rel(r['crop'], r['path'])
    f = frozen[rel]  # KeyError here would mean the DGX tree differs from local
    rows.append(dict(path=rel, dataset=f['dataset'], class_name=f['class_name'], class_id=int(f['class_id']),
                     split=r['split'], group_id=r['group_id'], group_size=int(r['group_size']),
                     is_rep=int(r['is_rep']), md5=r['md5'], orig_split=f['split']))

# quarantine: drop whole groups touched by the DGX cross-label queue
qgroups = {r['group_id'] for r in rows if r['path'] in quar}
already_out = sorted(quar - {r['path'] for r in rows})
rows = [r for r in rows if r['group_id'] not in qgroups]

# integrity checks
ids_by_name = Counter((r['dataset'], r['class_name'], r['class_id']) for r in rows)
assert len({(d, n) for d, n, _ in ids_by_name}) == len({i for _, _, i in ids_by_name}) == 19
g_split = {}
for r in rows:
    assert g_split.setdefault(r['group_id'], r['split']) == r['split'], 'group crosses splits'
md5_split = {}
for r in rows:
    s = md5_split.setdefault(r['md5'], r['split'])
    assert s == r['split'], 'identical file in two splits'
assert all(r['is_rep'] == 1 for r in rows if r['split'] != 'train')

rows.sort(key=lambda r: (r['split'], r['class_id'], r['path']))
cols = ['path', 'dataset', 'class_name', 'class_id', 'split', 'group_id', 'group_size', 'is_rep', 'md5', 'orig_split']
mf = out / 'manifest_sprat_v2_dedup.csv'
with open(mf, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n'); w.writeheader(); w.writerows(rows)
sha = hashlib.sha256(mf.read_bytes()).hexdigest()

per = Counter((r['class_id'], r['class_name'], r['dataset'], r['split']) for r in rows)
classes = sorted({(r['class_id'], r['dataset'], r['class_name']) for r in rows})
card = dict(
    manifest=mf.name, sha256=sha, rows=len(rows), split_counts=dict(Counter(r['split'] for r in rows)),
    groups=len({r['group_id'] for r in rows}),
    quarantine_paths_in_dgx_queue=len(quar), quarantine_paths_already_excluded=len(already_out),
    quarantine_groups_dropped_now=len(qgroups),
    method='dihedral-invariant 64-bit DCT pHash, Hamming<=6, single-linkage groups over wheat+rice; '
           'groups with >1 class label dropped; groups split 70/15/15 per class (seed 20261023); '
           'train keeps all distinct-file members of a group, val/test keep one representative per group',
    per_class=[dict(class_id=i, dataset=d, class_name=n,
                    train=per[(i, n, d, 'train')], val=per[(i, n, d, 'val')], test=per[(i, n, d, 'test')])
               for i, d, n in classes])
(out / 'manifest_sprat_v2_dedup.card.json').write_text(json.dumps(card, indent=2))
print(json.dumps({k: v for k, v in card.items() if k != 'per_class'}, indent=2))
for c in card['per_class']:
    print(f"  {c['class_id']:2d} {c['dataset']:5s} {c['class_name']:22s} {c['train']:5d} {c['val']:4d} {c['test']:4d}")
