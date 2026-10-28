"""Build every results table (.tex) and figure (.pdf/.png) of the revised manuscript from the aggregated CSVs.

Nothing numeric is typed by hand: tables read results_3seed/ (aggregate_v2.py, aggregate_extra_v2.py), the
split manifest card and efficiency.json; the schedule figure calls sprat_common.schedule itself.

Usage: python make_paper_assets.py --res <results_3seed> --card <manifest card.json> --dedup <dedup_class_table.csv>
                                   --runs_tsv <runs.tsv> --out <paper dir>
Writes <out>/tables/tab_*.tex, <out>/figures/fig_*.pdf (+ .png) and <out>/supplement/*.tex.
"""
import argparse, csv, json, math, re, sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--res', required=True)
p.add_argument('--card', required=True)
p.add_argument('--dedup', required=True)
p.add_argument('--runs_tsv', required=True)
p.add_argument('--out', required=True)
args = p.parse_args()
RES, OUT = Path(args.res), Path(args.out)
for d in ('tables', 'figures', 'supplement'):
    (OUT / d).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent))
from sprat_common import full_strength_epoch, schedule  # noqa: E402

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
NAME = {'clean': 'Clean', 'naive_pgd': 'Naive PGD', 'naive_pgd_joint': 'Naive PGD-joint', 'madry_pgd': 'PGD-AT',
        'madry_w10r10': 'PGD-AT-w10r10', 'trades': 'TRADES', 'trades_w10r10': 'TRADES-w10r10', 'cat_r10': 'CAT-style',
        'sprat_w10r10': 'SPRAT', 'sprat_w10r10_joint': 'SPRAT-joint', 'sprat_w0r10': 'SPRAT-w0r10',
        'sprat_w10r10_epsonly': r'SPRAT-$\varepsilon$', 'sprat_w10r10_wonly': r'SPRAT-$\lambda$',
        'sprat_w5r5': 'SPRAT-w5r5', 'sprat_w10r5': 'SPRAT-w10r5', 'sprat_w15r5': 'SPRAT-w15r5',
        'clean_effb0': 'Clean', 'naive_pgd_effb0': 'Naive PGD', 'trades_effb0': 'TRADES',
        'sprat_w10r10_effb0': 'SPRAT'}
PLAIN = {k: v.replace(r'$\varepsilon$', 'ε').replace(r'$\lambda$', 'λ') for k, v in NAME.items()}  # figure labels
MOBILEVIT = ['clean', 'naive_pgd', 'naive_pgd_joint', 'madry_pgd', 'madry_w10r10', 'trades', 'trades_w10r10', 'cat_r10',
             'sprat_w10r10', 'sprat_w10r10_joint', 'sprat_w0r10', 'sprat_w10r10_epsonly', 'sprat_w10r10_wonly',
             'sprat_w5r5', 'sprat_w10r5', 'sprat_w15r5']
EFFB0 = ['clean_effb0', 'naive_pgd_effb0', 'trades_effb0', 'sprat_w10r10_effb0']
COLOURS = ['black', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:gray']


def class_title(name):
    """Dataset folder name -> title-case class name ('Bacterialblight' -> 'Bacterial Blight', 'Stem fly' -> 'Stem Fly')."""
    name = {'Bacterialblight': 'Bacterial blight', 'Brownspot': 'Brown spot'}.get(name, name)
    return ' '.join(w[:1].upper() + w[1:] for w in name.split())


def cls_label(cls):
    """'rice_Bacterialblight' -> 'Bacterial Blight (rice)'."""
    crop, name = cls.split('_', 1)
    return class_title(name) + (' (rice)' if crop == 'rice' else '')


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def primary_ck(cond):
    return 'best_clean' if cond.startswith('clean') else 'best_robust'


def per_seed(row, metric):
    """{seed: value in %} from a summary.csv '<metric>_per_seed' field ('42:75.10 123:...')."""
    out = {}
    for tok in row.get(metric + '_per_seed', '').split():
        s, v = tok.split(':')
        if v not in ('—', ''):
            out[int(s)] = float(v)
    return out


def pm(vals, digits=2):
    v = list(vals)
    if not v:
        return 'n/a'
    if len(v) == 1:
        return f'${v[0]:.{digits}f}$'
    return f'${np.mean(v):.{digits}f} \\pm {np.std(v, ddof=1):.{digits}f}$'


def mean_sd(vals):
    v = list(vals)
    return (float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) > 1 else (float(v[0]), 0.0)


def write(name, text):
    (OUT / name).write_text(text.strip() + '\n', encoding='utf-8')


def table(env, caption, label, colspec, header, rows, note=None, size=r'\footnotesize', tabcolsep='3pt'):
    body = '\n'.join(' & '.join(r) + r' \\' if isinstance(r, (list, tuple)) else r for r in rows)
    notes = f'\n\\vspace{{2pt}}\n\\parbox{{\\linewidth}}{{\\scriptsize {note}}}' if note else ''
    return (f'\\begin{{{env}}}[!t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n'
            f'\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n'
            f'{" & ".join(header)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}{notes}\n\\end{{{env}}}')


summary = {(r['condition'], r['checkpoint']): r for r in read_csv(RES / 'summary.csv')}
paired = read_csv(RES / 'paired.csv')
paired_final = read_csv(RES / 'paired_final.csv')
cost = read_csv(RES / 'cost.csv')
X = RES / 'extra'
aa_sum = {r['condition']: r for r in read_csv(X / 'aa_breakdown_summary.csv')}
aa_runs = read_csv(X / 'aa_breakdown.csv')
bn_sum = {r['condition']: r for r in read_csv(X / 'bn_recal_summary.csv')}
bn_runs = read_csv(X / 'bn_recal.csv')
perclass = read_csv(X / 'per_class_summary.csv')
hist = read_csv(X / 'history_long.csv')
card = json.loads(Path(args.card).read_text(encoding='utf-8'))
eff = json.loads((RES / 'efficiency.json').read_text(encoding='utf-8'))
trades_sel = json.loads((RES / 'trades_selection.json').read_text(encoding='utf-8'))


def metric(cond, ck, m):
    r = summary.get((cond, ck))
    return per_seed(r, m) if r else {}


def sm(cond, ck, m):
    """'mean ± SD' exactly as aggregate_v2.py computed it from unrounded per-seed values."""
    r = summary.get((cond, ck))
    v = r[m] if r else '—'
    return 'n/a' if v == '—' else '$' + v.replace(' ± ', ' \\pm ') + '$'


def train_hours(cond):
    return [float(r['total_train_h']) for r in cost if r['condition'] == cond]


def find_pair(rows, cond, ref, m):
    hits = [r for r in rows if r['condition'] == cond and r['reference'] == ref and r['metric'] == m]
    return hits[0] if hits else None


def seed_ci(r):
    d, h = float(r['diff_pct']), float(r['seed_t_ci95_halfwidth'])
    return d, d - h, d + h


def signs(r):
    return '$' + '/'.join(t.split(':')[1] for t in r['per_seed_diff'].split()) + '$'


def supported(r):
    d, lo, hi = seed_ci(r)
    return (lo > 0 or hi < 0) and r['same_sign_all_seeds'] == 'True'


# ──────────────────────────────────────────────────────────────────────────
# TABLES
# ──────────────────────────────────────────────────────────────────────────
# Audit of the original splits (facts from the pre-revision audit, REVISION_FINDINGS.md sections 1, 2 and 5)
write('tables/tab_audit.tex', table(
    'table', 'Audit of the original data splits used in the first submission.', 'tab:audit', 'p{5.3cm}r',
    [r'\textbf{Finding}', r'\textbf{Count}'],
    [['Wheat test images with a byte-identical file in training', '495/750 (66\\%)'],
     ['Wheat test images with an identical, rotated or flipped copy in training', '654/750 (87\\%)'],
     ['Wheat validation images with such a copy in training', '262/300 (87\\%)'],
     ['Groups of identical files carrying different class labels', '256'],
     ['Distinct rice photographs behind the 5,932 rice images', '329'],
     ['Median number of rotated or flipped variants per rice photograph', '16'],
     ['Rice images of exactly $300 \\times 300$ pixels', '4,624/5,932'],
     ['Rice images filed inside wheat class folders', '40']],
    note='Copies were identified with a rotation- and flip-invariant perceptual hash (Hamming distance at most 6) '
         'and confirmed by visual inspection.'))

dedup = {r['cls']: r for r in read_csv(args.dedup)}
rows, tot = [], np.zeros(5, dtype=int)
for c in card['per_class']:
    key = f"{c['dataset']}_{c['class_name'].lower().replace(' ', '_')}"
    d = dedup[key]
    vals = [int(d['original']), int(d['in_conflict_groups']), c['train'], c['val'], c['test']]
    tot += vals
    rows.append([c['dataset'].capitalize(), class_title(c['class_name'])] + [f'{v:,}' for v in vals])
rows.append(r'\midrule')
rows.append([r'\textbf{Total}', r'\textbf{19 classes}'] + [f'\\textbf{{{v:,}}}' for v in tot])
write('tables/tab_dataset.tex', table(
    'table', 'Composition of the leakage-controlled split. Original is the number of images in the public '
             'collection, Removed the images in groups with conflicting labels. Training keeps every distinct '
             'file of a group, validation and test keep one representative image per group.',
    'tab:dataset', 'llrrrrr',
    [r'\textbf{Crop}', r'\textbf{Class}', r'\textbf{Original}', r'\textbf{Removed}', r'\textbf{Train}',
     r'\textbf{Val}', r'\textbf{Test}'], rows, tabcolsep='2pt'))

# Conditions (from runs.tsv; full-strength epoch computed by the training code's own schedule())
OBJ = {'clean': 'CE', 'pgd': 'mixed', 'trades': 'TRADES'}
crow = {}
for line in Path(args.runs_tsv).read_text(encoding='utf-8').splitlines():
    if not line or line.startswith('#'):
        continue
    phase, name, targs = line.split('\t')
    if phase == 'tune':
        continue
    a = dict(re.findall(r'--(\w+) (\S+)', targs))
    mode, w, r = a['mode'], int(a.get('warmup', 0)), int(a.get('ramp', 0))
    what = a.get('ramp_what', 'both')
    wmax = 6.0 if mode == 'trades' else float(a.get('w_max', 0))
    obj = 'PGD-AT' if mode == 'pgd' and wmax == 1.0 else OBJ[mode]
    ramped = 'none' if mode == 'clean' or (w == 0 and r == 0) else \
        {'both': r'$\varepsilon,\lambda$', 'eps': r'$\varepsilon$', 'weight': r'$\lambda$', 'steps': '$K$'}[what]
    if mode == 'trades' and ramped != 'none':
        ramped = r'$\varepsilon,\beta$'
    fse = 'n/a' if mode == 'clean' else str(full_strength_epoch(w, r, 8 / 255, wmax, what))
    adv_ep = '0' if mode == 'clean' else str(30 - w)
    weight = 'n/a' if mode == 'clean' else (r'$\beta=6$' if mode == 'trades' else f'$\\lambda={wmax:g}$')
    bn = 'n/a' if mode == 'clean' else 'adv. only' if obj == 'PGD-AT' else a.get('bn_mode', 'separate')
    crow[name] = [NAME[name], 'EffB0' if 'effb0' in name else 'MViT-S', obj, weight, str(w), str(r), ramped, bn,
                  fse, adv_ep]
write('tables/tab_conditions.tex', table(
    'table*', 'Training conditions. Every run lasts 30 epochs with three seeds. $W$ and $R$ are the warm-up and '
              'ramp lengths in epochs, Full is the first epoch at full strength ($\\varepsilon=8/255$, $K=7$, target '
              'loss weight), Adv. is the number of epochs with adversarial examples. BN passes describes how clean '
              'and adversarial images pass through the network in training mode (separate batches, one joint batch, '
              'or adversarial images only once PGD-AT reaches full strength). TRADES uses a stop-gradient clean '
              'target. MViT-S denotes MobileViT-S and EffB0 EfficientNet-B0.',
    'tab:conditions', 'llllcclccc',
    [r'\textbf{Condition}', r'\textbf{Backbone}', r'\textbf{Objective}', r'\textbf{Weight}', '$W$', '$R$',
     r'\textbf{Ramped}', r'\textbf{BN passes}', r'\textbf{Full}', r'\textbf{Adv.}'],
    [crow[c] for c in MOBILEVIT + EFFB0]))

# Main results, MobileViT-S, primary checkpoint
M = [('clean_acc', 'Clean'), ('fgsm_8', 'FGSM'), ('pgd10_8', 'PGD-10'), ('pgd50_ce_8', 'PGD-50'),
     ('pgd50_cw_8', 'PGD-50-CW'), ('worst_case_pgd_8', 'WC-PGD'), ('autoattack_8', 'AA'), ('corrupt_mean', 'Corr.')]
rows = []
for c in MOBILEVIT:
    ck = primary_ck(c)
    rows.append([NAME[c]] + [sm(c, ck, m) for m, _ in M] + [f'{np.mean(train_hours(c)):.2f}'])
write('tables/tab_main.tex', table(
    'table*', 'Test accuracy (\\%) of MobileViT-S at the primary checkpoint (best\\_robust, clean training '
              'best\\_clean), mean $\\pm$ sample standard deviation over three seeds. All attacks at '
              '$\\varepsilon=8/255$. WC-PGD counts an image as robust only if it survives FGSM and every PGD '
              'variant, AA is standard AutoAttack, Corr. the mean accuracy over 15 corruption settings and Time '
              'the training time per run in hours.',
    'tab:main', 'l' + 'c' * 9,
    [r'\textbf{Condition}'] + [f'\\textbf{{{h}}}' for _, h in M] + [r'\textbf{Time}'], rows, size=r'\scriptsize',
    tabcolsep='2.2pt'))

# Pre-declared primary comparisons
PRIM = [('sprat_w10r10', 'naive_pgd'), ('sprat_w10r10', 'cat_r10'), ('sprat_w10r10', 'trades'),
        ('sprat_w10r10', 'madry_pgd'), ('trades_w10r10', 'trades'), ('sprat_w10r10_effb0', 'naive_pgd_effb0'),
        ('sprat_w10r10_effb0', 'trades_effb0')]
rows = []
for cond, ref in PRIM:
    first = True
    for m, lab, src in (('autoattack_8', 'AA', paired), ('clean', 'Clean', paired), ('clean', 'Clean (final)', paired_final)):
        r = find_pair(src, cond, ref, m)
        if not r:
            continue
        d, lo, hi = seed_ci(r)
        comp = f'{NAME[cond]} vs {NAME[ref]}' + (' (EffB0)' if 'effb0' in cond else '') if first else ''
        rows.append([comp, lab, f'${d:+.2f}$', f'[${lo:.2f}$, ${hi:.2f}$]', signs(r), 'yes' if supported(r) else 'no'])
        first = False
    rows.append(r'\midrule')
rows.pop()
write('tables/tab_primary.tex', table(
    'table*', 'Pre-declared primary comparisons. Diff is the mean of the per-seed accuracy differences in '
              'percentage points (condition minus reference) at the primary checkpoint, Final the same at epoch 30. '
              'The 95\\% interval is the seed-level $t$ interval with two degrees of freedom. A difference is marked '
              'supported only if this interval excludes zero and all three seeds agree in sign. Differences in '
              'corruption accuracy and worst-case PGD accuracy are listed in Table S2 of the supplementary material.',
    'tab:primary', 'llcccc',
    [r'\textbf{Comparison}', r'\textbf{Metric}', r'\textbf{Diff}', r'\textbf{95\% CI (seeds)}',
     r'\textbf{Per seed (42/123/3407)}', r'\textbf{Supported}'], rows))

# Exploratory ablations
ABL = [('sprat_w10r10_epsonly', 'sprat_w10r10'), ('sprat_w10r10_wonly', 'sprat_w10r10'), ('sprat_w0r10', 'sprat_w10r10'),
       ('sprat_w5r5', 'sprat_w10r10'), ('sprat_w10r5', 'sprat_w10r10'), ('sprat_w15r5', 'sprat_w10r10'),
       ('sprat_w10r10_joint', 'sprat_w10r10'), ('naive_pgd_joint', 'naive_pgd'), ('madry_w10r10', 'madry_pgd')]
rows = []
for cond, ref in ABL:
    cells = [NAME[cond], NAME[ref]]
    for m in ('autoattack_8', 'clean'):
        r = find_pair(paired, cond, ref, m)
        d, lo, hi = seed_ci(r)
        cells.append(f'${d:+.2f}$ [${lo:.2f}$, ${hi:.2f}$]')
    rows.append(cells)
write('tables/tab_ablation.tex', table(
    'table*', 'Exploratory schedule ablations at the primary checkpoint. Difference (condition minus reference) in '
             'AutoAttack (AA) and clean accuracy in percentage points, with the seed-level 95\\% interval in brackets.',
    'tab:ablation', 'llcc',
    [r'\textbf{Condition}', r'\textbf{Reference}', r'\textbf{AA diff [95\% CI]}', r'\textbf{Clean diff [95\% CI]}'],
    rows, size=r'\footnotesize', tabcolsep='6pt'))

# AutoAttack components
AAC = ['madry_pgd', 'madry_w10r10', 'sprat_w10r10_epsonly', 'sprat_w0r10', 'sprat_w10r10', 'sprat_w10r10_joint',
       'trades_w10r10', 'sprat_w10r10_effb0']
rows = []
for c in AAC:
    rr = [r for r in aa_runs if r['condition'] == c]
    col = lambda k: [float(r[k]) for r in rr]
    red = int(aa_sum[c]['runs_reduced_by_fab_or_square'])
    rows.append([NAME[c] + (' (EffB0)' if 'effb0' in c else ''), sm(c, 'best_robust', 'worst_case_pgd_8'),
                 pm(col('after_apgd_ce')), pm(col('after_apgd_t')), pm(col('after_fab_t')),
                 sm(c, 'best_robust', 'autoattack_8'), f'{red}/{len(rr)}'])
write('tables/tab_aa_components.tex', table(
    'table*', 'Robust accuracy (\\%) after each AutoAttack component, applied in sequence at $\\varepsilon=8/255$, '
              'next to the worst case over the PGD family (WC-PGD). Intermediate stages are read from the AutoAttack '
              'logs (two decimals per run), the final column equals the reported AutoAttack accuracy. The last column '
              'counts the seeds in which the FAB-T or Square stage lowered the accuracy further.',
    'tab:aa_components', 'lcccccc',
    [r'\textbf{Condition}', r'\textbf{WC-PGD}', r'\textbf{APGD-CE}', r'\textbf{+APGD-T}', r'\textbf{+FAB-T}',
     r'\textbf{+Square}', r'\textbf{Reduced}'], rows))

# Checkpoint selection
CK = ['naive_pgd', 'cat_r10', 'madry_pgd', 'sprat_w10r10', 'sprat_w10r10_epsonly', 'trades']
rows = []
for c in CK:
    first = True
    for ck, lab in (('best_robust', 'best\\_robust'), ('final', 'final'), ('best_clean', 'best\\_clean')):
        ep = list(metric(c, ck, 'ckpt_epoch').values())
        rows.append([NAME[c] if first else '', lab, '/'.join(str(int(e)) for e in ep),
                     sm(c, ck, 'clean_acc'), sm(c, ck, 'pgd50_ce_8'), sm(c, ck, 'worst_case_pgd_8')])
        first = False
    rows.append(r'\midrule')
rows.pop()
write('supplement/tab_ckpt.tex', table(  # supplementary Table S5 (main text refers to it by number)
    'table', 'Effect of checkpoint selection (MobileViT-S), test accuracy (\\%), mean $\\pm$ sample standard deviation '
             'over three seeds. Epochs are listed per seed (42/123/3407).',
    'tab:ckpt', 'llcccc',
    [r'\textbf{Condition}', r'\textbf{Checkpoint}', r'\textbf{Epochs}', r'\textbf{Clean}', r'\textbf{PGD-50}',
     r'\textbf{WC-PGD}'], rows, size=r'\small', tabcolsep='5pt'))

# BatchNorm recalibration
BN = ['naive_pgd', 'sprat_w10r10_wonly', 'sprat_w10r10', 'sprat_w10r10_epsonly', 'madry_pgd', 'naive_pgd_effb0',
      'sprat_w10r10_effb0', 'naive_pgd_joint', 'sprat_w10r10_joint']
bn_images = {int(r['images']) for r in bn_runs}  # eval caps the requested batches at the training-split size
assert len(bn_images) == 1, f'BN recalibration used different image counts: {bn_images}'
bn_images = bn_images.pop()
rows = []
for c in BN:
    rr = [r for r in bn_runs if r['condition'] == c]
    col = lambda k: [float(r[k]) for r in rr]
    # "before" columns from summary.csv (unrounded, identical to tab_main), "after" columns from the diagnostic
    rows.append([NAME[c] + (' (EffB0)' if 'effb0' in c else ''), sm(c, 'best_robust', 'clean_acc'),
                 pm(col('bnrecal_clean')), sm(c, 'best_robust', 'pgd20_8'), pm(col('bnrecal_pgd20_8'))])
write('supplement/tab_bnrecal.tex', table(  # supplementary Table S6
    'table', 'BatchNorm recalibration diagnostic at the primary checkpoint. Running statistics are re-estimated on '
             f'{bn_images:,} clean training images ({bn_images // 256} batches of 256). Clean and PGD-20 accuracy (\\%) '
             'before and after.',
    'tab:bnrecal', 'lcccc',
    [r'\textbf{Condition}', r'\textbf{Clean}', r'\textbf{Clean, recal.}', r'\textbf{PGD-20}', r'\textbf{PGD-20, recal.}'],
    rows, size=r'\small', tabcolsep='5pt'))

# EfficientNet-B0
rows = []
for c in EFFB0:
    ck = primary_ck(c)
    rows.append([NAME[c], sm(c, ck, 'clean_acc'), sm(c, 'final', 'clean_acc'), sm(c, ck, 'pgd50_ce_8'),
                 sm(c, ck, 'worst_case_pgd_8'), sm(c, ck, 'autoattack_8'), sm(c, ck, 'corrupt_mean')])
write('tables/tab_effb0.tex', table(
    'table*', 'EfficientNet-B0 test accuracy (\\%), mean $\\pm$ sample standard deviation over three seeds, at the '
              'primary checkpoint unless marked final. Columns as in Table~\\ref{tab:main}.', 'tab:effb0', 'lcccccc',
    [r'\textbf{Condition}', r'\textbf{Clean}', r'\textbf{Clean (final)}', r'\textbf{PGD-50}', r'\textbf{WC-PGD}',
     r'\textbf{AA}', r'\textbf{Corr.}'], rows, size=r'\footnotesize', tabcolsep='4pt'))

# Efficiency
rows = []
for key, lab, conds in (('mobilevit_s', 'MobileViT-S', ('clean', 'naive_pgd', 'sprat_w10r10', 'madry_pgd')),
                        ('efficientnet_b0', 'EfficientNet-B0', ('clean_effb0', 'naive_pgd_effb0', 'sprat_w10r10_effb0', None))):
    e = eff['models'][key]
    hrs = [f'{np.mean(train_hours(c)):.2f}' if c else 'n/a' for c in conds]
    rows.append([lab, f"{e['params_M']:.2f}", f"{e['gflops']:.2f}", f"{e['gpu_b1']['median_ms']:.2f}",
                 f"{e['cpu_b1']['median_ms']:.1f}", f"{e['gpu_b32']['images_per_s']:,.0f}",
                 f"{e['gpu_b1']['peak_mem_MB']:.1f}"] + hrs)
write('tables/tab_efficiency.tex', table(
    'table*', 'Inference cost and training time. Latency is the median over repeated batch-1 forward passes on one '
              'NVIDIA B200 MIG 1g.45gb slice (GPU) and four threads of an Intel Xeon Platinum 8570 (CPU). Throughput '
              'uses batch 32. Training time is per 30-epoch run in hours. Adversarial training does not change '
              'inference cost.',
    'tab:efficiency', 'lcccccccccc',
    [r'\textbf{Backbone}', r'\textbf{Params (M)}', r'\textbf{GFLOPs}', r'\textbf{GPU (ms)}', r'\textbf{CPU (ms)}',
     r'\textbf{Img/s}', r'\textbf{Mem. (MB)}', r'\textbf{Clean (h)}', r'\textbf{Naive (h)}', r'\textbf{SPRAT (h)}',
     r'\textbf{PGD-AT (h)}'], rows, size=r'\scriptsize', tabcolsep='2pt'))

# Supplement: TRADES tuning, per-seed values, all paired rows, per-class recall
rows = [[c['name'].replace('trades_tune_', '').replace('_', '\\_'), c['args'].replace('--w_max ', '$\\beta=').replace(
    ' --trades_detach 1', '$, stop-gradient') + ('$' if 'detach' not in c['args'] else ''), str(c['best_robust_epoch']),
         f"{c['val_pgd10'] * 100:.2f}", f"{c['val_clean'] * 100:.2f}"] for c in trades_sel['candidates']]
write('supplement/tab_trades_tuning.tex', table(
    'table', 'TRADES selection on the validation split (seed 42). Rule: highest validation PGD-10 accuracy at the '
             'best\\_robust checkpoint, ties broken by clean accuracy, candidates below 2\\% rejected.',
    'tab:trades_tuning', 'llccc', [r'\textbf{Run}', r'\textbf{Setting}', r'\textbf{Epoch}', r'\textbf{Val PGD-10}',
                                   r'\textbf{Val clean}'], rows))

ps_rows = []
for c in MOBILEVIT + EFFB0:
    for ck in ('best_robust', 'final', 'best_clean'):
        if (c, ck) not in summary:
            continue
        cells = [NAME[c] + (' (EffB0)' if 'effb0' in c else ''), ck.replace('_', '\\_')]
        for m in ('clean_acc', 'pgd50_ce_8', 'worst_case_pgd_8', 'autoattack_8'):
            v = metric(c, ck, m)
            cells.append(' / '.join(f'{v[s]:.2f}' for s in sorted(v)) if v else '--')
        ps_rows.append(cells)
write('supplement/tab_per_seed.tex', '\\begin{longtable}{llcccc}\n\\caption{Per-seed test accuracy (\\%), seeds '
      '42 / 123 / 3407.}\\label{tab:per_seed}\\\\\n\\toprule\nCondition & Checkpoint & Clean & PGD-50 & WC-PGD & AA '
      '\\\\\n\\midrule\n' + '\n'.join(' & '.join(r) + r' \\' for r in ps_rows) + '\n\\bottomrule\n\\end{longtable}')

pr_rows = []
for r in paired:
    d, lo, hi = seed_ci(r)
    pr_rows.append([NAME[r['condition']] + (' (EffB0)' if 'effb0' in r['condition'] else ''), NAME[r['reference']],
                    'yes' if r['primary'] == 'True' else 'no', r['metric'].replace('_', '\\_'), f'${d:+.2f}$',
                    f'[${lo:.2f}$, ${hi:.2f}$]', f"[{r['ci95_two_level_low']}, {r['ci95_two_level_high']}]", signs(r)])
write('supplement/tab_paired_all.tex', '\\begin{longtable}{lllllccl}\n\\caption{All paired comparisons at the '
      'primary checkpoint: seed-level $t$ interval and two-level bootstrap range.}\\label{tab:paired_all}\\\\\n'
      '\\toprule\nCondition & Reference & Primary & Metric & Diff & Seed CI & Bootstrap & Per seed \\\\\n\\midrule\n'
      + '\n'.join(' & '.join(r) + r' \\' for r in pr_rows) + '\n\\bottomrule\n\\end{longtable}')

PC = ['clean', 'sprat_w10r10', 'sprat_w10r10_epsonly', 'madry_pgd']
classes = [r['cls'] for r in perclass if r['condition'] == 'clean' and r['eval'] == 'clean']
pc_rows = []
for cls in classes:
    sup = next(r['support'] for r in perclass if r['cls'] == cls)
    cells = [cls_label(cls), sup]
    for c in PC:
        for ev in ('clean', 'autoattack_8'):
            hit = [r for r in perclass if r['condition'] == c and r['eval'] == ev and r['cls'] == cls]
            cells.append(hit[0]['recall'].split(' ')[0] if hit else '--')
    pc_rows.append(cells)
write('supplement/tab_per_class.tex', '\\begin{longtable}{lr' + 'cc' * len(PC) + '}\n\\caption{Per-class recall '
      '(\\%), mean over three seeds, clean and AutoAttack, primary checkpoint.}\\label{tab:per_class}\\\\\n\\toprule\n'
      'Class & $n$ & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{NAME[c]}}}' for c in PC) + ' \\\\\n & & '
      + ' & '.join(['Clean & AA'] * len(PC)) + ' \\\\\n\\midrule\n' + '\n'.join(' & '.join(r) + r' \\' for r in pc_rows)
      + '\n\\bottomrule\n\\end{longtable}')


# ──────────────────────────────────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────────────────────────────────
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cbook import STEP_LOOKUP_MAP  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.text import Annotation, Text  # noqa: E402
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.7, 'lines.linewidth': 1.2,
                     'legend.frameon': False, 'pdf.fonttype': 42})
LAYOUT_ISSUES = []


def _sample(xy, step=1.0):
    """Points every `step` display pixels along a polyline (display coordinates)."""
    pts = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(2, int(np.hypot(*(b - a)) / step) + 1)
        pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.concatenate(pts) if len(xy) else np.empty((0, 2))


def _obstacles(ax):
    """Display-space points of everything drawn as data in `ax` (lines, markers, error bars, bars)."""
    pts, boxes = [], []
    for ln in ax.get_lines():
        if not ln.get_visible():
            continue
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        ds = ln.get_drawstyle()
        if ds in STEP_LOOKUP_MAP and ds != 'default' and len(x) > 1:
            x, y = STEP_LOOKUP_MAP[ds](x, y)
        xy = ln.get_transform().transform(np.column_stack([x, y]))
        xy = xy[np.all(np.isfinite(xy), 1)]
        if ln.get_linestyle() not in ('None', '', ' ') and len(xy) > 1:
            pts.append(_sample(xy))
        if ln.get_marker() not in (None, 'None', '', ' '):
            pts.append(xy)
    for coll in ax.collections:
        if isinstance(coll, LineCollection) and coll.get_visible():
            for seg in coll.get_segments():
                if len(seg) > 1:
                    pts.append(_sample(coll.get_transform().transform(np.asarray(seg, float))))
    for pa in ax.patches:
        if isinstance(pa, Rectangle) and pa.get_visible() and pa.get_width() > 0:
            boxes.append(pa.get_window_extent())
    return (np.concatenate(pts) if pts else np.empty((0, 2))), boxes


def check_layout(fig, name):
    """Record overlapping text, text crossing plotted data, text outside the figure and text overflowing boxes."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    all_ticklabels, drawn_ticklabels = set(), set()  # tick labels outside the view interval exist but are not drawn
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.majorTicks + axis.minorTicks:
                all_ticklabels |= {id(tk.label1), id(tk.label2)}
            for tk in (axis._update_ticks() if ax.axison else []):  # axis('off') draws no tick labels
                drawn_ticklabels |= {id(tk.label1), id(tk.label2)}
    texts = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or t.figure is None:
            continue
        if id(t) in all_ticklabels and id(t) not in drawn_ticklabels:
            continue
        bb = Text.get_window_extent(t, r)  # text only (an Annotation's own extent also covers its leader line)
        if bb.width > 0 and bb.height > 0:
            texts.append((t, bb))
    issues = []
    for i, (a, ba) in enumerate(texts):
        if ba.x0 < fb.x0 - 0.5 or ba.y0 < fb.y0 - 0.5 or ba.x1 > fb.x1 + 0.5 or ba.y1 > fb.y1 + 0.5:
            issues.append(f'outside figure: "{a.get_text()[:30]}"')
        for b, bb in texts[i + 1:]:
            if ba.shrunk(0.98, 0.9).overlaps(bb.shrunk(0.98, 0.9)):
                issues.append(f'text overlap: "{a.get_text()[:30]}" / "{b.get_text()[:30]}"')
    for ax in fig.axes:
        for pa in ax.patches:  # boxes and bars must not be clipped by the axes or the figure edge
            pb = pa.get_window_extent(r)
            lim = ax.bbox if pa.get_clip_on() else fb
            if pa.get_visible() and pb.width > 0 and (pb.x0 < lim.x0 - 0.5 or pb.x1 > lim.x1 + 0.5 or
                                                      pb.y0 < lim.y0 - 0.5 or pb.y1 > lim.y1 + 0.5):
                issues.append(f'shape clipped at the edge ({type(pa).__name__})')
        pts, bars = _obstacles(ax)
        labels = [(t, Text.get_window_extent(t, r)) for t in ax.texts if t.get_visible() and t.get_text().strip()]
        leg = ax.get_legend()
        if leg is not None:
            labels.append((leg, leg.get_window_extent(r)))
        frames = [pa.get_window_extent() for pa in ax.patches if isinstance(pa, FancyBboxPatch)]
        for t, bb in labels:
            what = t.get_text()[:30] if isinstance(t, Text) else 'legend'
            inner = bb.expanded(1.02, 1.04)  # a small margin so text does not touch data either
            if len(pts):
                inside = (pts[:, 0] > inner.x0) & (pts[:, 0] < inner.x1) & (pts[:, 1] > inner.y0) & (pts[:, 1] < inner.y1)
                if inside.any():
                    issues.append(f'text crosses data: "{what}" ({int(inside.sum())} px)')
            if any(bb.overlaps(b) for b in bars):
                issues.append(f'text overlaps bars: "{what}"')
            for fr in frames:
                cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
                if fr.x0 < cx < fr.x1 and fr.y0 < cy < fr.y1 and not (
                        bb.x0 >= fr.x0 + 2 and bb.x1 <= fr.x1 - 2 and bb.y0 >= fr.y0 + 2 and bb.y1 <= fr.y1 - 2):
                    issues.append(f'text overflows box: "{what}"')
    for msg in dict.fromkeys(issues):
        LAYOUT_ISSUES.append(f'{name}: {msg}')
    print(f'  layout {name}: {"OK" if not issues else str(len(set(issues))) + " issue(s)"}')


def save(fig, name, sub='figures', tight=True):
    if tight:
        fig.tight_layout()
    check_layout(fig, name)
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / sub / f'{name}.{ext}', dpi=300)
    plt.close(fig)


# Workflow
fig, ax = plt.subplots(figsize=(7.16, 1.9))
ax.set_xlim(-0.5, 100.5); ax.set_ylim(0, 23); ax.axis('off')
BW, BH, STEP = 17.3, 8.5, 20.25  # box width / height and spacing in axis units
top = ['Public collections\nwheat 14,154 images\nrice 5,932 images',
       'Near-duplicate audit\ndihedral-invariant pHash\nHamming distance $\\leq$ 6',
       'Group-aware split\n10,721 / 1,443 / 1,441\nreleased manifest',
       '20 conditions, 3 seeds\n30 epochs, MobileViT-S\nand EfficientNet-B0',
       'Checkpoints\nbest_robust (primary)\nfinal, best_clean']
bottom = ['Test evaluation\nFGSM, PGD, CW loss\nAutoAttack, corruptions',
          'Diagnostics\nAutoAttack components\nBN recalibration, cost',
          'Statistics\nseed-level $t$ intervals\nbootstrap, McNemar']
xs = [0.5 + STEP * i for i in range(5)]
for x, txt in zip(xs, top):
    ax.add_patch(FancyBboxPatch((x, 13), BW, BH, boxstyle='round,pad=0.25', fc='#f2f2f2', ec='black', lw=0.7))
    ax.text(x + BW / 2, 13 + BH / 2, txt, ha='center', va='center', fontsize=6.0)
for x, txt in zip(xs[::-1], bottom):
    ax.add_patch(FancyBboxPatch((x, 1), BW, BH, boxstyle='round,pad=0.25', fc='#f2f2f2', ec='black', lw=0.7))
    ax.text(x + BW / 2, 1 + BH / 2, txt, ha='center', va='center', fontsize=6.0)
arrow = dict(arrowstyle='-|>', lw=0.8, color='black', mutation_scale=7)
for x in xs[:-1]:
    ax.annotate('', xy=(x + STEP - 0.4, 13 + BH / 2), xytext=(x + BW + 0.4, 13 + BH / 2), arrowprops=arrow)
ax.annotate('', xy=(xs[-1] + BW / 2, 1 + BH + 0.5), xytext=(xs[-1] + BW / 2, 12.5), arrowprops=arrow)
for x in xs[-1:-3:-1]:
    ax.annotate('', xy=(x - STEP + BW + 0.4, 1 + BH / 2), xytext=(x - 0.4, 1 + BH / 2), arrowprops=arrow)
save(fig, 'fig_workflow')

# Schedules (computed by the training code's own schedule())
E = np.arange(1, 31)
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
for (w, r), col in zip([(0, 10), (5, 5), (10, 5), (10, 10), (15, 5)], COLOURS):
    axes[0].step(E, [schedule(e, w, r, 8 / 255, 0.5)[0] * 255 for e in E], where='mid', color=col, label=f'w{w}r{r}')
axes[0].step(E, [8] * 30, where='mid', color='tab:gray', ls=':', label='Naive PGD')
axes[0].set_ylabel(r'$\varepsilon_e \times 255$')
axes[0].legend(fontsize=5.5, loc='lower right', handlelength=1.2, labelspacing=0.25, borderaxespad=0.3)
# (b) is the same for SPRAT, SPRAT-eps and SPRAT-lambda up to which quantity ramps; the caption names them
axes[1].step(E, [schedule(e, 10, 10, 8 / 255, 0.5, 'eps')[0] / (8 / 255) for e in E], where='mid', color='black',
             label='ramped')
axes[1].step(E, [schedule(e, 10, 10, 8 / 255, 0.5, 'eps')[1] / 0.5 for e in E], where='mid', color='tab:red', ls='--',
             label='held at target')
axes[1].set_ylabel('Fraction of target value')
axes[1].legend(fontsize=5.5, loc='lower right', handlelength=1.6, borderaxespad=0.3)
axes[2].step(E, [schedule(e, 0, 10, 8 / 255, 0.5, 'steps')[2] for e in E], where='mid', color='black', label='CAT-style')
axes[2].step(E, [schedule(e, 10, 10, 8 / 255, 0.5)[2] for e in E], where='mid', color='tab:blue', ls='--', label='SPRAT')
axes[2].set_ylabel('PGD steps $K_e$')
axes[2].legend(fontsize=5.5, loc='lower right', handlelength=1.6, borderaxespad=0.3)
for a, t in zip(axes, ('(a)', '(b)', '(c)')):
    a.set_xlabel('Epoch'); a.grid(alpha=0.3, lw=0.4); a.set_title(t, fontsize=8)
save(fig, 'fig_schedules')

# Clean vs AutoAttack: (a) MobileViT-S, (b) EfficientNet-B0. Labels are placed in data coordinates; points at 0 %
# AutoAttack get leader lines to two label rows below the axis (y = -4.5 and -9), arranged so no leader crosses a label.
LAB = {  # condition: (x, y, horizontal alignment, leader line?)
    'madry_pgd': (64.8, 33.3, 'left', False), 'madry_w10r10': (71.2, 29.6, 'left', False),
    'sprat_w10r10_epsonly': (77.8, 22.6, 'left', False), 'sprat_w0r10': (73.3, 20.24, 'right', False),
    'sprat_w10r10': (73.0, 13.67, 'right', False), 'sprat_w10r10_joint': (83.0, 7.8, 'left', False),
    'trades_w10r10': (59.0, 9.4, 'right', False), 'trades': (36.5, 2.01, 'right', False),
    'sprat_w10r5': (59.4, 2.6, 'left', False),
    'sprat_w10r10_wonly': (16.6, -4.0, 'center', True), 'naive_pgd': (52.0, -4.0, 'center', True),
    'cat_r10': (79.0, -4.0, 'center', True), 'clean': (92.0, -4.0, 'center', True),
    'sprat_w15r5': (60.0, -8.5, 'center', True), 'sprat_w5r5': (74.0, -8.5, 'center', True),
    'naive_pgd_joint': (89.5, -8.5, 'center', True),
    'sprat_w10r10_effb0': (73.0, 14.16, 'right', False), 'trades_effb0': (12.0, -4.5, 'center', True),
    'naive_pgd_effb0': (26.0, -9.0, 'center', True), 'clean_effb0': (86.0, -4.5, 'center', True)}
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True, gridspec_kw=dict(width_ratios=(2, 1)))
for ax, conds, title in ((axes[0], MOBILEVIT, '(a) MobileViT-S'), (axes[1], EFFB0, '(b) EfficientNet-B0')):
    pts = []
    for c in conds:
        ck = primary_ck(c)
        (mx, sx), (my, sy) = mean_sd(metric(c, ck, 'clean_acc').values()), mean_sd(metric(c, ck, 'autoattack_8').values())
        col = 'tab:blue' if c.startswith('sprat') else 'tab:red' if c.startswith('madry') else 'black'
        ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt='o', ms=3.2, color=col, elinewidth=0.6, capsize=1.5, zorder=3)
        pts.append((mx, my))
        lx, ly, ha, lead = LAB[c]
        ax.annotate(PLAIN[c], (mx, my), xytext=(lx, ly), textcoords='data', ha=ha, va='top' if lead else 'center',
                    fontsize=5.8, arrowprops=dict(arrowstyle='-', lw=0.4, color='gray', shrinkA=0, shrinkB=2)
                    if lead else None)
    if ax is axes[0]:
        front = sorted(p for p in pts if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in pts))
        ax.plot(*zip(*front), color='gray', ls='--', lw=0.7, zorder=1)
    ax.set_title(title, fontsize=8); ax.set_xlabel('Clean accuracy (%)'); ax.grid(alpha=0.3, lw=0.4)
    ax.axhline(0, color='black', lw=0.4, zorder=0)
axes[0].set_xlim(0, 99); axes[1].set_xlim(-5, 96)
axes[0].set_ylim(-12, 36); axes[0].set_yticks(range(0, 36, 5))
axes[0].set_ylabel('AutoAttack accuracy (%), ε = 8/255')
save(fig, 'fig_pareto')

# Validation curves
SHOW = ['naive_pgd', 'cat_r10', 'madry_pgd', 'sprat_w10r10', 'sprat_w10r10_epsonly', 'sprat_w10r10_wonly', 'sprat_w5r5']
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
for c, col in zip(SHOW, COLOURS):
    for ax, key, sc in zip(axes, ('val_clean_acc', 'val_pgd_acc', 'grad_norm_mean'), (100, 100, 1)):
        by_seed = {}
        for h in hist:
            if h['condition'] == c:
                by_seed.setdefault(h['seed'], []).append(float(h[key]) * sc)
        Mx = np.array(list(by_seed.values()))
        ax.plot(E, Mx.mean(0), color=col, label=PLAIN[c])
        ax.fill_between(E, Mx.min(0), Mx.max(0), color=col, alpha=0.12, lw=0)
for ax, (t, yl) in zip(axes, (('(a) Validation clean accuracy', 'Accuracy (%)'),
                              ('(b) Validation PGD-10 accuracy', 'Accuracy (%)'),
                              ('(c) Gradient norm before clipping', 'Norm'))):
    ax.set_title(t, fontsize=7.5); ax.set_xlabel('Epoch'); ax.set_ylabel(yl); ax.grid(alpha=0.3, lw=0.4)
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=len(SHOW), fontsize=6, handlelength=1.6)
fig.tight_layout(rect=(0, 0.09, 1, 1))
save(fig, 'fig_curves', tight=False)

# Per-class AutoAttack recall
fig, ax = plt.subplots(figsize=(3.5, 4.4))
y = np.arange(len(classes))
for i, (c, col) in enumerate(zip(['sprat_w10r10', 'sprat_w10r10_epsonly', 'madry_pgd'], ['tab:blue', 'tab:green', 'tab:red'])):
    v = [float(next(r['recall'] for r in perclass if r['condition'] == c and r['eval'] == 'autoattack_8'
                    and r['cls'] == cls).split(' ')[0]) for cls in classes]
    ax.barh(y + (i - 1) * 0.27, v, height=0.27, color=col, label=PLAIN[c])
ax.set_yticks(y); ax.set_yticklabels([cls_label(cls) for cls in classes], fontsize=6)
ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel('AutoAttack recall (%), ε = 8/255'); ax.grid(axis='x', alpha=0.3, lw=0.4)
ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, handlelength=1.2)
save(fig, 'fig_perclass', sub='supplement/figures')  # supplementary Fig. S3

# Supplement: training loss terms (PGD-AT logs no clean term because it is at full strength from epoch 1)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7))
for c, col in zip(SHOW, COLOURS):
    for ax, key in zip(axes, ('train_clean_ce', 'train_adv_term')):
        by_seed = {}
        for h in hist:
            if h['condition'] == c:
                by_seed.setdefault(h['seed'], []).append(float(h[key]) if h[key] not in ('', 'None') else np.nan)
        Mx = np.array(list(by_seed.values()))
        if np.all(np.isnan(Mx)):
            continue
        ax.plot(E, np.nanmean(Mx, 0), color=col, label=PLAIN[c])
        ax.fill_between(E, np.nanmin(Mx, 0), np.nanmax(Mx, 0), color=col, alpha=0.12, lw=0)
for ax, t in zip(axes, ('(a) Clean cross-entropy term', '(b) Adversarial term')):
    ax.set_title(t, fontsize=7.5); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(alpha=0.3, lw=0.4)
fig.legend(*axes[1].get_legend_handles_labels(), loc='lower center', ncol=len(SHOW), fontsize=6, handlelength=1.6)
fig.tight_layout(rect=(0, 0.09, 1, 1))
save(fig, 'fig_loss', sub='supplement/figures', tight=False)

# Supplement: confusion matrices summed over seeds, one 2 x 2 figure with a shared colour scale
conf = json.loads((X / 'confusion_summed.json').read_text(encoding='utf-8'))
PANELS = [('clean', 'clean', '(a) Clean training, clean images'), ('sprat_w10r10', 'clean', '(b) SPRAT, clean images'),
          ('sprat_w10r10', 'autoattack_8', '(c) SPRAT, AutoAttack'), ('madry_pgd', 'autoattack_8', '(d) PGD-AT, AutoAttack')]
fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.3))
for ax, (c, ev, title) in zip(axes.flat, PANELS):
    cm = np.array(conf[c][ev]['matrix'], dtype=float)
    cm = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)  # rows = true class, % of its test images
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=100)
    lab = [cls_label(n).replace(' (rice)', ' (R)') for n in conf[c][ev]['classes']]
    ax.set_xticks(range(len(lab))); ax.set_yticks(range(len(lab)))
    ax.set_xticklabels(lab if ax in axes[1] else [], rotation=90, fontsize=5.5)
    ax.set_yticklabels(lab if ax in axes[:, 0] else [], fontsize=5.5)
    ax.tick_params(length=2)
    ax.set_title(title, fontsize=7.5)
for ax in axes[1]:
    ax.set_xlabel('Predicted class', fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel('True class', fontsize=7)
fig.tight_layout(rect=(0, 0, 0.87, 1))
cax = fig.add_axes([0.885, 0.3, 0.015, 0.4])
fig.colorbar(im, cax=cax).set_label('% of true-class test images', fontsize=7)
cax.tick_params(labelsize=6)
save(fig, 'fig_confusion', sub='supplement/figures', tight=False)

print('assets written to', OUT)
if LAYOUT_ISSUES:
    print('LAYOUT ISSUES:\n  ' + '\n  '.join(LAYOUT_ISSUES))
else:
    print('layout check: no overlaps in any figure')


