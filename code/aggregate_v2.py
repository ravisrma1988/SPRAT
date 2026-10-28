"""Aggregate SPRAT v2 runs into paper tables, paired statistics and training-curve figures.

Run directories are named <condition>_s<seed> and contain config.json, history.json,
train_summary.json and eval_<checkpoint>/{eval.json, per_sample_preds.npz}.

Outputs (in --out):
  summary.csv / summary.md   mean ± SD over seeds per condition and checkpoint, per-seed values, n seeds
  paired.csv                 condition vs reference: mean accuracy difference, two-level bootstrap 95% CI
                             (seeds resampled, then test images), per-seed differences, seed-level t
                             interval, sign consistency, exact McNemar p per seed (descriptive).
                             `primary` marks the pre-declared comparisons (RUN_PLAN.md).
  per_crop.csv, shortcut.csv per-crop accuracy; cross-crop error rate (crop/source shortcut check)
  cost.csv                   training time per epoch (warm-up vs adversarial epochs), total hours
  curves_<group>.png         val clean / PGD accuracy and gradient norm per epoch (mean, min–max over seeds)
  warnings.txt               provenance problems found while loading

Worst-case robust accuracy at budget e: a test image counts as robust only if it is classified
correctly under every gradient attack evaluated at e (FGSM + all PGD variants) — `worst_case_pgd`;
`worst_case_all` additionally requires AutoAttack and is only defined where AutoAttack was run.

Usage: python aggregate_v2.py --runs /workspace/ravi/SPRAT/outputs/v2_runs --out .../v2_report
"""
import argparse, csv, json, math, re
from collections import defaultdict
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--runs', required=True)
p.add_argument('--out', required=True)
p.add_argument('--ckpts', default='best_robust,final,best_clean')
p.add_argument('--boot', type=int, default=2000)
p.add_argument('--eps', type=float, default=8.0, help='budget (x/255) for worst-case robust accuracy')
p.add_argument('--allow_debug_evals', action='store_true', help='smoke tests only: accept --max_n evaluations')
args = p.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
E = f'{args.eps:g}'
warnings = []

PRIMARY = {('sprat_w10r10', 'naive_pgd'), ('sprat_w10r10', 'trades'), ('sprat_w10r10', 'cat_r10'),
           ('sprat_w10r10', 'madry_pgd'), ('trades_w10r10', 'trades'), ('sprat_w10r10_effb0', 'naive_pgd_effb0'),
           ('sprat_w10r10_effb0', 'trades_effb0')}

# ──────────────────────────────────────────────────────────────────────────
# LOAD + PROVENANCE CHECKS
# ──────────────────────────────────────────────────────────────────────────
runs = defaultdict(dict)  # cond -> seed -> dict
ref_paths, ref_n = None, None
for d in sorted(Path(args.runs).iterdir()):
    m = re.fullmatch(r'(.+)_s(\d+)', d.name)
    if not (m and (d / 'config.json').exists()):
        continue
    cond, seed = m.group(1), int(m.group(2))
    r = dict(dir=d, config=json.loads((d / 'config.json').read_text()))
    r['history'] = json.loads((d / 'history.json').read_text()) if (d / 'history.json').exists() else []
    r['evals'] = {}
    for ck in args.ckpts.split(','):
        e = d / f'eval_{ck}'
        if not (e / 'eval.json').exists():
            continue
        js = json.loads((e / 'eval.json').read_text())
        pr = dict(np.load(e / 'per_sample_preds.npz', allow_pickle=False))
        if js['split'] != 'test' or (js['args'].get('max_n') and not args.allow_debug_evals):
            warnings.append(f'{d.name}/{ck}: not a full test-split evaluation; skipped'); continue
        if ref_paths is None:
            ref_paths, ref_n = pr['paths'], js['n']
        elif js['n'] != ref_n or not np.array_equal(pr['paths'], ref_paths):
            warnings.append(f'{d.name}/{ck}: test set differs from the first run; skipped'); continue
        r['evals'][ck] = dict(json=js, preds=pr)
    runs[cond][seed] = r
for cond, seeds in runs.items():
    for key in ('script_sha256', 'common_sha256', 'manifest_sha256'):
        vals = {s: r['config'].get(key) for s, r in seeds.items()}
        if len(set(vals.values())) > 1:
            warnings.append(f'{cond}: {key} differs across seeds {vals}')
print(f'loaded {sum(len(v) for v in runs.values())} runs in {len(runs)} conditions; {len(warnings)} warnings')


def keys_at(js, with_aa):
    ks = list(js.get('attack_keys', {}).get(E, []))
    return [k for k in ks if with_aa or not k.startswith('autoattack')]


def worst_case_correct(js, pr, with_aa):
    ks = keys_at(js, with_aa)
    if not ks or (with_aa and not any(k.startswith('autoattack') for k in ks)) or any(k not in pr for k in ks):
        return None
    ok = np.ones_like(pr['y'], dtype=bool)
    for k in ks:
        ok &= pr[k] == pr['y']
    return ok


def result_acc(js, pattern):
    hits = [(k, v['acc']) for k, v in js['results'].items() if isinstance(v, dict) and 'acc' in v and re.fullmatch(pattern, k)]
    return hits[0][1] if len(hits) == 1 else None


EE = re.escape(E)
METRICS = [('clean_acc', lambda j, pr: j['results']['clean']['acc']),
           ('clean_bal_acc', lambda j, pr: j['results']['clean']['balanced_acc']),
           ('clean_macro_f1', lambda j, pr: j['results']['clean']['macro_f1']),
           (f'fgsm_{E}', lambda j, pr: result_acc(j, rf'fgsm_{EE}')),
           (f'pgd10_{E}', lambda j, pr: result_acc(j, rf'pgd10_ce_r1_{EE}')),
           (f'pgd20_{E}', lambda j, pr: result_acc(j, rf'pgd20_ce_r1_{EE}')),
           (f'pgd50_ce_{E}', lambda j, pr: result_acc(j, rf'pgd50_ce_r\d+_{EE}')),
           (f'pgd50_cw_{E}', lambda j, pr: result_acc(j, rf'pgd50_cw_r\d+_{EE}')),
           (f'autoattack_{E}', lambda j, pr: result_acc(j, rf'autoattack_{EE}')),
           (f'worst_case_pgd_{E}', lambda j, pr: (lambda ok: None if ok is None else float(ok.mean()))(
               worst_case_correct(j, pr, False))),
           (f'worst_case_all_{E}', lambda j, pr: (lambda ok: None if ok is None else float(ok.mean()))(
               worst_case_correct(j, pr, True))),
           ('pgd20_4', lambda j, pr: result_acc(j, r'pgd20_ce_r1_4')),
           ('corrupt_mean', lambda j, pr: j['results'].get('corrupt_mean_acc')),
           ('ckpt_epoch', lambda j, pr: j['ckpt_epoch'])]


def fmt(vals, pct=True):
    v = [x for x in vals if x is not None]
    if not v:
        return '—'
    s = 100 if pct else 1
    return f'{np.mean(v)*s:.2f} ± {np.std(v, ddof=1)*s:.2f}' if len(v) > 1 else f'{v[0]*s:.2f} (n=1)'


# ──────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────
rows = []
for cond in sorted(runs):
    for ck in args.ckpts.split(','):
        seeds = [s for s in sorted(runs[cond]) if ck in runs[cond][s]['evals']]
        if not seeds:
            continue
        row = dict(condition=cond, checkpoint=ck, n_seeds=len(seeds), seeds=' '.join(map(str, seeds)),
                   pgd50_restarts=' '.join(sorted({str(runs[cond][s]['evals'][ck]['json']['args']['pgd_restarts'])
                                                   for s in seeds})))
        for name, f in METRICS:
            vals = [f(runs[cond][s]['evals'][ck]['json'], runs[cond][s]['evals'][ck]['preds']) for s in seeds]
            row[name] = fmt(vals, pct=name != 'ckpt_epoch')
            row[name + '_n'] = sum(v is not None for v in vals)
            row[name + '_per_seed'] = ' '.join(f'{s}:' + ('—' if v is None else (f'{v*100:.2f}' if name != 'ckpt_epoch'
                                                                                  else str(v))) for s, v in zip(seeds, vals))
        rows.append(row)
if rows:
    with open(out / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    cols = ['condition', 'checkpoint', 'n_seeds'] + [m for m, _ in METRICS]
    md = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    md += ['| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for r in rows]
    (out / 'summary.md').write_text(
        'Accuracy in %, mean ± SD over seeds (sample SD). worst_case_pgd = correct under FGSM and every PGD variant '
        f'at {E}/255; worst_case_all additionally under AutoAttack (primary checkpoints only).\n\n' + '\n'.join(md) + '\n',
        encoding='utf-8')


# ──────────────────────────────────────────────────────────────────────────
# PAIRED COMPARISONS
# ──────────────────────────────────────────────────────────────────────────
def mcnemar_exact(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def correct_vec(ev, metric):
    js, pr = ev['json'], ev['preds']
    if metric == 'clean':
        return pr['clean'] == pr['y']
    if metric == f'autoattack_{E}':
        return pr[metric] == pr['y'] if metric in pr else None
    if metric == f'worst_case_pgd_{E}':
        return worst_case_correct(js, pr, False)
    if metric == f'worst_case_all_{E}':
        return worst_case_correct(js, pr, True)
    if metric == 'corrupt_mean':
        ks = [k for k in pr if k.startswith('corrupt_')]
        return np.mean([pr[k] == pr['y'] for k in ks], axis=0) if ks else None


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def refs_for(cond):
    sfx = '_effb0' if cond.endswith('_effb0') else ''
    base = cond[: -len(sfx)] if sfx else cond
    refs = ['clean', 'naive_pgd', 'trades']
    if base.startswith('madry_w'): refs.append('madry_pgd')
    if base.endswith('_joint'): refs.append(base[:-len('_joint')])  # BN-mode ablation vs its separate-pass twin
    if base == 'sprat_w10r10': refs += ['cat_r10', 'madry_pgd']
    if base != 'sprat_w10r10' and (re.fullmatch(r'sprat_w\d+r\d+(_\w+)?', base) or base in
                                   ('cat_r10', 'trades_w10r10', 'madry_w10r10')):
        refs.append('sprat_w10r10')
    return [r + sfx for r in dict.fromkeys(refs) if r + sfx != cond]


paired = []
ck = args.ckpts.split(',')[0]
for cond in sorted(runs):
    if cond.startswith('clean'):
        continue
    for ref in refs_for(cond):
        if ref not in runs:
            continue
        ref_ck = 'best_clean' if ref.startswith('clean') else ck  # clean runs have no best_robust
        seeds = sorted(s for s in runs[cond] if s in runs[ref] and ck in runs[cond][s]['evals']
                       and ref_ck in runs[ref][s]['evals'])
        if not seeds:
            continue
        for metric in ('clean', f'worst_case_pgd_{E}', f'worst_case_all_{E}', f'autoattack_{E}', 'corrupt_mean'):
            A = [correct_vec(runs[cond][s]['evals'][ck], metric) for s in seeds]
            B = [correct_vec(runs[ref][s]['evals'][ref_ck], metric) for s in seeds]
            if any(v is None for v in A + B):
                continue
            A, B = np.array(A, dtype=float), np.array(B, dtype=float)  # (seeds, n_test)
            D = A - B
            per_seed = D.mean(1)
            S, n = D.shape
            bs = []
            for _ in range(args.boot):  # two-level bootstrap: seeds, then test images
                si = rng.integers(0, S, S); ii = rng.integers(0, n, n)
                bs.append(D[np.ix_(si, ii)].mean())
            t_half = T975.get(S - 1, 2.0) * per_seed.std(ddof=1) / math.sqrt(S) if S > 1 else float('nan')
            pv = [f'{s}:{mcnemar_exact(A[i] > 0.5, B[i] > 0.5):.2g}' for i, s in enumerate(seeds)] \
                if metric != 'corrupt_mean' else []
            paired.append(dict(condition=cond, reference=ref, primary=(cond, ref) in PRIMARY, checkpoint=ck,
                               ref_checkpoint=ref_ck, metric=metric, n_seeds=S, diff_pct=round(per_seed.mean() * 100, 2),
                               ci95_two_level_low=round(np.percentile(bs, 2.5) * 100, 2),
                               ci95_two_level_high=round(np.percentile(bs, 97.5) * 100, 2),
                               seed_t_ci95_halfwidth=round(t_half * 100, 2) if S > 1 else '',
                               per_seed_diff=' '.join(f'{s}:{d*100:+.2f}' for s, d in zip(seeds, per_seed)),
                               same_sign_all_seeds=bool(np.all(per_seed > 0) or np.all(per_seed < 0)),
                               mcnemar_p_per_seed=' '.join(pv)))
if paired:
    with open(out / 'paired.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys())); w.writeheader(); w.writerows(paired)


# ──────────────────────────────────────────────────────────────────────────
# PER-CROP, SHORTCUT CHECK, TRAINING COST
# ──────────────────────────────────────────────────────────────────────────
crop_rows, short_rows, cost_rows = [], [], []
for cond in sorted(runs):
    for s, r in sorted(runs[cond].items()):
        names = r['config']['class_names']
        crop_of = np.array([n.split('_')[0] for n in names])
        for ck2, e in r['evals'].items():
            res = e['json']['results']
            for k in ('clean', f'autoattack_{E}'):
                if k in res:
                    crop_rows.append(dict(condition=cond, seed=s, checkpoint=ck2, eval=k,
                                          **{f'acc_{c}': round(v * 100, 2) for c, v in res[k]['per_crop_acc'].items()}))
            pr = e['preds']
            for k in ('clean', f'autoattack_{E}', f'pgd20_ce_r1_{E}'):
                if k in pr:
                    wrong = pr[k] != pr['y']
                    xc = crop_of[pr[k]] != crop_of[pr['y']]
                    short_rows.append(dict(condition=cond, seed=s, checkpoint=ck2, eval=k,
                                           cross_crop_rate=round(float(xc.mean()) * 100, 3),
                                           cross_crop_share_of_errors=round(float(xc[wrong].mean()) * 100, 2)
                                           if wrong.any() else 0.0))
        h = r['history']
        if h:
            adv = [x['train_s'] for x in h if x.get('eps_x255', 0) > 0]
            cln = [x['train_s'] for x in h if x.get('eps_x255', 0) == 0]
            cost_rows.append(dict(condition=cond, seed=s, epochs=len(h), amp=r['config'].get('amp'),
                                  clean_epoch_s=round(np.mean(cln), 1) if cln else '',
                                  adv_epoch_s=round(np.mean(adv), 1) if adv else '',
                                  val_epoch_s=round(np.mean([x['val_s'] for x in h]), 1),
                                  total_train_h=round(sum(x['train_s'] + x['val_s'] for x in h) / 3600, 2)))
for name, rr in (('per_crop', crop_rows), ('shortcut', short_rows), ('cost', cost_rows)):
    if rr:
        keys = list(dict.fromkeys(k for r in rr for k in r))
        with open(out / f'{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rr)

# ──────────────────────────────────────────────────────────────────────────
# CURVES
# ──────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for cond in runs:
        groups['effb0' if cond.endswith('_effb0') else 'mobilevit'].append(cond)
    cmap = plt.get_cmap('tab20')
    styles = ['-', '--', ':', '-.']
    for g, conds in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ci, cond in enumerate(sorted(conds)):
            H = [runs[cond][s]['history'] for s in runs[cond] if runs[cond][s]['history']]
            if not H:
                continue
            L = min(len(h) for h in H); ep = np.arange(1, L + 1)
            col, ls = cmap(ci % 20), styles[(ci // 10) % len(styles)]
            for ax, key in zip(axes, ('val_clean_acc', 'val_pgd_acc', 'grad_norm_mean')):
                M = np.array([[h[i][key] for i in range(L)] for h in H], dtype=float)
                sc = 100 if key != 'grad_norm_mean' else 1
                ax.plot(ep, M.mean(0) * sc, label=f'{cond} (n={len(H)})', color=col, linestyle=ls)
                ax.fill_between(ep, M.min(0) * sc, M.max(0) * sc, alpha=0.12, color=col)
        for ax, t in zip(axes, ('Validation clean accuracy (%)', 'Validation PGD-10 accuracy, 8/255 (%)',
                                'Mean pre-clip gradient norm')):
            ax.set_title(t); ax.set_xlabel('epoch'); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, ncol=1)
        fig.tight_layout(); fig.savefig(out / f'curves_{g}.png', dpi=150); plt.close(fig)
except ImportError:
    warnings.append('matplotlib not available: curves skipped')

(out / 'warnings.txt').write_text('\n'.join(warnings) + '\n', encoding='utf-8')
for wmsg in warnings:
    print('WARNING:', wmsg)
print(f'written to {out}')
