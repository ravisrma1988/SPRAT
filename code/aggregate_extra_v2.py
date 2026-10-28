"""Second-pass aggregation of finished SPRAT v2 evaluations (reads existing outputs only: no training, no attacks).

Reads the run directories that aggregate_v2.py reads and writes, to --out:
  per_class.csv / per_class_summary.csv   per-class precision / recall / F1 of clean and AutoAttack predictions
                                          (summary: primary checkpoint, mean ± SD over seeds)
  confusion_summed.json                   confusion matrices summed over seeds (primary checkpoint, every condition)
  confusion_<cond>_<eval>.pdf/.png        row-normalised matrices for the conditions in CONF_SHOW
  bn_recal.csv / bn_recal_summary.csv     BN-recalibration diagnostic: clean and PGD-20 (8/255) before / after
  aa_breakdown.csv / aa_breakdown_summary.csv  accuracy after each AutoAttack component (parsed from
                                          autoattack.log) next to PGD-50 CE / CW and worst-case PGD
  history_long.csv, grad_clip.csv         per-epoch training logs in long form; share of clipped steps
  loss_curves.pdf/.png                    clean CE and adversarial term per epoch (mean, min–max over seeds)
Paired statistics at the final checkpoint come from aggregate_v2.py itself:
  aggregate_v2.py --ckpts final,best_robust,best_clean --out .../v2_report_final

Usage: python aggregate_extra_v2.py --runs /workspace/ravi/SPRAT/outputs/v2_runs --out .../v2_report/extra
"""
import argparse, csv, json, math, re
from collections import defaultdict
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--runs', required=True)
p.add_argument('--out', required=True)
p.add_argument('--eps', type=float, default=8.0)
args = p.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
E = f'{args.eps:g}'
AA = f'autoattack_{E}'
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
STAGES = ['APGD-CE', 'APGD-T', 'FAB-T', 'SQUARE']
CONF_SHOW = [('clean', 'clean'), ('sprat_w10r10', 'clean'), ('sprat_w10r10', AA), ('madry_pgd', AA)]
LOSS_SHOW = ['naive_pgd', 'cat_r10', 'madry_pgd', 'sprat_w10r10', 'sprat_w10r10_epsonly', 'sprat_w10r10_wonly',
             'sprat_w5r5']
COLOURS = ['black', 'tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
HKEYS = ['epoch', 'eps_x255', 'weight', 'attack_steps', 'train_loss', 'train_clean_ce', 'train_adv_term',
         'grad_norm_mean', 'grad_clip_frac', 'val_clean_acc', 'val_pgd_acc']


def primary_ck(cond):
    return 'best_clean' if cond.startswith('clean') else 'best_robust'


# ──────────────────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────────────────
runs = defaultdict(dict)  # cond -> seed -> dict
for d in sorted(Path(args.runs).iterdir()):
    m = re.fullmatch(r'(.+)_s(\d+)', d.name)
    if not (m and (d / 'config.json').exists()):
        continue
    r = dict(dir=d, config=json.loads((d / 'config.json').read_text()), evals={})
    r['history'] = json.loads((d / 'history.json').read_text()) if (d / 'history.json').exists() else []
    for f in sorted(d.glob('eval_*/eval.json')):
        js = json.loads(f.read_text())
        if js['split'] == 'test' and not js['args'].get('max_n'):  # full test-split evaluations only
            r['evals'][f.parent.name[len('eval_'):]] = js
    runs[m.group(1)][int(m.group(2))] = r
print(f'loaded {sum(len(v) for v in runs.values())} runs in {len(runs)} conditions')


def msd(v):
    v = [x for x in v if x is not None]
    if not v:
        return '—'
    return f'{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}' if len(v) > 1 else f'{v[0]:.2f} (n=1)'


def t_half(v):
    return T975.get(len(v) - 1, 2.0) * np.std(v, ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float('nan')


def write_csv(name, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def acc_of(js, pattern):
    hits = [v['acc'] for k, v in js['results'].items() if isinstance(v, dict) and 'acc' in v and re.fullmatch(pattern, k)]
    return round(hits[0] * 100, 2) if len(hits) == 1 else None


# ──────────────────────────────────────────────────────────────────────────
# PER-CLASS METRICS AND CONFUSION MATRICES
# ──────────────────────────────────────────────────────────────────────────
pc_rows, pc_sum, conf = [], [], {}
for cond in sorted(runs):
    for s, r in sorted(runs[cond].items()):
        for ck, js in r['evals'].items():
            for ev in ('clean', AA):
                for c in js['results'].get(ev, {}).get('per_class', []):
                    pc_rows.append(dict(condition=cond, seed=s, checkpoint=ck, eval=ev, cls=c['cls'], support=c['support'],
                                        **{k: round(c[k] * 100, 2) for k in ('precision', 'recall', 'f1')}))
    ck = primary_ck(cond)
    for ev in ('clean', AA):
        sj = [(s, r['evals'][ck]['results'][ev]) for s, r in sorted(runs[cond].items())
              if ck in r['evals'] and ev in r['evals'][ck]['results']]
        if not sj:
            continue
        names = [c['cls'] for c in sj[0][1]['per_class']]
        for i, cls in enumerate(names):
            rec = [res['per_class'][i]['recall'] * 100 for _, res in sj]
            f1 = [res['per_class'][i]['f1'] * 100 for _, res in sj]
            pc_sum.append(dict(condition=cond, checkpoint=ck, eval=ev, cls=cls, support=sj[0][1]['per_class'][i]['support'],
                               n_seeds=len(sj), recall=msd(rec), f1=msd(f1),
                               recall_per_seed=' '.join(f'{s}:{v:.2f}' for (s, _), v in zip(sj, rec))))
        conf.setdefault(cond, {})[ev] = dict(checkpoint=ck, seeds=[s for s, _ in sj], classes=names,
                                             matrix=np.sum([res['confusion'] for _, res in sj], axis=0).tolist())
write_csv('per_class.csv', pc_rows)
write_csv('per_class_summary.csv', pc_sum)
(out / 'confusion_summed.json').write_text(json.dumps(conf, indent=1), encoding='utf-8')


# ──────────────────────────────────────────────────────────────────────────
# BN RECALIBRATION DIAGNOSTIC
# ──────────────────────────────────────────────────────────────────────────
bn_rows = []
for cond in sorted(runs):
    for s, r in sorted(runs[cond].items()):
        js = r['evals'].get(primary_ck(cond))
        if not js or 'bn_recal' not in js:
            continue
        R = js['results']
        bn_rows.append(dict(condition=cond, seed=s, ckpt_epoch=js['ckpt_epoch'], batches=js['bn_recal']['batches'],
                            images=js['bn_recal']['images'], clean=round(R['clean']['acc'] * 100, 2),
                            bnrecal_clean=round(R['bnrecal_clean']['acc'] * 100, 2),
                            pgd20_8=acc_of(js, rf'pgd20_ce_r1_{re.escape(E)}'),
                            bnrecal_pgd20_8=round(R['bnrecal_pgd20_8']['acc'] * 100, 2)))
bn_sum = []
for cond in dict.fromkeys(r['condition'] for r in bn_rows):
    rr = [r for r in bn_rows if r['condition'] == cond]
    row = dict(condition=cond, n_seeds=len(rr))
    for k in ('clean', 'pgd20_8'):
        d = [r['bnrecal_' + k] - r[k] for r in rr]
        row.update({k: msd([r[k] for r in rr]), 'bnrecal_' + k: msd([r['bnrecal_' + k] for r in rr]),
                    f'delta_{k}': round(float(np.mean(d)), 2), f'delta_{k}_seed_t_halfwidth': round(t_half(d), 2),
                    f'delta_{k}_per_seed': ' '.join(f'{r["seed"]}:{x:+.2f}' for r, x in zip(rr, d))})
    bn_sum.append(row)
write_csv('bn_recal.csv', bn_rows)
write_csv('bn_recal_summary.csv', bn_sum)


# ──────────────────────────────────────────────────────────────────────────
# AUTOATTACK COMPONENT BREAKDOWN
# ──────────────────────────────────────────────────────────────────────────
aa_rows = []
for cond in sorted(runs):
    for s, r in sorted(runs[cond].items()):
        ck = primary_ck(cond)
        js, log = r['evals'].get(ck), r['dir'] / f'eval_{ck}' / 'autoattack.log'
        if not js or not log.exists():
            continue
        txt = log.read_text()
        init = re.search(r'initial accuracy: ([\d.]+)%', txt)
        after = dict(re.findall(r'robust accuracy after (\S+): ([\d.]+)%', txt))
        final = re.search(r'^robust accuracy: ([\d.]+)%', txt, re.M)
        row = dict(condition=cond, seed=s, checkpoint=ck, clean=round(js['results']['clean']['acc'] * 100, 2),
                   aa_initial=float(init.group(1)) if init else None)
        prev = row['aa_initial']
        for st in STAGES:  # AutoAttack skips later components once no correctly classified point is left
            prev = float(after[st]) if st in after else prev
            row['after_' + st.lower().replace('-', '_')] = prev
        row['components_run'] = ' '.join(st for st in STAGES if st in after)
        row['aa_final_log'] = float(final.group(1)) if final else None
        row['aa_eval_json'] = round(js['results'][AA]['acc'] * 100, 2) if AA in js['results'] else None
        row['pgd50_ce'] = acc_of(js, rf'pgd50_ce_r\d+_{re.escape(E)}')
        row['pgd50_cw'] = acc_of(js, rf'pgd50_cw_r\d+_{re.escape(E)}')
        pr = np.load(r['dir'] / f'eval_{ck}' / 'per_sample_preds.npz', allow_pickle=False)
        ks = [k for k in js.get('attack_keys', {}).get(E, []) if not k.startswith('autoattack')]
        row['worst_case_pgd'] = round(float(np.all([pr[k] == pr['y'] for k in ks], axis=0).mean()) * 100, 2) if ks else None
        aa_rows.append(row)
aa_sum = []
for cond in dict.fromkeys(r['condition'] for r in aa_rows):
    rr = [r for r in aa_rows if r['condition'] == cond]
    col = lambda k: [r[k] for r in rr]
    drops = dict(drop_apgd_t=[r['after_apgd_ce'] - r['after_apgd_t'] for r in rr],
                 drop_fab_t=[r['after_apgd_t'] - r['after_fab_t'] for r in rr],
                 drop_square=[r['after_fab_t'] - r['after_square'] for r in rr],
                 gap_worst_case_pgd_to_aa=[r['worst_case_pgd'] - r['aa_final_log'] for r in rr])
    aa_sum.append(dict(condition=cond, n_seeds=len(rr), clean=msd(col('clean')), pgd50_ce=msd(col('pgd50_ce')),
                       pgd50_cw=msd(col('pgd50_cw')), worst_case_pgd=msd(col('worst_case_pgd')),
                       after_apgd_ce=msd(col('after_apgd_ce')), after_apgd_t=msd(col('after_apgd_t')),
                       after_fab_t=msd(col('after_fab_t')), after_square=msd(col('after_square')),
                       **{k: msd(v) for k, v in drops.items()},
                       runs_reduced_by_fab_or_square=sum(a > 0 or b > 0 for a, b in zip(drops['drop_fab_t'],
                                                                                         drops['drop_square'])),
                       log_vs_eval_json_mismatches=sum(r['aa_final_log'] != r['aa_eval_json'] for r in rr)))
write_csv('aa_breakdown.csv', aa_rows)
write_csv('aa_breakdown_summary.csv', aa_sum)


# ──────────────────────────────────────────────────────────────────────────
# TRAINING LOGS, GRADIENT CLIPPING, LOSS CURVES
# ──────────────────────────────────────────────────────────────────────────
hist_rows, clip_rows = [], []
for cond in sorted(runs):
    for s, r in sorted(runs[cond].items()):
        H = r['history']
        hist_rows += [dict(condition=cond, seed=s, **{k: h.get(k) for k in HKEYS}) for h in H]
        if H:
            adv = [h['grad_clip_frac'] for h in H if h.get('eps_x255', 0) > 0]
            clip_rows.append(dict(condition=cond, seed=s, clip_frac_all_pct=round(np.mean([h['grad_clip_frac'] for h in H]) * 100, 1),
                                  clip_frac_adv_epochs_pct=round(np.mean(adv) * 100, 1) if adv else ''))
write_csv('history_long.csv', hist_rows)
write_csv('grad_clip.csv', clip_rows)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.linewidth': 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for cond, col in zip(LOSS_SHOW, COLOURS):
        H = [r['history'] for r in runs.get(cond, {}).values() if r['history']]
        if not H:
            continue
        L = min(len(h) for h in H); ep = np.arange(1, L + 1)
        for ax, key in zip(axes, ('train_clean_ce', 'train_adv_term')):
            M = np.array([[h[i][key] if h[i][key] is not None else np.nan for i in range(L)] for h in H], dtype=float)
            if np.all(np.isnan(M)):
                continue
            ax.plot(ep, np.nanmean(M, 0), color=col, lw=1.2, label=cond)
            ax.fill_between(ep, np.nanmin(M, 0), np.nanmax(M, 0), color=col, alpha=0.12, lw=0)
    for ax, t in zip(axes, ('Clean cross-entropy (training)', 'Adversarial loss term (training)')):
        ax.set_xlabel('Epoch'); ax.set_ylabel(t); ax.grid(alpha=0.3, lw=0.5)
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'loss_curves.{ext}', dpi=300)
    plt.close(fig)

    for cond, ev in CONF_SHOW:
        c = conf.get(cond, {}).get(ev)
        if not c:
            continue
        M = np.array(c['matrix'], dtype=float)
        M = 100 * M / np.maximum(M.sum(1, keepdims=True), 1)  # rows = true class, in % of its test images
        labels = [n.split('_', 1)[-1] + (' (R)' if n.startswith('rice') else '') for n in c['classes']]
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        im = ax.imshow(M, cmap='Blues', vmin=0, vmax=100)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label='% of true-class test images')
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'confusion_{cond}_{ev}.{ext}', dpi=300)
        plt.close(fig)
except ImportError:
    print('matplotlib not available: figures skipped')
print(f'written to {out}')
