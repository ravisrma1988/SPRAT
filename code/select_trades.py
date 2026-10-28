"""Pre-declared TRADES baseline selection (validation only, never the test split).

Candidates: runs/trades_tune_*_s42 (original TRADES with beta in {1, 3, 6}; beta 6 with a stop-gradient
KL target). Rule: highest validation PGD-10 accuracy at the best_robust checkpoint; ties -> higher
validation clean accuracy. Writes the chosen training arguments to code/v2/trades_choice.txt and the
full table to outputs/v2_report/trades_selection.json.
"""
import json, sys
from pathlib import Path

ROOT = Path('/workspace/ravi/SPRAT')
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'outputs/v2_runs'
ARGS = {'trades_tune_b1': '--w_max 1', 'trades_tune_b3': '--w_max 3', 'trades_tune_b6': '--w_max 6',
        'trades_tune_b6det': '--w_max 6 --trades_detach 1'}
table = []
for name, a in ARGS.items():
    f = RUNS / f'{name}_s42' / 'train_summary.json'
    if not f.exists():
        sys.exit(f'missing {f}: tuning incomplete, nothing selected')
    s = json.loads(f.read_text())['best_robust']
    table.append(dict(name=name, args=a, best_robust_epoch=s['epoch'], val_pgd10=s['val_pgd'], val_clean=s['val_clean']))
best = max(table, key=lambda r: (r['val_pgd10'], r['val_clean']))
MIN_VAL_PGD = 0.02  # guard: never adopt a collapsed TRADES (near-uniform outputs, ~0 % robust accuracy)
ok = best['val_pgd10'] >= MIN_VAL_PGD
out = ROOT / 'outputs/v2_report'; out.mkdir(parents=True, exist_ok=True)
(out / 'trades_selection.json').write_text(json.dumps(dict(rule='max val PGD-10 at best_robust; tie -> val clean; '
                                                                f'reject if < {MIN_VAL_PGD:.0%}',
                                                           candidates=table, chosen=best if ok else None), indent=2))
for r in table:
    print(f"{r['name']:20s} {r['args']:28s} ep={r['best_robust_epoch']:2d} val_pgd10={r['val_pgd10']*100:5.2f}% "
          f"val_clean={r['val_clean']*100:5.2f}%{'   <- chosen' if (r is best and ok) else ''}")
if not ok:
    sys.exit(f'no TRADES candidate reaches {MIN_VAL_PGD:.0%} val PGD-10 (best {best["val_pgd10"]:.2%}): '
             'all collapsed - nothing selected, TRADES conditions will be skipped')
(ROOT / 'code/v2/trades_choice.txt').write_text(best['args'] + '\n')
