"""SPRAT v2 evaluation — the only script that touches the test split.

For one checkpoint: clean, FGSM, PGD (several step counts, CE and CW-margin losses, restarts on
survivors), common corruptions (5 types x 3 severities), standard AutoAttack, optional
BatchNorm-recalibration diagnostic. Reports accuracy, balanced accuracy, macro-F1, per-class
precision/recall/F1, per-crop accuracy, confusion matrices and bootstrap 95% CIs, and saves
per-sample predictions for paired comparisons between methods.

Gradient attacks are run only on images the model classifies correctly when clean (AutoAttack's
convention); misclassified images keep their clean prediction. All evaluation is fp32 (TF32 allowed).

Usage
  python eval_sprat_v2.py --ckpt runs/x/best_robust.pth --out runs/x/eval_best_robust
  python eval_sprat_v2.py --ckpt runs/x/final.pth --out runs/x/eval_final --attacks clean,fgsm,pgd
"""
import os
os.environ.setdefault('TIMM_FUSED_ATTN', '0')  # math attention: reproducible attention backward in attacks
import argparse, copy, glob, hashlib, inspect, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprat_common import (CORRUPTIONS, EVAL_SIZE, atomic_write_text, build_model, corrupt, fgsm,  # noqa: E402
                          load_cache, pgd, read_manifest, seed_everything, set_tf32, sha256_file, to_float)

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--out', required=True)
p.add_argument('--manifest', default='/workspace/ravi/SPRAT/data/manifest/manifest_sprat_v2_dedup.csv')
p.add_argument('--cache_dir', default='/workspace/ravi/SPRAT/data/cache_v2')
p.add_argument('--aa_dir', default='/workspace/ravi/SPRAT/vendor/auto-attack', help='vendored AutoAttack repo')
p.add_argument('--split', default='test', choices=['test', 'val'])
p.add_argument('--attacks', default='clean,fgsm,pgd,corrupt,aa')
p.add_argument('--fgsm_eps', default='2,4,8,16', help='x/255')
p.add_argument('--pgd_eps', default='4,8', help='x/255')
p.add_argument('--pgd_steps', default='10,20,50')
p.add_argument('--pgd_restarts', type=int, default=3, help='restarts for the largest step count')
p.add_argument('--aa_eps', type=float, default=8)
p.add_argument('--aa_n', type=int, default=0, help='0 = whole split')
p.add_argument('--bn_recal', type=int, default=0, help='diagnostic: recompute BN stats on N clean train batches')
p.add_argument('--attack_misclassified', action='store_true', help='also attack clean-misclassified images')
p.add_argument('--batch_size', type=int, default=256)
p.add_argument('--boot', type=int, default=1000)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--tf32', type=int, default=1)
p.add_argument('--max_n', type=int, default=0, help='debug only: evaluate the first N images')
args = p.parse_args()

seed_everything(args.seed)
set_tf32(bool(args.tf32))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

rows, class_names = read_manifest(args.manifest)
MANIFEST_SHA = sha256_file(args.manifest)
ev_rows = [r for r in rows if r['split'] == args.split]
ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
assert ck['class_names'] == class_names, 'checkpoint class order differs from manifest'
assert ck['config']['manifest_sha256'] == MANIFEST_SHA, 'checkpoint was trained on a different manifest'
model = build_model(ck['config']['backbone'], len(class_names), pretrained=False)
model.load_state_dict(ck['model_state']); model.to(device).eval()

Xu8, Yd, _ = load_cache(args.cache_dir, args.split, device, MANIFEST_SHA, ev_rows)
if args.max_n:
    Xu8, Yd, ev_rows = Xu8[:args.max_n], Yd[:args.max_n], ev_rows[:args.max_n]
X, Y = to_float(Xu8), Yd  # whole split on the GPU (~0.9 GB float for 1,441 images)
del Xu8
crop = np.array([r['dataset'] for r in ev_rows])
y_np = Y.cpu().numpy()
C = len(class_names)


# ──────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────
def confusion(y, pr):
    return np.bincount(y * C + pr, minlength=C * C).reshape(C, C)  # rows = true, cols = predicted


def scores(y, pr):
    cm = confusion(y, pr)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(C), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(C), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(C), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    return dict(acc=float(tp.sum() / len(y)), balanced_acc=float(rec[present].mean()),
                macro_f1=float(f1[present].mean())), prec, rec, f1, cm


def report(pr, name):
    s, prec, rec, f1, cm = scores(y_np, pr)
    rng = np.random.default_rng(args.seed)
    boots = {k: [] for k in s}
    for _ in range(args.boot):
        idx = rng.integers(0, len(y_np), len(y_np))
        b = scores(y_np[idx], pr[idx])[0]
        for k in s:
            boots[k].append(b[k])
    s['ci95'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    s['per_crop_acc'] = {c: float((pr[crop == c] == y_np[crop == c]).mean()) for c in np.unique(crop)}
    s['per_class'] = [dict(cls=class_names[k], support=int(cm[k].sum()), precision=float(prec[k]),
                           recall=float(rec[k]), f1=float(f1[k])) for k in range(C)]
    s['confusion'] = cm.tolist()
    print(f'  {name:28s} acc={s["acc"]*100:6.2f}%  bal={s["balanced_acc"]*100:6.2f}%  '
          f'mF1={s["macro_f1"]*100:6.2f}%  CI=[{s["ci95"]["acc"][0]*100:.1f},{s["ci95"]["acc"][1]*100:.1f}]', flush=True)
    return s


def predict(m, Xin):
    with torch.no_grad():
        return torch.cat([m(Xin[i:i + args.batch_size]).argmax(1)
                          for i in range(0, len(Xin), args.batch_size)]).cpu().numpy()


def attack_all(fn, seed, m=None):
    """Adversarial predictions for the whole split. Only clean-correct images are attacked unless
    --attack_misclassified; the others keep their clean prediction."""
    m = m or model
    torch.manual_seed(seed)
    pred_clean = torch.from_numpy(predict(m, X)).to(device)
    todo = torch.arange(len(X), device=device) if args.attack_misclassified else torch.nonzero(pred_clean == Y).flatten()
    pr = pred_clean.clone()
    for i in range(0, len(todo), args.batch_size):
        j = todo[i:i + args.batch_size]
        xa = fn(X[j], Y[j])
        with torch.no_grad():
            pr[j] = m(xa).argmax(1)
    return pr.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────
res = dict(ckpt=str(args.ckpt), ckpt_sha256=sha256_file(args.ckpt), ckpt_epoch=ck['epoch'],
           train_config={k: ck['config'].get(k) for k in ('mode', 'seed', 'epochs', 'warmup', 'ramp', 'ramp_what',
                                                         'w_max', 'bn_mode', 'backbone', 'amp', 'manifest_sha256',
                                                         'script_sha256')},
           split=args.split, n=int(len(Y)), manifest_sha256=MANIFEST_SHA, eval_script_sha256=sha256_file(__file__),
           common_sha256=sha256_file(Path(__file__).with_name('sprat_common.py')),
           args=vars(args), precision=f'fp32, tf32={args.tf32}', timm_fused_attn=os.environ.get('TIMM_FUSED_ATTN'),
           attacked_points='clean-correct only' if not args.attack_misclassified else 'all',
           results={}, timings_s={}, attack_keys={})
preds = dict(y=y_np)
attacks = args.attacks.split(',')
print(f'[EVAL] {args.ckpt} epoch={ck["epoch"]} split={args.split} n={len(Y)}', flush=True)

t = time.time(); preds['clean'] = predict(model, X)
res['results']['clean'] = report(preds['clean'], 'clean'); res['timings_s']['clean'] = time.time() - t

if 'fgsm' in attacks:
    for e in map(float, args.fgsm_eps.split(',')):
        k = f'fgsm_{e:g}'; t = time.time()
        preds[k] = attack_all(lambda x, y: fgsm(model, x, y, e / 255), 0)
        res['results'][k] = report(preds[k], k); res['timings_s'][k] = time.time() - t
        res['attack_keys'].setdefault(f'{e:g}', []).append(k)

if 'pgd' in attacks:
    steps = sorted(map(int, args.pgd_steps.split(',')))
    for e in map(float, args.pgd_eps.split(',')):
        for K in steps:
            for loss in ('ce', 'cw'):
                if loss == 'cw' and K != steps[-1]:
                    continue  # margin loss only at the strongest step count
                R = args.pgd_restarts if K == steps[-1] else 1
                k = f'pgd{K}_{loss}_r{R}_{e:g}'; t = time.time()
                preds[k] = attack_all(  # step size 2.5*eps/K (Madry et al.)
                    lambda x, y: pgd(model, x, y, e / 255, 2.5 * (e / 255) / K, K, restarts=R, loss=loss), 1)
                res['results'][k] = report(preds[k], k); res['timings_s'][k] = time.time() - t
                res['attack_keys'].setdefault(f'{e:g}', []).append(k)

if 'corrupt' in attacks:
    accs = []
    for kind in CORRUPTIONS:
        for sev in (1, 2, 3):
            k = f'corrupt_{kind}_{sev}'; t = time.time()
            xc = torch.cat([corrupt(X[i:i + args.batch_size], kind, sev, seed=i)
                            for i in range(0, len(X), args.batch_size)])
            preds[k] = predict(model, xc)
            res['results'][k] = report(preds[k], k); res['timings_s'][k] = time.time() - t
            accs.append(res['results'][k]['acc'])
    res['results']['corrupt_mean_acc'] = float(np.mean(accs))
    print(f'  mean corruption accuracy: {np.mean(accs)*100:.2f}%', flush=True)

if 'aa' in attacks:
    aa_dir = Path(args.aa_dir)
    if aa_dir.exists():
        sys.path.insert(0, str(aa_dir))
    import autoattack
    from autoattack import AutoAttack
    r = subprocess.run(['git', '-C', str(aa_dir), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    commit = r.stdout.strip() if r.returncode == 0 and len(r.stdout.strip()) == 40 else 'unknown'
    src_hash = hashlib.sha256(b''.join(Path(f).read_bytes() for f in
                                      sorted(glob.glob(str(Path(autoattack.__file__).parent / '*.py'))))).hexdigest()
    res['autoattack'] = dict(commit=commit, module_file=autoattack.__file__, source_sha256=src_hash, version='standard')
    n = args.aa_n or len(X)
    t = time.time()
    adversary = AutoAttack(model, norm='Linf', eps=args.aa_eps / 255, version='standard', device=str(device),
                           seed=args.seed, log_path=str(out / 'autoattack.log'))
    if 'return_labels' in inspect.signature(adversary.run_standard_evaluation).parameters:
        x_adv, y_aa = adversary.run_standard_evaluation(X[:n], Y[:n], bs=args.batch_size, return_labels=True)
        pr = y_aa.cpu().numpy()  # AutoAttack's own per-sample verdicts
        res['autoattack']['label_source'] = 'run_standard_evaluation(return_labels=True)'
        res['autoattack']['disagreements_with_reforward'] = int((predict(model, x_adv) != pr).sum())
    else:
        x_adv = adversary.run_standard_evaluation(X[:n], Y[:n], bs=args.batch_size)
        pr = predict(model, x_adv)
        res['autoattack']['label_source'] = 'forward pass on returned x_adv'
    k = f'autoattack_{args.aa_eps:g}'
    if n == len(X):
        preds[k] = pr
        res['results'][k] = report(pr, k)
        res['attack_keys'].setdefault(f'{args.aa_eps:g}', []).append(k)
    else:  # debug subset: accuracy on the attacked subset only, never mixed into full-split metrics
        res['results'][k + '_subset'] = dict(n=n, acc=float((pr == y_np[:n]).mean()))
        print(f'  {k} on first {n}: {res["results"][k + "_subset"]["acc"]*100:.2f}%', flush=True)
    res['timings_s'][k] = time.time() - t

if args.bn_recal:
    # Diagnostic for the Naive-PGD collapse hypothesis: if clean accuracy recovers after
    # re-estimating BN statistics on clean training images, the BN running stats were polluted.
    m2 = copy.deepcopy(model)
    for mod in m2.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats(); mod.momentum = None  # cumulative average over the batches below
    tr_rows = [r for r in rows if r['split'] == 'train']
    Xtr, _, _ = load_cache(args.cache_dir, 'train', device, MANIFEST_SHA, tr_rows)
    order = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(0)).to(device)
    nb = min(args.bn_recal, len(Xtr) // args.batch_size)
    m2.train()
    with torch.no_grad():
        for i in range(nb):
            xb = to_float(Xtr[order[i * args.batch_size:(i + 1) * args.batch_size]])
            m2(F.interpolate(xb, size=EVAL_SIZE, mode='bilinear', antialias=True, align_corners=False))
    del Xtr
    m2.eval()
    res['bn_recal'] = dict(batches=nb, images=nb * args.batch_size, requested_batches=args.bn_recal)
    preds['bnrecal_clean'] = predict(m2, X)
    res['results']['bnrecal_clean'] = report(preds['bnrecal_clean'], f'bn_recal({nb}x{args.batch_size}) clean')
    preds['bnrecal_pgd20_8'] = attack_all(lambda x, y: pgd(m2, x, y, 8 / 255, 2.5 * (8 / 255) / 20, 20), 1, m=m2)
    res['results']['bnrecal_pgd20_8'] = report(preds['bnrecal_pgd20_8'], 'bn_recal pgd20 8/255')

# per-sample predictions first, eval.json last (its presence marks a complete evaluation)
tmp = out / 'per_sample_preds.tmp.npz'
np.savez_compressed(tmp, paths=np.array([r['path'] for r in ev_rows]), **preds)
os.replace(tmp, out / 'per_sample_preds.npz')
atomic_write_text(out / 'eval.json', json.dumps(res, indent=1))
print(f'[DONE] -> {out}', flush=True)
