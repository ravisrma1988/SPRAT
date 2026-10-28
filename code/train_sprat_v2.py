"""SPRAT v2 training — manifest-driven, matched-budget, robust checkpoint selection, GPU-resident.

Never reads the test split. Evaluate saved checkpoints afterwards with eval_sprat_v2.py.

Modes
  clean   cross-entropy only
  pgd     (1-w) * CE(clean) + w * CE(PGD)          w = --w_max (0.5 mixed; 1.0 = Madry PGD-AT,
                                                    adversarial examples only in the train-mode pass)
  trades  CE(clean) + w * KL(p(clean) || p(adv))    w = --w_max (beta)
Any adversarial mode can be scheduled with --warmup / --ramp (SPRAT); --ramp_what picks what ramps
(eps, the loss weight, both, or the number of attack steps for the CAT-style baseline).

Examples
  python train_sprat_v2.py --mode clean --seed 42 --out_dir ...
  python train_sprat_v2.py --mode pgd --w_max 0.5 --warmup 10 --ramp 10 --seed 42 --out_dir ...
"""
import argparse, contextlib, csv, json, math, platform, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprat_common import (atomic_save, atomic_write_text, build_model, full_strength_epoch,  # noqa: E402
                          gpu_augment, load_cache, pgd, read_manifest, schedule, seed_everything,
                          set_tf32, sha256_file, to_float, trades_inner)

# ──────────────────────────────────────────────────────────────────────────
# CLI ARGS
# ──────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['clean', 'pgd', 'trades'], required=True)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--warmup', type=int, default=0, help='clean warm-up epochs')
p.add_argument('--ramp', type=int, default=0, help='linear ramp epochs after warm-up')
p.add_argument('--ramp_what', choices=['both', 'eps', 'weight', 'steps'], default='both',
               help="'steps' = CAT-style curriculum on the number of attack steps (eps, weight fixed)")
p.add_argument('--w_max', type=float, default=None, help='lambda (pgd, default 0.5) or beta (trades, default 6)')
p.add_argument('--eps', type=float, default=8 / 255)
p.add_argument('--train_steps', type=int, default=7,
               help='PGD/TRADES inner steps at full strength; step size rule in sprat_common.schedule')
p.add_argument('--min_full_epochs', type=int, default=5, help='required epochs at full strength')
p.add_argument('--manifest', default='/workspace/ravi/SPRAT/data/manifest/manifest_sprat_v2_dedup.csv')
p.add_argument('--cache_dir', default='/workspace/ravi/SPRAT/data/cache_v2', help='built by build_cache.py')
p.add_argument('--out_dir', required=True)
p.add_argument('--backbone', default='mobilevit_s')
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--weight_decay', type=float, default=1e-4)
p.add_argument('--label_smoothing', type=float, default=0.0,
               help='training CE only; attacks use plain CE. v2 uses 0 (standard PGD-AT/TRADES; label smoothing '
                    'can mask gradients and inflate PGD robustness). v1 used 0.1.')
p.add_argument('--trades_detach', type=int, default=0,
               help='1: KL target p(clean) detached (v1 behaviour); 0: gradient through both (original TRADES)')
p.add_argument('--bn_mode', choices=['separate', 'joint'], default='separate',
               help='separate: clean and adversarial forwards in two train-mode passes (as v1); '
                    'joint: one train-mode pass over the concatenated batch')
p.add_argument('--val_attack_steps', type=int, default=10)
p.add_argument('--val_batch_size', type=int, default=256)
# speed (chosen once from bench_train_speed.py and applied to every run via train_flags.txt)
p.add_argument('--amp', choices=['off', 'bf16'], default='off', help='autocast for training and validation')
p.add_argument('--tf32', type=int, default=1)
p.add_argument('--channels_last', type=int, default=0)
p.add_argument('--compile', type=int, default=0)
p.add_argument('--deterministic', type=int, default=0, help='bitwise-deterministic kernels (slower)')
p.add_argument('--resume', action='store_true')
p.add_argument('--allow_code_change', action='store_true', help='resume even if the script hashes changed')
p.add_argument('--max_batches', type=int, default=0, help='debug only: cap batches per epoch')
p.add_argument('--stop_after_epoch', type=int, default=0, help='debug only: exit after this epoch (resume test)')
args = p.parse_args()
if args.w_max is None:
    args.w_max = {'clean': 0.0, 'pgd': 0.5, 'trades': 6.0}[args.mode]

ADV = args.mode != 'clean'
ADV_ONLY = args.mode == 'pgd' and args.w_max == 1.0  # Madry PGD-AT: no clean train-mode pass at full strength
FULL_EP = full_strength_epoch(args.warmup, args.ramp, args.eps, args.w_max, args.ramp_what, args.train_steps) if ADV else 1
if ADV and args.epochs - FULL_EP + 1 < args.min_full_epochs:
    sys.exit(f'schedule never completes properly: full strength at epoch {FULL_EP}, '
             f'only {args.epochs - FULL_EP + 1} of {args.epochs} epochs at full strength')

out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_everything(args.seed, deterministic=bool(args.deterministic))
set_tf32(bool(args.tf32))

# ──────────────────────────────────────────────────────────────────────────
# DATA (train + val only), resident on the GPU as uint8
# ──────────────────────────────────────────────────────────────────────────
rows, class_names = read_manifest(args.manifest)
MANIFEST_SHA = sha256_file(args.manifest)
train_rows = [r for r in rows if r['split'] == 'train']
val_rows = [r for r in rows if r['split'] == 'val']
del rows  # the test split is never loaded in this process
Xtr, Ytr, cache_meta = load_cache(args.cache_dir, 'train', device, MANIFEST_SHA, train_rows)
Xva, Yva, _ = load_cache(args.cache_dir, 'val', device, MANIFEST_SHA, val_rows)
CACHE_GIB = (Xtr.numel() + Xva.numel()) / 2**30

counts = torch.bincount(Ytr, minlength=len(class_names)).cpu().numpy()
sample_w = (1.0 / torch.tensor(counts, dtype=torch.double))[Ytr.cpu()]  # inverse class frequency
gen = torch.Generator().manual_seed(args.seed)                          # which images (CPU, tiny)
aug_gen = torch.Generator(device=device).manual_seed(args.seed)         # augmentation draws (GPU)

# ──────────────────────────────────────────────────────────────────────────
# MODEL / OPTIMISER
# ──────────────────────────────────────────────────────────────────────────
model = build_model(args.backbone, len(class_names)).to(device)
if args.channels_last:
    model = model.to(memory_format=torch.channels_last)
net = torch.compile(model) if args.compile else model  # forward/attacks use `net`; state_dict from `model`
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
steps_per_epoch = len(train_rows) // args.batch_size  # drop_last
if args.max_batches:
    steps_per_epoch = min(steps_per_epoch, args.max_batches)
total_steps = args.epochs * steps_per_epoch
# Plain per-step cosine decay: no warm restarts, so no LR jump coincides with a schedule boundary.
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: 1e-6 / args.lr + (1 - 1e-6 / args.lr) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
ce_train = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def cur_schedule(epoch):
    if not ADV:
        return 0.0, 0.0, 0, 0.0
    return schedule(epoch, args.warmup, args.ramp, args.eps, args.w_max, args.ramp_what, args.train_steps)


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16) if args.amp == 'bf16' else contextlib.nullcontext()


def fmt(x):
    return x.contiguous(memory_format=torch.channels_last) if args.channels_last else x


TRAIN_KEYS = ('mode', 'seed', 'epochs', 'warmup', 'ramp', 'ramp_what', 'w_max', 'eps', 'train_steps', 'backbone',
              'batch_size', 'lr', 'weight_decay', 'label_smoothing', 'trades_detach', 'bn_mode', 'val_attack_steps', 'amp', 'tf32',
              'channels_last', 'compile', 'deterministic', 'max_batches')
config = dict(vars(args), class_names=class_names, full_strength_epoch=FULL_EP, adv=ADV, adv_only_trainmode_pass=ADV_ONLY,
              train_rows=len(train_rows), val_rows=len(val_rows), class_counts_train=counts.tolist(),
              manifest_sha256=MANIFEST_SHA, cache_meta=cache_meta, cache_gib_on_gpu=round(CACHE_GIB, 3),
              script_sha256=sha256_file(__file__), common_sha256=sha256_file(Path(__file__).with_name('sprat_common.py')),
              input_norm=dict(mean=model.mean.flatten().tolist(), std=model.std.flatten().tolist()),
              augmentation='GPU: jitter b/c/s=0.3/0.3/0.2 hue 0.05 -> random 224 crop of 256, H/V flip, '
                           'rotation +-15 deg (bilinear, one affine resample) -> grayscale p=0.05',
              torch=torch.__version__, python=platform.python_version(),
              gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
              argv=sys.argv, step_size_rule='alpha_t = eps_t/4 * train_steps/steps_t',
              lr_schedule='per-step cosine to 1e-6, no restarts',
              selection='best_robust.pth = max val PGD acc among epochs >= full_strength_epoch '
                        '(tie: higher clean); best_clean.pth = max val clean acc; final.pth = last epoch',
              reproducibility='seeded data order/augmentation/init/attack starts; GPU kernels not bitwise '
                              'deterministic unless --deterministic 1')
with open(out / 'schedule.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['epoch', 'eps_x255', 'weight', 'attack_steps', 'train_alpha_x255'])
    for e in range(1, args.epochs + 1):
        eps_t, w_t, k_t, a_t = cur_schedule(e)
        w.writerow([e, round(eps_t * 255, 4), round(w_t, 4), k_t, round(a_t * 255, 4)])


# ──────────────────────────────────────────────────────────────────────────
# TRAIN / VAL
# ──────────────────────────────────────────────────────────────────────────
def train_one_epoch(epoch):
    eps_t, w_t, k_t, alpha_t = cur_schedule(epoch)
    adv_only = ADV_ONLY and w_t == 1.0
    net.train()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    # running sums stay on the GPU; one host sync per epoch (plus the occasional progress print)
    acc = torch.zeros(5, device=device)  # loss, clean_ce, adv_term, correct_clean, correct_adv (sums)
    gnorms = []
    idx = torch.multinomial(sample_w, steps_per_epoch * args.batch_size, replacement=True, generator=gen).to(device)
    for i in range(steps_per_epoch):
        b_idx = idx[i * args.batch_size:(i + 1) * args.batch_size]
        x, y = fmt(gpu_augment(Xtr[b_idx], aug_gen)), Ytr[b_idx]
        zero = torch.zeros((), device=device)
        adv_term, clean_ce, correct_c, correct_a = zero, zero, zero, zero
        with autocast():
            if eps_t > 0:
                net.eval()
                x_adv = fmt(pgd(net, x, y, eps_t, alpha_t, k_t) if args.mode == 'pgd'
                            else trades_inner(net, x, eps_t, alpha_t, k_t))
                net.train()
                if adv_only:  # standard PGD-AT: only adversarial examples see the train-mode pass
                    logits_a = net(x_adv)
                    adv_term = loss = ce_train(logits_a.float(), y)
                    correct_a = (logits_a.argmax(1) == y).sum().float()
                else:
                    if args.bn_mode == 'joint':
                        logits_c, logits_a = net(torch.cat([x, x_adv])).chunk(2)
                    else:
                        logits_c, logits_a = net(x), net(x_adv)
                    logits_c, logits_a = logits_c.float(), logits_a.float()
                    clean_ce = ce_train(logits_c, y)
                    if args.mode == 'pgd':
                        adv_term = ce_train(logits_a, y)
                        loss = (1 - w_t) * clean_ce + w_t * adv_term
                    else:  # TRADES: by default gradient flows through both distributions, as in the original
                        target = F.softmax(logits_c.detach() if args.trades_detach else logits_c, 1)
                        adv_term = F.kl_div(F.log_softmax(logits_a, 1), target, reduction='batchmean')
                        loss = clean_ce + w_t * adv_term
                    correct_c = (logits_c.argmax(1) == y).sum().float()
                    correct_a = (logits_a.argmax(1) == y).sum().float()
            else:
                logits_c = net(x).float()
                clean_ce = loss = ce_train(logits_c, y)
                correct_c = (logits_c.argmax(1) == y).sum().float()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorms.append(nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach())
        opt.step(); sched.step()
        b = x.size(0)
        acc += torch.stack([loss.detach() * b, clean_ce.detach() * b, adv_term.detach() * b, correct_c, correct_a])
        if i % 100 == 0:
            print(f'  ep{epoch} batch {i}/{steps_per_epoch} loss={loss.item():.4f}', flush=True)
    n = steps_per_epoch * args.batch_size
    s = (acc / n).tolist()
    g = torch.stack(gnorms).float().cpu().numpy()
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.0
    return dict(eps_x255=eps_t * 255, weight=w_t, attack_steps=k_t, lr_end=opt.param_groups[0]['lr'],
                train_loss=s[0], train_clean_ce=None if adv_only else s[1], train_adv_term=s[2],
                train_clean_acc_trainmode=None if adv_only else s[3],
                train_adv_acc_trainmode=s[4] if eps_t > 0 else None,
                grad_norm_mean=float(g.mean()), grad_norm_p95=float(np.percentile(g, 95)),
                grad_clip_frac=float((g > 1.0).mean()),
                peak_gpu_mem_gib=round(peak, 2))  # includes the resident data cache (config.cache_gib_on_gpu)


def validate():
    net.eval()
    c = torch.zeros(2, device=device)
    n_val = len(Yva) if not args.max_batches else min(len(Yva), args.max_batches * args.val_batch_size)
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []), autocast():
        torch.manual_seed(1234)  # identical PGD random starts at every epoch
        for i in range(0, n_val, args.val_batch_size):
            x, y = fmt(to_float(Xva[i:i + args.val_batch_size])), Yva[i:i + args.val_batch_size]
            x_adv = pgd(net, x, y, args.eps, args.eps / 4, args.val_attack_steps)
            with torch.no_grad():
                c += torch.stack([(net(x).argmax(1) == y).sum(), (net(x_adv).argmax(1) == y).sum()]).float()
    c_clean, c_rob = (c / n_val).tolist()
    return c_clean, c_rob


def save(path, epoch):
    atomic_save(dict(epoch=epoch, model_state=model.state_dict(), class_names=class_names, config=config), path)


history, start = [], 1
best = dict(clean=(-1, -1, 0), robust=(-1, -1, 0))  # (primary, tiebreak, epoch)
if args.resume and (out / 'last.pth').exists():
    # map to CPU: generator/RNG states must stay CPU ByteTensors; load_state_dict moves weights/moments
    ck = torch.load(out / 'last.pth', map_location='cpu', weights_only=False)
    old = ck['config']
    diff = {k: (old.get(k), config[k]) for k in TRAIN_KEYS + ('manifest_sha256',) if old.get(k) != config[k]}
    if diff:
        sys.exit(f'refusing to resume: training configuration differs from last.pth: {diff}')
    code_diff = [k for k in ('script_sha256', 'common_sha256') if old.get(k) != config[k]]
    if code_diff and not args.allow_code_change:
        sys.exit(f'refusing to resume: code changed since last.pth ({code_diff}); use --allow_code_change or a fresh out_dir')
    model.load_state_dict(ck['model_state']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
    history, best, start = ck['history'], ck['best'], ck['epoch'] + 1
    gen.set_state(ck['gen']); aug_gen.set_state(ck['aug_gen']); torch.set_rng_state(ck['rng_cpu'])
    if torch.cuda.is_available() and ck.get('rng_cuda') is not None:
        torch.cuda.set_rng_state_all(ck['rng_cuda'])
    config = old  # keep the original config; record the resume
    config.setdefault('resumed', []).append(dict(at_epoch=ck['epoch'], argv=sys.argv, time=time.strftime('%F %T')))
    print(f'[RESUME] from epoch {ck["epoch"]}', flush=True)
atomic_write_text(out / 'config.json', json.dumps(config, indent=2))

print(f'[RUN] mode={args.mode} seed={args.seed} warmup={args.warmup} ramp={args.ramp} ramp_what={args.ramp_what} '
      f'w_max={args.w_max} epochs={args.epochs} full_strength_epoch={FULL_EP} train={len(train_rows)} '
      f'val={len(val_rows)} steps/epoch={steps_per_epoch} amp={args.amp} tf32={args.tf32} '
      f'channels_last={args.channels_last} compile={args.compile} device={config["gpu"]}', flush=True)

for epoch in range(start, args.epochs + 1):
    t0 = time.time()
    rec = dict(epoch=epoch, **train_one_epoch(epoch))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    rec['val_clean_acc'], rec['val_pgd_acc'] = validate()
    rec['train_s'], rec['val_s'] = round(t1 - t0, 1), round(time.time() - t1, 1)
    history.append(rec)
    print(f"Epoch {epoch:03d}/{args.epochs} eps={rec['eps_x255']:.2f}/255 w={rec['weight']:.3f} K={rec['attack_steps']} "
          f"loss={rec['train_loss']:.4f} | val clean={rec['val_clean_acc']*100:.2f}% "
          f"pgd{args.val_attack_steps}={rec['val_pgd_acc']*100:.2f}% | gnorm={rec['grad_norm_mean']:.3f} "
          f"| mem={rec['peak_gpu_mem_gib']}GiB | {rec['train_s']:.0f}s+{rec['val_s']:.0f}s", flush=True)

    new_best = []
    if (rec['val_clean_acc'], rec['val_pgd_acc']) > tuple(best['clean'][:2]):
        best['clean'] = (rec['val_clean_acc'], rec['val_pgd_acc'], epoch); new_best.append('best_clean')
    if ADV and epoch >= FULL_EP and (rec['val_pgd_acc'], rec['val_clean_acc']) > tuple(best['robust'][:2]):
        best['robust'] = (rec['val_pgd_acc'], rec['val_clean_acc'], epoch); new_best.append('best_robust')
    # last.pth (which records `best`) first, then the best_* files, so a kill never leaves them inconsistent
    atomic_save(dict(epoch=epoch, model_state=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                     history=history, best=best, gen=gen.get_state(), aug_gen=aug_gen.get_state(),
                     rng_cpu=torch.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     class_names=class_names, config=config), out / 'last.pth')
    for nb in new_best:
        save(out / f'{nb}.pth', epoch)
    atomic_write_text(out / 'history.json', json.dumps(history, indent=1))
    if args.stop_after_epoch and epoch >= args.stop_after_epoch and epoch < args.epochs:
        print(f'[STOP] debug stop after epoch {epoch}', flush=True)
        sys.exit(0)

with open(out / 'history.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
summary = dict(best_clean=dict(epoch=best['clean'][2], val_clean=best['clean'][0], val_pgd=best['clean'][1]),
               best_robust=(dict(epoch=best['robust'][2], val_pgd=best['robust'][0], val_clean=best['robust'][1])
                            if ADV else None),
               final=dict(epoch=args.epochs, val_clean=history[-1]['val_clean_acc'], val_pgd=history[-1]['val_pgd_acc']),
               primary_checkpoint='best_robust.pth' if ADV else 'best_clean.pth',
               total_train_s=sum(h['train_s'] for h in history), total_val_s=sum(h['val_s'] for h in history))
atomic_write_text(out / 'train_summary.json', json.dumps(summary, indent=2))
save(out / 'final.pth', args.epochs)  # written last: its presence marks a complete run
print('[DONE]', json.dumps(summary), flush=True)
