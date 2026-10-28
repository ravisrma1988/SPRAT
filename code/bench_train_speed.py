"""Training-speed and attack-parity benchmark (Phase 0, a few GPU-minutes).

1. Speed: images/s of one full adversarial training step (GPU augmentation, PGD-7 in eval mode,
   clean + adversarial forward in train mode, backward, AdamW) at batch 32 for each backbone under
   fp32 / TF32 / bf16 autocast / + channels_last / + torch.compile.
2. Parity: does a bf16-autocast PGD-7 attack hurt the model as much as an fp32 one? Adversarial
   examples are generated both ways on the same batches and both are scored in strict fp32
   (cross-entropy and accuracy). bf16 is acceptable for training only if its fp32-scored loss is
   not materially lower than the fp32 attack's.

Usage: python bench_train_speed.py --out outputs/v2_report/train_speed.json
"""
import argparse, contextlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprat_common import build_model, gpu_augment, load_cache, pgd, read_manifest, sha256_file  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--manifest', default='/workspace/ravi/SPRAT/data/manifest/manifest_sprat_v2_dedup.csv')
p.add_argument('--cache_dir', default='/workspace/ravi/SPRAT/data/cache_v2')
p.add_argument('--backbones', default='mobilevit_s,efficientnet_b0')
p.add_argument('--steps', type=int, default=30, help='timed steps per configuration (after 5 warm-up steps)')
p.add_argument('--parity_batches', type=int, default=8)
p.add_argument('--out', required=True)
args = p.parse_args()
dev = torch.device('cuda')
eps, K = 8 / 255, 7

rows, class_names = read_manifest(args.manifest)
Xtr, Ytr, _ = load_cache(args.cache_dir, 'train', dev, sha256_file(args.manifest),
                         [r for r in rows if r['split'] == 'train'])
gen = torch.Generator(device=dev).manual_seed(0)
bidx = [torch.randint(0, len(Ytr), (32,), device=dev, generator=gen) for _ in range(args.steps + 5)]


def set_precision(tf32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def step_fn(model, opt, amp, cl):
    ac = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if amp else contextlib.nullcontext

    def step(i):
        x, y = gpu_augment(Xtr[bidx[i]], gen), Ytr[bidx[i]]
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        with ac():
            model.eval(); xa = pgd(model, x, y, eps, eps / 4, K); model.train()
            loss = 0.5 * F.cross_entropy(model(x), y) + 0.5 * F.cross_entropy(model(xa), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return step


res = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), batch=32, speed={}, parity={})
CONFIGS = [('fp32', False, False, False, False), ('tf32', True, False, False, False),
           ('bf16', True, True, False, False), ('bf16+channels_last', True, True, True, False),
           ('bf16+compile', True, True, False, True)]
for bb in args.backbones.split(','):
    res['speed'][bb] = {}
    for name, tf32, amp, cl, comp in CONFIGS:
        set_precision(tf32)
        torch.manual_seed(0)
        m = build_model(bb, len(class_names)).to(dev)
        if cl:
            m = m.to(memory_format=torch.channels_last)
        mm = torch.compile(m) if comp else m
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        step = step_fn(mm, opt, amp, cl)
        try:
            t_c = time.time()
            for i in range(5):
                step(i)
            torch.cuda.synchronize(); warm = time.time() - t_c
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for i in range(5, 5 + args.steps):
                step(i)
            torch.cuda.synchronize(); dt = time.time() - t0
            r = dict(img_per_s=round(32 * args.steps / dt, 1), s_per_step=round(dt / args.steps, 4),
                     warmup_s=round(warm, 1), peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        except Exception as e:  # e.g. compile unsupported for a graph
            r = dict(error=repr(e)[:300])
        res['speed'][bb][name] = r
        print(bb, name, r, flush=True)
        del m, mm, opt
        torch.cuda.empty_cache()

    # parity: bf16 vs fp32 attack generation, both scored in strict fp32
    set_precision(False)
    torch.manual_seed(0)
    m = build_model(bb, len(class_names)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    for i in range(20):  # a few clean steps so the head is not random
        x, y = gpu_augment(Xtr[bidx[i % len(bidx)]], gen), Ytr[bidx[i % len(bidx)]]
        m.train(); loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    out = {'fp32': [], 'bf16': []}
    for j in range(args.parity_batches):
        x, y = gpu_augment(Xtr[bidx[j]], gen), Ytr[bidx[j]]
        for name in ('fp32', 'bf16'):
            torch.manual_seed(100 + j)  # same random start for both
            with (torch.autocast('cuda', dtype=torch.bfloat16) if name == 'bf16' else contextlib.nullcontext()):
                xa = pgd(m, x, y, eps, eps / 4, K)
            with torch.no_grad():
                lo = m(xa.float())
            out[name].append((F.cross_entropy(lo, y).item(), (lo.argmax(1) == y).float().mean().item()))
    res['parity'][bb] = {k: dict(fp32_scored_ce=float(np.mean([a for a, _ in v])),
                                 fp32_scored_acc=float(np.mean([b for _, b in v]))) for k, v in out.items()}
    print(bb, 'parity', res['parity'][bb], flush=True)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=2))
print('[DONE]', args.out)
