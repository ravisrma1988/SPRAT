"""Shared pieces for SPRAT v2: cached GPU data, GPU augmentation, model wrapper, attacks, schedule.

Design changes from SPRAT_ABLATION_TRAIN_FINAL.py (see RUN_PLAN.md for the reasons):
  * data come only from a frozen manifest, decoded and resized once into a uint8 cache
    (build_cache.py); training/evaluation keep the whole split on the GPU and augment there,
    so the loop is GPU-bound with no per-sample CPU work;
  * models take raw RGB in [0,1]; each backbone's own pretrained normalisation (from timm's
    pretrained config) lives inside the model. MobileViT-S (cvnets weights) expects mean 0 /
    std 1 - v1 wrongly applied ImageNet mean/std to it;
  * attacks always use plain cross-entropy (never the label-smoothed training loss).
"""
import csv, hashlib, json, math, os, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────
def seed_everything(seed, deterministic=False):
    """Seeds fix data order, augmentation draws, initialisation and attack starts. GPU kernels
    (cuDNN autotuning, atomics in some backward ops, fused attention) still make runs statistically
    rather than bitwise reproducible unless deterministic=True (slower)."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def set_tf32(enabled):
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def atomic_save(obj, path):
    """torch.save to a temporary file then rename, so a kill never leaves a truncated checkpoint."""
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp); os.replace(tmp, path)


def atomic_write_text(path, text):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8'); os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────
# DATA: manifest + uint8 cache (built once by build_cache.py)
# ──────────────────────────────────────────────────────────────────────────
TRAIN_CACHE_SIZE, EVAL_SIZE = 256, 224  # train: random 224 crop of 256 (as v1); eval: resize to 224 (as v1)


def read_manifest(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    for r in rows:
        r['class_id'] = int(r['class_id'])
    classes = sorted({(r['class_id'], r['dataset'], r['class_name']) for r in rows})
    assert [c[0] for c in classes] == list(range(len(classes))), 'class ids must be 0..C-1'
    class_names = [f'{d}_{n}' for _, d, n in classes]
    return rows, class_names


def load_cache(cache_dir, split, device, manifest_sha, rows):
    """rows: the manifest rows of this split, in manifest order.
    Returns (X uint8 [N,3,S,S] on device, Y long [N] on device, meta)."""
    d = Path(cache_dir)
    meta = json.loads((d / 'meta.json').read_text())
    sm = meta['splits'].get(split)
    if sm is None or sm['manifest_sha256'] != manifest_sha:
        raise SystemExit(f'cache {d} has no {split} split built from this manifest; rebuild it')
    paths = (d / f'{split}_paths.txt').read_text(encoding='utf-8').splitlines()
    if paths != [r['path'] for r in rows]:
        raise SystemExit(f'cache {split} order differs from manifest; rebuild it')
    Xn = np.load(d / f'{split}_x.npy'); Yn = np.load(d / f'{split}_y.npy')
    size = TRAIN_CACHE_SIZE if split == 'train' else EVAL_SIZE
    if Xn.shape != (len(rows), 3, size, size) or Xn.dtype != np.uint8:
        raise SystemExit(f'cache {split} has shape {Xn.shape} {Xn.dtype}, expected ({len(rows)}, 3, {size}, {size}) uint8')
    if not np.array_equal(Yn, np.array([r['class_id'] for r in rows])):
        raise SystemExit(f'cache {split} labels differ from manifest; rebuild it')
    return torch.from_numpy(Xn).to(device), torch.from_numpy(Yn).long().to(device), meta


def _gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def _rgb_to_hsv(x):
    r, g, b = x.unbind(1)
    maxc, minc = x.amax(1), x.amin(1)
    delta = maxc - minc
    s = torch.where(maxc > 0, delta / maxc.clamp_min(1e-8), torch.zeros_like(maxc))
    dc = delta.clamp_min(1e-8)
    rc, gc, bc = (maxc - r) / dc, (maxc - g) / dc, (maxc - b) / dc
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(delta > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    return torch.stack([h, s, maxc], 1)


def _hsv_to_rgb(x):
    h, s, v = x.unbind(1)
    i = torch.floor(h * 6); f = h * 6 - i
    i = (i.long() % 6).unsqueeze(1)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    pick = lambda *c: torch.stack(c, 1).gather(1, i).squeeze(1)  # noqa: E731
    return torch.stack([pick(v, q, p, p, t, v), pick(t, v, v, q, p, p), pick(p, p, t, v, v, q)], 1)


def gpu_augment(xu8, gen, out=EVAL_SIZE, rot_deg=15.0, jitter=(0.3, 0.3, 0.2, 0.05), p_gray=0.05):
    """Per-sample random augmentation of a uint8 batch [B,3,S,S] on the GPU -> float [B,3,out,out] in [0,1].

    Same family and ranges as v1's torchvision pipeline (ColorJitter 0.3/0.3/0.2/0.05, random
    224 crop of the 256 resize, H/V flips, rotation +-15 deg, grayscale p=0.05), applied as batched
    tensor ops: jitter in fixed order brightness -> contrast -> saturation -> hue, then crop, flips
    and rotation in one bilinear affine resampling. Pixels rotated in from outside the crop come
    from the 256 image; only those outside the 256 image are black."""
    B, dev = xu8.size(0), xu8.device
    U = lambda *s: torch.rand(*s, generator=gen, device=dev)  # noqa: E731
    bri, con, sat, hue = jitter
    x = xu8.float().div_(255)
    x = (x * (1 + (2 * U(B, 1, 1, 1) - 1) * bri)).clamp_(0, 1)
    m = _gray(x).mean((2, 3), keepdim=True)
    x = ((x - m) * (1 + (2 * U(B, 1, 1, 1) - 1) * con) + m).clamp_(0, 1)
    g = _gray(x)
    x = ((x - g) * (1 + (2 * U(B, 1, 1, 1) - 1) * sat) + g).clamp_(0, 1)
    hsv = _rgb_to_hsv(x)
    hsv[:, 0] = (hsv[:, 0] + (2 * U(B, 1, 1) - 1) * hue) % 1.0
    x = _hsv_to_rgb(hsv).clamp_(0, 1)

    S = xu8.size(-1)
    sc = out / S
    ox, oy = torch.floor(U(B) * (S - out + 1)), torch.floor(U(B) * (S - out + 1))
    th = (2 * U(B) - 1) * math.radians(rot_deg)
    fx = torch.where(U(B) < 0.5, -1.0, 1.0); fy = torch.where(U(B) < 0.5, -1.0, 1.0)
    A = torch.zeros(B, 2, 3, device=dev)
    A[:, 0, 0], A[:, 0, 1] = sc * fx * torch.cos(th), -sc * fx * torch.sin(th)
    A[:, 1, 0], A[:, 1, 1] = sc * fy * torch.sin(th), sc * fy * torch.cos(th)
    A[:, 0, 2], A[:, 1, 2] = (2 * ox + out - S) / S, (2 * oy + out - S) / S
    grid = F.affine_grid(A, (B, 3, out, out), align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return torch.where(U(B, 1, 1, 1) < p_gray, _gray(x).expand_as(x), x)


def to_float(xu8):
    return xu8.float().div_(255)


# ──────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────
class Normalized(nn.Module):
    """Wraps a backbone so it takes raw RGB in [0,1]; applies the backbone's own normalisation."""

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build_model(backbone, num_classes, pretrained=True):
    import timm
    from timm.data import resolve_data_config
    net = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    cfg = resolve_data_config({}, model=net)
    return Normalized(net, cfg['mean'], cfg['std'])


# ──────────────────────────────────────────────────────────────────────────
# ATTACKS (raw RGB domain, l_inf). Callers set train/eval mode; these never change it.
# ──────────────────────────────────────────────────────────────────────────
def fgsm(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    g, = torch.autograd.grad(loss, x_adv)
    return (x + eps * g.sign()).clamp(0, 1).detach()


def _pgd_once(model, x, y, eps, alpha, steps, loss):
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        if loss == 'ce':
            l = F.cross_entropy(logits.float(), y)
        else:  # CW margin loss
            logits = logits.float()
            onehot = F.one_hot(y, logits.size(1)).bool()
            other = logits.masked_fill(onehot, float('-inf')).amax(1)
            l = (other - logits[onehot]).clamp(max=50).sum()
        g, = torch.autograd.grad(l, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = (x + (x_adv - x).clamp(-eps, eps)).clamp(0, 1).detach()
    return x_adv


def pgd(model, x, y, eps, alpha, steps, restarts=1, loss='ce'):
    """Untargeted l_inf PGD with uniform random start. With restarts > 1, each further restart is
    run only on the samples not yet fooled; per sample the first fooling attempt is kept (or the
    last attempt if none fools it)."""
    if restarts == 1:
        return _pgd_once(model, x, y, eps, alpha, steps, loss)
    best = x.clone()
    todo = torch.arange(len(x), device=x.device)
    for _ in range(restarts):
        x_adv = _pgd_once(model, x[todo], y[todo], eps, alpha, steps, loss)
        with torch.no_grad():
            fooled = model(x_adv).argmax(1) != y[todo]
        best[todo] = x_adv
        todo = todo[~fooled]
        if len(todo) == 0:
            break
    return best


def trades_inner(model, x, eps, alpha, steps):
    """PGD maximising KL(p(x) || p(x_adv)) from a small random start, as in TRADES."""
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)
    x_adv = (x + 0.001 * torch.randn_like(x)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x_adv).float(), dim=1), p_clean.float(), reduction='sum')
        g, = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = (x + (x_adv - x).clamp(-eps, eps)).clamp(0, 1).detach()
    return x_adv


# ──────────────────────────────────────────────────────────────────────────
# COMMON CORRUPTIONS (ImageNet-C style, simplified; deterministic given the seed)
# ──────────────────────────────────────────────────────────────────────────
CORRUPTIONS = {
    'gaussian_noise': (0.04, 0.08, 0.12),   # noise std
    'gaussian_blur': (1.0, 2.0, 3.0),       # blur sigma (pixels at 224x224)
    'brightness': (0.1, 0.2, 0.3),          # additive shift
    'contrast': (0.6, 0.4, 0.2),            # contrast factor
    'jpeg': (40, 20, 10),                   # JPEG quality
}


def corrupt(x, kind, severity, seed=0):
    """x: float tensor (B,3,H,W) in [0,1] on any device; severity 1..3."""
    from torchvision.transforms.functional import gaussian_blur
    from torchvision.io import decode_jpeg, encode_jpeg
    v = CORRUPTIONS[kind][severity - 1]
    if kind == 'gaussian_noise':
        g = torch.Generator(device=x.device).manual_seed(seed)
        return (x + v * torch.randn(x.shape, generator=g, device=x.device)).clamp(0, 1)
    if kind == 'gaussian_blur':
        k = 2 * math.ceil(3 * v) + 1
        return gaussian_blur(x, [k, k], [v, v])
    if kind == 'brightness':
        return (x + v).clamp(0, 1)
    if kind == 'contrast':
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - m) * v + m).clamp(0, 1)
    if kind == 'jpeg':  # encode/decode on CPU (one-off cost at evaluation)
        u8 = (x * 255).round().to(torch.uint8).cpu()
        return (torch.stack([decode_jpeg(encode_jpeg(im, quality=int(v))) for im in u8]).float() / 255).to(x.device)
    raise ValueError(kind)


# ──────────────────────────────────────────────────────────────────────────
# SCHEDULE
# ──────────────────────────────────────────────────────────────────────────
def schedule(epoch, warmup, ramp, eps_max, w_max, ramp_what='both', steps_max=7):
    """(eps_t, w_t, steps_t, alpha_t) for 1-based epoch.

    w is the adversarial-loss weight (lambda for the mixed CE objective, beta for TRADES).
    ramp_what selects what ramps after the warm-up; everything else jumps to its maximum:
      both / eps / weight  SPRAT variants (perturbation budget and/or loss weight)
      steps                CAT-style curriculum: eps and weight at maximum, the number of
                           attack steps grows from 1 to steps_max (attack strength by iterations)
    Step size alpha_t = (eps_t / 4) * (steps_max / steps_t), so every configuration at full
    strength uses the same attack (alpha = eps/4, steps_max steps) and short attacks can still
    reach the boundary of the eps-ball. warmup = ramp = 0 means full strength from epoch 1."""
    if epoch <= warmup:
        return 0.0, 0.0, 0, 0.0
    k = epoch - warmup  # 1-based position in the ramp
    p = 1.0 if ramp == 0 else min(1.0, k / ramp)
    eps = eps_max * (p if ramp_what in ('both', 'eps') else 1.0)
    w = w_max * (p if ramp_what in ('both', 'weight') else 1.0)
    if ramp_what == 'steps' and ramp > 1 and k < ramp:
        steps = 1 + round((steps_max - 1) * (k - 1) / (ramp - 1))  # 1 at the first ramp epoch, steps_max at the last
    else:
        steps = steps_max
    return eps, w, steps, eps / 4 * steps_max / steps


def full_strength_epoch(warmup, ramp, eps_max=8 / 255, w_max=0.5, ramp_what='both', steps_max=7):
    """First 1-based epoch at which every scheduled quantity is at its maximum (computed from schedule())."""
    target = schedule(warmup + ramp + 1, warmup, ramp, eps_max, w_max, ramp_what, steps_max)
    for e in range(1, warmup + ramp + 2):
        if schedule(e, warmup, ramp, eps_max, w_max, ramp_what, steps_max) == target:
            return e
    return warmup + ramp + 1
