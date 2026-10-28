"""Model-complexity and inference-cost benchmark (R1, R3.8). Run it ALONE on the GPU (not next to training).

The training method does not change the architecture, so inference cost is measured once per
backbone. Training overhead per method comes from the per-epoch times in each run's history
(aggregate_v2.py -> cost.csv).

Reports: parameters; forward FLOPs and MACs at 224x224 (torch FlopCounterMode with timm fused
attention disabled so attention matmuls are counted; FLOPs = 2 x MACs); median / p90 latency for
batch 1 and batch 32 on the GPU; batch-1 latency on CPU with a fixed thread count (a stand-in for an
edge CPU, not an edge-device measurement); peak GPU memory; hardware provenance; and a check that the
pretrained weights of every backbone download/load.

Usage: python bench_efficiency.py --backbones mobilevit_s,efficientnet_b0 --out efficiency.json
"""
import os
os.environ['TIMM_FUSED_ATTN'] = '0'  # count attention FLOPs explicitly
import argparse, json, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprat_common import build_model  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--backbones', default='mobilevit_s,efficientnet_b0')
p.add_argument('--num_classes', type=int, default=19)
p.add_argument('--iters', type=int, default=100)
p.add_argument('--cpu_threads', type=int, default=4)
p.add_argument('--out', required=True)
args = p.parse_args()


def timed(model, x, iters, sync):
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if sync: torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); model(x)
            if sync: torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
    return dict(median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)))


def cpu_model_name():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return repr(e)


res = dict(torch=torch.__version__, python=platform.python_version(), cpu=cpu_model_name(),
           cpu_affinity=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
           cpu_threads_for_latency=args.cpu_threads,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           nvidia_smi_L=sh(['nvidia-smi', '-L']), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
           gpu_processes_at_start=sh(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader']),
           models={})
for bb in args.backbones.split(','):
    pre = build_model(bb, args.num_classes, pretrained=True)  # also verifies the pretrained download
    r = dict(pretrained_tag=pre.net.pretrained_cfg.get('hf_hub_id') or pre.net.pretrained_cfg.get('tag'),
             input_mean=pre.mean.flatten().tolist(), input_std=pre.std.flatten().tolist())
    m = pre.eval()
    r['params_M'] = sum(q.numel() for q in m.parameters()) / 1e6
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(torch.rand(1, 3, 224, 224))
    r['gflops'] = fc.get_total_flops() / 1e9
    r['gmacs'] = r['gflops'] / 2
    torch.set_num_threads(args.cpu_threads)
    r['cpu_b1'] = timed(m, torch.rand(1, 3, 224, 224), max(20, args.iters // 5), False)
    if torch.cuda.is_available():
        g = m.cuda()
        for bs in (1, 32):
            torch.cuda.reset_peak_memory_stats()
            r[f'gpu_b{bs}'] = timed(g, torch.rand(bs, 3, 224, 224, device='cuda'), args.iters, True)
            r[f'gpu_b{bs}']['peak_mem_MB'] = torch.cuda.max_memory_allocated() / 2**20
            r[f'gpu_b{bs}']['images_per_s'] = bs * 1000 / r[f'gpu_b{bs}']['median_ms']
    res['models'][bb] = r
    print(bb, json.dumps(r), flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=2))
print('[DONE]', args.out, flush=True)
