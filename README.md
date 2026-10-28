# SPRAT: Scheduled Progressive Adversarial Training for Lightweight Vision Transformers in Plant Disease Recognition

Code, split manifest and per-seed results for the revised IEEE Access submission (manuscript Access-2026-45041).
Everything needed to reproduce the reported numbers is here, except the two public image collections, which must be
downloaded from their original sources (see Data).

Repository: https://github.com/ravisrma1988/SPRAT

```bash
git clone https://github.com/ravisrma1988/SPRAT.git
```

## Contents

| Folder | What it holds |
|---|---|
| `code/` | Training, evaluation, aggregation and run-queue scripts, byte-identical to the versions that produced the reported results (SHA-256 of every file in `SHA256SUMS`) |
| `data_pipeline/` | Near-duplicate audit and the group-aware deduplicated split that produced the manifest |
| `manifest/` | `manifest_sprat_v2_dedup.csv` (13,605 images: path, class, split, duplicate group, MD5) and its card (counts, method, SHA-256) |
| `results/` | Aggregated outputs of all 60 runs: mean and per-seed tables, paired statistics, AutoAttack component breakdown, BatchNorm recalibration diagnostic, per-class metrics, confusion matrices, training histories, efficiency and training-cost measurements |
| `results/runs/` | For every run (60 grid runs and 4 TRADES tuning runs): the full configuration with code and manifest hashes, the per-epoch schedule and history, the training summary and every evaluation file (`eval_<checkpoint>.json`, including per-class metrics and the AutoAttack log) |
| `paper/` | Script that generates every results table and figure of the article from `results/` |

## Data

- Wheat: "Wheat Plant Diseases" collection on Kaggle (owner `kushagra3204`).
- Rice: "Rice Leaf Disease Images", Mendeley Data, DOI 10.17632/fwcj7stb8r.1.

Place the extracted collections so that the manifest paths resolve relative to the project root:
`data/dataset/wheat/data/{train,valid,test}/<class>/...` and `data/dataset/rice/Rice Leaf Disease Images/<class>/...`.
Note that in some distributed copies the two archive names are swapped (the archive named after wheat holds the rice
images and vice versa). The MD5 column of the manifest identifies every file.

## Leakage-controlled split

The original wheat splits share copies of most test images with training, and the rice collection consists of 329
photographs expanded by rotation and flipping. `data_pipeline/` hashes every image with a 64-bit DCT perceptual hash
under all eight rotations and flips, links images within Hamming distance 6 into groups across both crops, removes groups
with conflicting labels and splits groups 70/15/15 within each class (seed 20261023). Training keeps every distinct file
of a group, validation and test keep one representative per group. The released manifest has SHA-256
`848c8d6b96f2a7d98ca43176273a820812fb4c48935ea78cb61da558b4f5121b` and is the one every run used. Rebuilding it is
optional (`leakage_audit.py`, `dedup_split.py`, then `build_canonical_manifest.py`, whose `LOCAL_ROOTS` must be edited
to the local download location).

## Environment

Results were produced in the NVIDIA PyTorch container 25.03 (Ubuntu 24.04, Python 3.12.3) on one NVIDIA B200 MIG
1g.45gb slice: torch 2.7.0a0 (nv25.03), torchvision 0.22.0a0, timm 1.0.29, numpy 1.26.4, Pillow 11.1.0,
matplotlib 3.10.1. AutoAttack is the reference implementation (github.com/fra31/auto-attack), commit
`a39220048b3c9f2cca9a4d3a54604793c68eca7e`, standard version. See `requirements.txt`.

Scripts assume the project root `/workspace/ravi/SPRAT` (variables `ROOT` in the shell scripts and default arguments
in the Python scripts). Change it to the local project root before running.

## Reproducing the results

1. Cache the images in the uint8 format used for training (train at 256 px, validation and test at 224 px):
   `python code/build_cache.py --manifest manifest/manifest_sprat_v2_dedup.csv --root . --out data/cache_v2`
2. Run the grid. `code/runs.tsv` lists every condition with its exact training arguments, and `code/train_flags.txt`
   the flags added to every run (`--amp bf16 --channels_last 1`). `code/grid.sh` runs the validation-only TRADES
   selection (`select_trades.py`), seed 42, then seeds 123 and 3407, evaluates every checkpoint and aggregates.
   `code/run_queue.sh` can run a single phase or a subset of runs, and long jobs are started detached with
   `code/launch_detached.sh <name> bash code/grid.sh`. A single run, for example SPRAT with seed 42:
   `python code/train_sprat_v2.py --mode pgd --w_max 0.5 --warmup 10 --ramp 10 --seed 42 --amp bf16 --channels_last 1
   --out_dir outputs/v2_runs/sprat_w10r10_s42`
3. Evaluate a checkpoint (primary checkpoints with the full suite, as in `run_queue.sh`):
   `python code/eval_sprat_v2.py --ckpt <run>/best_robust.pth --out <run>/eval_best_robust
   --attacks clean,fgsm,pgd,corrupt,aa --pgd_restarts 3` (add `--bn_recal 100` for the BatchNorm diagnostic).
4. Aggregate: `python code/aggregate_v2.py --runs outputs/v2_runs --out outputs/v2_report` (summary and paired
   statistics at the primary checkpoint), the same with `--ckpts final,best_robust,best_clean` for final-checkpoint
   statistics, and `python code/aggregate_extra_v2.py --runs outputs/v2_runs --out outputs/v2_report/extra`.
5. Paper tables and figures: `python paper/make_paper_assets.py` (arguments in its docstring).

Training is seeded (data order, augmentation, initialisation, attack starts) but GPU kernels are not bitwise
deterministic, so a rerun reproduces the results statistically rather than to the last digit.

## Protocol in brief

- MobileViT-S (timm `mobilevit_s.cvnets_in1k`, input in [0,1]) and EfficientNet-B0 (timm `efficientnet_b0.ra_in1k`,
  ImageNet normalisation applied inside the model), AdamW (lr 3e-4, weight decay 1e-4), per-step cosine decay without
  restarts, batch 32, 30 epochs, gradient-norm clipping at 1.0, no label smoothing, seeds 42, 123 and 3407.
- Training attack: l-infinity PGD, epsilon 8/255, 7 steps, step size epsilon/4 at full strength.
- Checkpoints: `best_robust` (highest validation PGD-10 accuracy among full-strength epochs, the primary checkpoint),
  `final` and `best_clean`.
- Test evaluation: FGSM, PGD-10/20 (cross-entropy), PGD-50 with cross-entropy and margin loss and three restarts,
  standard AutoAttack at 8/255 on the primary checkpoint, five common corruptions at three severities, and a BatchNorm
  recalibration diagnostic.

## File checksums (scripts used for the reported runs)

`SHA256SUMS` lists the SHA-256 of every file in this repository (check with `sha256sum -c SHA256SUMS`). The run
records under `results/runs/` store `script_sha256`, `common_sha256` and `eval_script_sha256`, which equal the
checksums of `code/train_sprat_v2.py`, `code/sprat_common.py` and `code/eval_sprat_v2.py`:

| File | SHA-256 |
|---|---|
| `code/train_sprat_v2.py` | `ede0379f5a815f26d25f6c288ee366c0358e7fde43eba092dce1190d9ba98e59` |
| `code/sprat_common.py` | `212866a2cce9b2f388d7ee709b9ed0eb36d45556e7c0cf30411f150bea759b90` |
| `code/eval_sprat_v2.py` | `b14555425cd3eaaba5a60efa9e4295c76694ef0fcf92a2fdbb5248fcfb47ef56` |

## Citation

If you use this code or the split, please cite the article (citation to be added on publication).

## License

A licence file will be added by the authors.
