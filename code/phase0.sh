#!/usr/bin/env bash
# Phase 0 on the DGX, run as ONE detached job so nothing shares the GPU:
#   cache build -> GPU smoke queue (all modes, tiny budgets) -> CUDA kill/resume test -> aggregation
#   on the smoke runs -> training speed/parity benchmark -> inference efficiency benchmark.
#   bash code/v2/launch_detached.sh phase0 bash code/v2/phase0.sh
set -u
ROOT=/workspace/ravi/SPRAT; CODE="$ROOT/code/v2"; PY="$ROOT/.venv/bin/python"
REPORT="$ROOT/outputs/v2_report"; SMOKE="$ROOT/outputs/v2_smoke"
mkdir -p "$REPORT" "$SMOKE"
step() { echo; echo "===== [$(date '+%F %T')] $*"; }
fail=0

step "syntax"
for f in "$CODE"/*.sh; do bash -n "$f" || fail=1; done
$PY -m py_compile "$CODE"/*.py || fail=1

step "cache build (md5-verified)"
if [[ -f "$ROOT/data/cache_v2/meta.json" ]] && $PY - <<'PY'
import json, sys, hashlib
m = json.load(open('/workspace/ravi/SPRAT/data/cache_v2/meta.json'))
sha = hashlib.sha256(open('/workspace/ravi/SPRAT/data/manifest/manifest_sprat_v2_dedup.csv', 'rb').read()).hexdigest()
sys.exit(0 if all(m['splits'].get(s, {}).get('manifest_sha256') == sha for s in ('train', 'val', 'test')) else 1)
PY
then echo "cache already built for this manifest"
else
  $PY "$CODE/build_cache.py" --manifest "$ROOT/data/manifest/manifest_sprat_v2_dedup.csv" --root "$ROOT" \
      --out "$ROOT/data/cache_v2" --workers 32 || { echo "CACHE FAILED"; exit 1; }
fi

step "GPU smoke queue (3 epochs x 10 batches, eval on 256 test images)"
RUNS="$SMOKE/runs" RUNS_TSV="$CODE/smoke_runs.tsv" \
EXTRA_TRAIN="--epochs 3 --max_batches 10 --min_full_epochs 1" \
EXTRA_EVAL="--max_n 256 --boot 50 --pgd_steps 10,20 --fgsm_eps 8 --aa_n 64" \
  bash "$CODE/run_queue.sh" smoke 42 full || fail=1
grep -h -E "^(Epoch|\[DONE\]|Traceback|.*Error)" "$SMOKE"/runs/*/train.log | tail -n 40
grep -h -E "(acc=|Error|Traceback|DONE)" "$SMOKE"/runs/*/eval.log | tail -n 60

step "CUDA kill-and-resume test"
R="$SMOKE/resume_test"; rm -rf "$R"
$PY "$CODE/train_sprat_v2.py" --mode pgd --warmup 1 --ramp 1 --epochs 3 --max_batches 10 --min_full_epochs 1 \
    --seed 7 --out_dir "$R" --stop_after_epoch 1 || fail=1
$PY "$CODE/train_sprat_v2.py" --mode pgd --warmup 1 --ramp 1 --epochs 3 --max_batches 10 --min_full_epochs 1 \
    --seed 7 --out_dir "$R" --resume 2>&1 | grep -E "RESUME|Epoch|DONE|Error|refusing" || fail=1
[[ -f "$R/final.pth" ]] && echo "resume test: OK (final.pth written)" || { echo "resume test: FAILED"; fail=1; }

step "aggregation on smoke runs"
$PY "$CODE/aggregate_v2.py" --runs "$SMOKE/runs" --out "$SMOKE/report" --boot 200 --allow_debug_evals || fail=1
head -c 3000 "$SMOKE/report/summary.md"; echo; head -n 12 "$SMOKE/report/paired.csv"; cat "$SMOKE/report/warnings.txt"

step "training speed + bf16 attack-parity benchmark"
$PY "$CODE/bench_train_speed.py" --out "$REPORT/train_speed.json" || fail=1

step "inference efficiency benchmark"
$PY "$CODE/bench_efficiency.py" --out "$REPORT/efficiency.json" || fail=1

step "phase0 finished, fail=$fail"
exit $fail
