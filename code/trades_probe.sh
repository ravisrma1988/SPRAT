#!/usr/bin/env bash
# TRADES gate probe (2026-10-24): three TRADES variants, 10 epochs each of the real 30-epoch cosine
# schedule (seed 42), to find a configuration that does not collapse to near-uniform outputs.
#   bash code/v2/launch_detached.sh probe bash code/v2/trades_probe.sh
ROOT=/workspace/ravi/SPRAT; CODE="$ROOT/code/v2"; PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs/v2_probe"; mkdir -p "$OUT"
FLAGS="$(cat "$CODE/train_flags.txt")"
run() {  # name, extra args
  echo "===== [$(date '+%F %T')] $1: $2"
  $PY "$CODE/train_sprat_v2.py" --mode trades --w_max 6 --seed 42 --epochs 30 --stop_after_epoch 10 \
      $FLAGS $2 --out_dir "$OUT/$1" < /dev/null > "$OUT/$1.log" 2>&1
  grep -E "^Epoch" "$OUT/$1.log" | cut -c1-200
}
run trades_ls0          "--label_smoothing 0"
run trades_ls01_detach  "--label_smoothing 0.1 --trades_detach 1"
run trades_ls0_detach   "--label_smoothing 0 --trades_detach 1"
echo "===== probe done $(date '+%F %T')"
