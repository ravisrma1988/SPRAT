#!/usr/bin/env bash
# Sequential train + evaluate queue for SPRAT v2. Safe to re-run: steps whose outputs match the
# current code/arguments are skipped, interrupted training resumes from last.pth, and a run whose
# provenance stamp differs (code or arguments changed) is moved aside and retrained.
#
#   bash run_queue.sh <phase|all> <seeds> [eval: full|quick]
#   phases (runs.tsv): pilot, main, ablation, backbone2
# Always start it detached (setsid + nohup, log + PID in outputs/v2_logs):
#   bash code/v2/launch_detached.sh pilot bash code/v2/run_queue.sh pilot 42 full
#   bash code/v2/launch_detached.sh main  bash code/v2/run_queue.sh all "42 123 3407" full
# Overrides (debug/smoke): RUNS=<dir> RUNS_TSV=<file> EXTRA_TRAIN="..." EXTRA_EVAL="..."
set -u
PHASE="$1"; SEEDS="$2"; EVAL="${3:-full}"
ROOT=/workspace/ravi/SPRAT
CODE="$ROOT/code/v2"
RUNS="${RUNS:-$ROOT/outputs/v2_runs}"
RUNS_TSV="${RUNS_TSV:-$CODE/runs.tsv}"
PY="$ROOT/.venv/bin/python"
TRAIN_FLAGS="$(cat "$CODE/train_flags.txt" 2>/dev/null || true)"   # speed flags chosen by bench_train_speed.py
EXTRA_TRAIN="${EXTRA_TRAIN:-}"; EXTRA_EVAL="${EXTRA_EVAL:-}"
mkdir -p "$RUNS"

phases=$(grep -v '^#' "$RUNS_TSV" | cut -f1 | sort -u | tr '\n' ' ')
if [[ "$PHASE" != all && " $phases " != *" $PHASE "* ]]; then
  echo "unknown phase '$PHASE' (known: $phases all)"; exit 2
fi
exec 8>"$RUNS/.queue.lock"
flock -n 8 || { echo "another queue is already running on $RUNS"; exit 1; }

n_train=0; n_skip=0; n_fail=0; n_eval=0; n_efail=0
run_eval() {  # $1 out  $2 ckpt-name  $3 attacks  $4 extra args
  local out="$1" ck="$2" attacks="$3" extra="$4" ed="$1/eval_$2"
  [[ -f "$out/$ck.pth" ]] || { echo "  (no $ck.pth)"; return; }
  if $PY "$CODE/queue_check.py" eval_done "$ed" "$out/$ck.pth" "$attacks" 3 < /dev/null; then return; fi
  echo "[$(date '+%F %T')] EVAL  $(basename "$out") $ck [$attacks]"
  if $PY "$CODE/eval_sprat_v2.py" --ckpt "$out/$ck.pth" --out "$ed" --attacks "$attacks" --pgd_restarts 3 \
       $extra $EXTRA_EVAL < /dev/null >> "$out/eval.log" 2>&1; then
    n_eval=$((n_eval + 1)); rm -f "$ed/FAILED"
  else
    n_efail=$((n_efail + 1)); mkdir -p "$ed"; touch "$ed/FAILED"; echo "  EVAL FAILED (see $out/eval.log)"
  fi
}

TRADES_CHOICE="$(cat "$CODE/trades_choice.txt" 2>/dev/null || true)"   # written by select_trades.py
while IFS=$'\t' read -r -u 3 phase name targs; do
  [[ "$phase" =~ ^# || -z "$phase" ]] && continue
  [[ "$PHASE" != all && "$PHASE" != "$phase" ]] && continue
  [[ "$PHASE" == all && "$phase" == tune ]] && continue   # tuning runs only when asked for explicitly
  if [[ "$targs" == *@TRADES@* ]]; then
    if [[ -z "$TRADES_CHOICE" ]]; then echo "  skip $name: TRADES not selected yet (trades_choice.txt missing)"; continue; fi
    targs="${targs//@TRADES@/$TRADES_CHOICE}"
  fi
  for seed in $SEEDS; do
    out="$RUNS/${name}_s${seed}"
    stamp=$($PY "$CODE/queue_check.py" stamp "$targs $TRAIN_FLAGS $EXTRA_TRAIN" < /dev/null)
    if $PY "$CODE/queue_check.py" train_done "$out" "$stamp" < /dev/null; then
      n_skip=$((n_skip + 1))
    else
      if [[ -f "$out/.stamp" && "$(cat "$out/.stamp")" != "$stamp" ]]; then
        stale="${out}.stale_$(date +%Y%m%d_%H%M%S)"
        echo "  provenance changed for $(basename "$out"): moving old run to $(basename "$stale")"
        mv "$out" "$stale"
      fi
      mkdir -p "$out"; echo "$stamp" > "$out/.stamp"
      echo "[$(date '+%F %T')] TRAIN $name seed=$seed"
      if $PY "$CODE/train_sprat_v2.py" $targs $TRAIN_FLAGS $EXTRA_TRAIN --seed "$seed" --out_dir "$out" --resume \
           < /dev/null >> "$out/train.log" 2>&1; then
        n_train=$((n_train + 1)); rm -f "$out/FAILED"
      else
        n_fail=$((n_fail + 1)); touch "$out/FAILED"; echo "  TRAIN FAILED (see $out/train.log)"; continue
      fi
    fi
    [[ "$phase" == tune ]] && continue                       # validation-only tuning: no test evaluation
    if [[ "$name" == clean* ]]; then
      primary=best_clean; secondary="final"
    else
      primary=best_robust; secondary="final best_clean"
    fi
    extra=""; [[ "$name" == naive_pgd* || "$name" == madry_pgd* || "$name" == sprat_w10r10* ]] && extra="--bn_recal 100"
    attacks=clean,fgsm,pgd,corrupt,aa; [[ "$EVAL" == quick ]] && attacks=clean,fgsm,pgd,corrupt
    run_eval "$out" "$primary" "$attacks" "$extra"
    for ck in $secondary; do
      run_eval "$out" "$ck" "clean,fgsm,pgd" ""   # same PGD/FGSM suite (3 restarts), no AutoAttack/corruptions
    done
  done
done 3< "$RUNS_TSV"
echo "[$(date '+%F %T')] queue done: phase=$PHASE seeds=$SEEDS trained=$n_train skipped=$n_skip train_failed=$n_fail evals=$n_eval eval_failed=$n_efail"
[[ $n_fail -eq 0 && $n_efail -eq 0 ]]
