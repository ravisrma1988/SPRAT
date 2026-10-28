#!/usr/bin/env bash
# Smoke-test the tune -> select -> @TRADES@ flow with tiny budgets in outputs/v2_smoke/flow, then (only if
# it passed) remove the smoke selection and start the real grid as its own detached job.
#   bash code/v2/launch_detached.sh flowtest bash code/v2/flowtest_then_grid.sh
ROOT=/workspace/ravi/SPRAT; CODE="$ROOT/code/v2"; PY="$ROOT/.venv/bin/python"
S="$ROOT/outputs/v2_smoke/flow"; rm -rf "$S"; mkdir -p "$S"
rm -f "$CODE/trades_choice.txt"
export RUNS="$S" EXTRA_TRAIN="--epochs 3 --max_batches 5 --min_full_epochs 1" EXTRA_EVAL="--max_n 128 --boot 20 --pgd_steps 10 --fgsm_eps 8"
ok=1
bash "$CODE/run_queue.sh" tune 42 quick || ok=0
ls "$S"
$PY "$CODE/select_trades.py" "$S" || ok=0
echo "choice: $(cat "$CODE/trades_choice.txt" 2>/dev/null)"
bash "$CODE/run_queue.sh" pilot 42 quick || ok=0
grep -h "^\[RUN\]" "$S"/trades_s42/train.log | cut -c1-200
[[ -f "$S/trades_s42/eval_best_robust/eval.json" ]] || ok=0
[[ ! -d "$S/trades_tune_b1_s42/eval_best_robust" ]] || { echo "tune run was evaluated (should not be)"; ok=0; }
rm -f "$CODE/trades_choice.txt"
if [[ $ok -eq 1 ]]; then
  echo "FLOW TEST PASSED - launching grid"
  unset RUNS EXTRA_TRAIN EXTRA_EVAL
  bash "$CODE/launch_detached.sh" grid bash "$CODE/grid.sh"
else
  echo "FLOW TEST FAILED - grid NOT launched"; exit 1
fi
