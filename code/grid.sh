#!/usr/bin/env bash
# The full v2 grid as one sequential job:
#   1. TRADES tuning on validation (seed 42) and pre-declared selection (select_trades.py)
#   2. seed 42 for every condition (pilot conditions are the first rows of runs.tsv)
#   3. seeds 123 and 3407
#   4. aggregate report
# A failure in one run does not stop the rest (each run is marked FAILED and the queue's final line
# counts failures).   bash code/v2/launch_detached.sh grid bash code/v2/grid.sh
ROOT=/workspace/ravi/SPRAT; CODE="$ROOT/code/v2"; PY="$ROOT/.venv/bin/python"
echo "train flags: $(cat "$CODE/train_flags.txt")"
bash "$CODE/run_queue.sh" tune 42 full; s0=$?
$PY "$CODE/select_trades.py" || { echo "TRADES selection failed: TRADES conditions will be skipped"; }
echo "TRADES choice: $(cat "$CODE/trades_choice.txt" 2>/dev/null)"
bash "$CODE/run_queue.sh" all 42 full; s1=$?
bash "$CODE/run_queue.sh" all "123 3407" full; s2=$?
$PY "$CODE/aggregate_v2.py" --runs "$ROOT/outputs/v2_runs" --out "$ROOT/outputs/v2_report"; s3=$?
echo "grid finished: tune=$s0 seed42=$s1 seeds123_3407=$s2 aggregate=$s3"
[[ $s0 -eq 0 && $s1 -eq 0 && $s2 -eq 0 && $s3 -eq 0 ]]
