#!/usr/bin/env bash
# Status of a detached job and of all v2 runs. Read-only.
#   bash queue_status.sh [job_name]
ROOT=/workspace/ravi/SPRAT
LOGDIR="$ROOT/outputs/v2_logs"
RUNS="$ROOT/outputs/v2_runs"
for pidf in "$LOGDIR"/${1:-*}.pid; do
  [[ -f "$pidf" ]] || continue
  n=$(basename "$pidf" .pid); pid=$(cat "$pidf")
  stat=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')
  if [[ -f "$LOGDIR/$n.exit" ]]; then st="FINISHED exit=$(cat "$LOGDIR/$n.exit")"
  elif [[ -n "$stat" && "$stat" != Z* ]]; then st="RUNNING"
  else st="STOPPED (no exit code: killed or crashed)"; fi
  echo "== job $n  pid=$pid  $st"
  tail -n 3 "$LOGDIR/$n.latest.log" 2>/dev/null | sed 's/^/   /'
done
echo "== runs (epochs done / configured, primary eval present)"
for d in "$RUNS"/*/; do
  [[ -f "$d/config.json" ]] || continue
  python3 - "$d" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); c = json.loads((d / 'config.json').read_text())
h = json.loads((d / 'history.json').read_text()) if (d / 'history.json').exists() else []
last = h[-1] if h else {}
ev = [p.parent.name for p in d.glob('eval_*/eval.json')]
bad = (['TRAIN-FAILED'] if (d / 'FAILED').exists() else []) + [p.parent.name + '-FAILED' for p in d.glob('eval_*/FAILED')]
print(f"   {d.name:32s} {len(h):2d}/{c['epochs']}  val clean={last.get('val_clean_acc', 0)*100:5.1f}% "
      f"pgd={last.get('val_pgd_acc', 0)*100:5.1f}%  {last.get('train_s', 0):.0f}s/ep  evals={','.join(ev) or '-'} "
      f"{' '.join(bad)}")
PY
done
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/== gpu (may read N\/A under MIG): /'
