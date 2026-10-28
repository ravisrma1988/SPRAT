#!/usr/bin/env bash
# Start any long job fully detached from the terminal / Jupyter session:
# new session (setsid), immune to hangups (nohup), no stdin, log + PID file in outputs/v2_logs.
# When the job ends it writes its exit code to <name>.exit. (In this container orphaned processes
# are not reaped, so a finished job can linger as a zombie: `kill -0` alone is not a liveness test.)
#
#   bash launch_detached.sh <job_name> <command> [args...]
#   e.g. bash launch_detached.sh phase0 bash code/v2/phase0.sh
#        bash launch_detached.sh grid   bash code/v2/grid.sh
#
# Check:  bash code/v2/queue_status.sh [job_name]
# Stop:   kill -- -$(cat outputs/v2_logs/<job_name>.pid)      (kills the whole process group)
set -eu
ROOT=/workspace/ravi/SPRAT
LOGDIR="$ROOT/outputs/v2_logs"
mkdir -p "$LOGDIR"
name="$1"; shift
pidf="$LOGDIR/$name.pid"; exitf="$LOGDIR/$name.exit"
if [[ -f "$pidf" && ! -f "$exitf" ]] && ps -o stat= -p "$(cat "$pidf")" 2>/dev/null | grep -qv '^Z'; then
  echo "job '$name' is already running (pid $(cat "$pidf")); not starting a second copy" >&2
  exit 1
fi
log="$LOGDIR/${name}_$(date +%Y%m%d_%H%M%S).log"
rm -f "$exitf"
cd "$ROOT"
setsid nohup bash -c '"$@"; echo $? > "$0"' "$exitf" "$@" > "$log" 2>&1 < /dev/null &
echo $! > "$pidf"
ln -sfn "$log" "$LOGDIR/$name.latest.log"
echo "started '$name' pid=$(cat "$pidf")"
echo "log: $log"
