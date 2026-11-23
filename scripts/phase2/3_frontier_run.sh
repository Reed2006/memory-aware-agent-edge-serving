#!/usr/bin/env bash
# Fairness/efficiency frontier supplement (docs/phase2/5_FRONTIER_PREREG.md).
#
# Waits for the main campaign session to end, checks that the environment
# still matches the campaign lock, then runs the frontier manifest with the
# same server flags, warm-up and plan runner the campaign uses. Safe to repeat:
# finished runs are skipped.
#
#   bash scripts/phase2/3_frontier_run.sh [--until HH:MM]     # default 09:00
#   touch runs/phase2/3_m4_frontier/STOP       # no further runs start
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
R=runs/phase2/2_m4_campaign
F=runs/phase2/3_m4_frontier
UNTIL=09:00
[ "${1:-}" = "--until" ] && UNTIL=$2
LOG=$F/frontier.log
mkdir -p "$F"

note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

deadline=$(date -j -f '%Y-%m-%d %H:%M' "$(date +%Y-%m-%d) $UNTIL" +%s)
[ "$deadline" -le "$(date +%s)" ] && deadline=$((deadline + 86400))
note "frontier driver started (deadline $(date -r "$deadline" '+%a %H:%M'))"

while pgrep -f 'phase2\.campaign' >/dev/null; do sleep 60; done
note "main campaign session has ended"

"$PY" - "$R/campaign/lock.json" <<'EOF' 2>&1 | tee -a "$LOG"
import hashlib, json, sys
lock = json.load(open(sys.argv[1]))['environment']['sources']
bad = [p for p, h in lock.items() if hashlib.sha256(open(p, 'rb').read()).hexdigest() != h]
if bad:
    sys.exit('locked sources changed: ' + ', '.join(bad))
print(f'{len(lock)} locked sources match the campaign lock')
EOF
if pmset -g batt | grep -q "Battery Power"; then note "on battery power; not starting"; exit 1; fi
free_gib=$(df -g . | awk 'NR==2 {print $4}')
if [ "$free_gib" -lt 6 ]; then note "only ${free_gib} GiB free; not starting"; exit 1; fi
if pgrep -x llama-server >/dev/null; then note "a llama-server is already running; not starting"; exit 1; fi

stop_server() { "$PY" -m phase2.server stop >>"$LOG" 2>&1 || true; }
trap stop_server EXIT

note "starting llama-server"
# Output goes straight to the log, not through a pipe: the detached server
# would otherwise hold the pipe open and the script would never continue.
"$PY" -m phase2.server --parallel 2 --ctx-size 24576 --cache-ram 0 --no-cache-idle-slots \
  --port 18080 --results-dir "$F/backend" >>"$LOG" 2>&1

for attempt in 1 2; do
  target=$F/warmup/$(date +%Y%m%d-%H%M%S)-$attempt
  note "warm-up: unmeasured replay in ${target#$F/}"
  if "$PY" -m phase2.bench --workload runs/final/workload_dev.json --results-dir "$target" \
      --policies fifo --seeds 1 --num-jobs 3 --cost-profile "$R/profiles/profile.json" \
      --platform-id m4-air-16g --gap lognormal:2:1 --arrival scaled:1 >>"$LOG" 2>&1; then
    break
  fi
  [ "$attempt" = 2 ] && { note "warm-up failed twice; stopping"; exit 1; }
done

note "stage frontier-runs"
"$PY" -m phase2.plan run "$F/manifests/frontier.json" --execute \
  --server-group P2-C24576-RAM0-idle0 --budget-s 100000000 --stop-file "$F/STOP" \
  --run-estimate-s 1800 --restart-server never --deadline-unix "$deadline" >>"$LOG" 2>&1 \
  || note "plan runner exited with $?"

stop_server
trap - EXIT
note "stage frontier-summary"
"$PY" -m phase2.summarize --results-dir "$F/eval" --reference-dir "$R/eval" --short-cutoff 782.5 \
  --baseline affinity-fixed-best --baseline joint --bootstrap-samples 2000 --plot >>"$LOG" 2>&1 \
  || note "summary failed"
note "frontier driver finished"
