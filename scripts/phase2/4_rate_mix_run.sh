#!/usr/bin/env bash
# Arrival-rate and workload-mix supplement (docs/phase2/6_RATE_MIX_PREREG.md).
#
# Checks that the environment still matches the campaign lock, starts the same
# backend as eval, runs one unmeasured warm-up, then runs the four manifests in
# order (heavy conditions first). Safe to repeat: finished runs are skipped, so
# later nights continue where the last one stopped.
#
#   bash scripts/phase2/4_rate_mix_run.sh [--until HH:MM]   # default 09:00
#   touch runs/phase2/4_m4_rate_mix/STOP                    # no further runs start
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
R=runs/phase2/2_m4_campaign
X=runs/phase2/4_m4_rate_mix
UNTIL=09:00
[ "${1:-}" = "--until" ] && UNTIL=$2
LOG=$X/rate_mix.log
mkdir -p "$X"
# manifest name and initial per-run estimate in seconds (replaced by the longest run seen)
PLAN=("1_arrival_double 1500" "2_mix_long 1800" "3_mix_short 1700" "4_arrival_half 3300")

note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

deadline=$(date -j -f '%Y-%m-%d %H:%M' "$(date +%Y-%m-%d) $UNTIL" +%s)
[ "$deadline" -le "$(date +%s)" ] && deadline=$((deadline + 86400))
note "rate/mix driver started (deadline $(date -r "$deadline" '+%a %H:%M'))"

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
  --port 18080 --results-dir "$X/backend" >>"$LOG" 2>&1

for attempt in 1 2; do
  target=$X/warmup/$(date +%Y%m%d-%H%M%S)-$attempt
  note "warm-up: unmeasured replay in ${target#$X/}"
  if "$PY" -m phase2.bench --workload runs/final/workload_dev.json --results-dir "$target" \
      --policies fifo --seeds 1 --num-jobs 3 --cost-profile "$R/profiles/profile.json" \
      --platform-id m4-air-16g --gap lognormal:2:1 --arrival scaled:1 >>"$LOG" 2>&1; then
    break
  fi
  [ "$attempt" = 2 ] && { note "warm-up failed twice; stopping"; exit 1; }
done

for item in "${PLAN[@]}"; do
  read -r name estimate <<<"$item"
  [ -e "$X/STOP" ] && { note "stop file present"; break; }
  note "stage $name"
  if ! "$PY" -m phase2.plan run "$X/manifests/$name.json" --execute \
      --server-group P2-C24576-RAM0-idle0 --budget-s 100000000 --stop-file "$X/STOP" \
      --run-estimate-s "$estimate" --restart-server never --deadline-unix "$deadline" >>"$LOG" 2>&1; then
    note "plan runner reported a failed run in $name; stopping"
    break
  fi
done

stop_server
trap - EXIT
note "stage summary"
"$PY" -m phase2.summarize --results-dir "$X/sensitivity" --reference-dir "$R/eval" --short-cutoff 782.5 \
  --baseline affinity-fixed-best --baseline joint --baseline fifo --bootstrap-samples 2000 --plot >>"$LOG" 2>&1 \
  || note "summary failed"
"$PY" scripts/phase2/4_rate_mix_check.py >>"$LOG" 2>&1 || note "prediction check failed"
note "rate/mix driver finished"
