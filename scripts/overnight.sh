#!/bin/bash
# Full Phase 1 pipeline: live capture -> micro -> cache -> macro rounds -> dev sweep -> report.
# Runs sequentially (never two experiments on the server at once). Each experiment is
# skipped, not truncated, once HOURS have elapsed; the report is regenerated after every round.
# Resumable: captures, micro/cache and logged macro runs are skipped on re-run.
# Usage (Mac):  nohup caffeinate -dims bash scripts/overnight.sh > /dev/null 2>&1 &
# Usage (Linux/EC2, CPU), inside tmux:
#   WORKLOADS=runs/final OUT=runs/aws DEV=runs/aws-dev SERVER_ARGS="--cpu --threads 8" \
#     TIMEOUT=1800 HOURS=9 STOP_WHEN_DONE=1 bash scripts/overnight.sh
# Environment:
#   WORKLOADS       reuse this directory's frozen workload_{eval,dev}.json and skip live capture
#                   (no DeepSeek key needed); unset = capture live as on the Mac
#   SERVER_ARGS     extra phase1.server arguments for every launch, e.g. "--cpu --threads 8"
#   TIMEOUT         per-request HTTP deadline in seconds (raise it for slow CPU prefill)
#   STOP_WHEN_DONE  1 = power the machine off when the pipeline ends, and at the latest
#                   HOURS + 30 min after start (needs passwordless sudo, as on EC2)
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/final}
DEV=${DEV:-runs/final-dev}
PY=.venv/bin/python
HOURS=${HOURS:-9}
WORKLOADS=${WORKLOADS:-}
SERVER_ARGS=${SERVER_ARGS:-}
TIMEOUT=${TIMEOUT:-600}
STOP_WHEN_DONE=${STOP_WHEN_DONE:-0}
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
mkdir -p "$OUT/logs" "$OUT/raw" "$DEV/raw"
exec >> "$OUT/logs/pipeline.log" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }
in_time() { [ "$(date +%s)" -lt "$DEADLINE" ]; }
clock() { date -r "$1" '+%F %T' 2>/dev/null || date -d "@$1" '+%F %T'; }  # BSD, then GNU date

server() {  # server <args...>: restart llama-server with this configuration
  $PY -m phase1.server stop
  log "server $SERVER_ARGS $*"
  # SERVER_ARGS is split into words on purpose.
  $PY -m phase1.server --results-dir "$OUT" $SERVER_ARGS "$@" || { log "SERVER START FAILED $*"; return 1; }
  if command -v pmset > /dev/null; then pmset -g batt | head -1; fi
}

bench() {  # bench <results-dir> <bench args...>
  local dir=$1; shift
  if ! in_time; then log "SKIP (deadline) $*"; return; fi
  log "bench $*"
  $PY -m phase1.bench "$@" --results-dir "$dir" --timeout "$TIMEOUT" || log "BENCH FAILED $*"
  sleep 2
}

report() {
  $PY -m phase1.summarize --results-dir "$OUT" > "$OUT/logs/summary.txt" 2>&1 || true
  $PY -m phase1.final --results-dir "$OUT" || log "REPORT FAILED"
}

log "=== pipeline start, deadline $(clock $DEADLINE), commit $(git rev-parse --short HEAD)"
if [ "$STOP_WHEN_DONE" = 1 ]; then
  sudo shutdown -h +$(( HOURS * 60 + 30 )) && log "machine will stop by $(clock $(( DEADLINE + 1800 )))"
fi

# ---- 1. live capture (resumes: existing task files are skipped) -------------------
server --parallel 2 --ctx-size 24576
if [ -n "$WORKLOADS" ]; then
  for name in workload_eval.json workload_dev.json; do
    [ -s "$OUT/$name" ] || cp "$WORKLOADS/$name" "$OUT/$name"
  done
  log "workloads from $WORKLOADS; live capture skipped"
else
  log "capture test 20"
  $PY -m phase1.capture --split test --count 20 --max-steps 40 --results-dir "$OUT" || log "CAPTURE TEST FAILED"
  log "capture train 10"
  $PY -m phase1.capture --split train --count 10 --max-steps 40 --results-dir "$OUT" || log "CAPTURE TRAIN FAILED"
fi

[ -s "$OUT/workload_eval.json" ] || $PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
from phase1.bench import make_workload
out = Path(sys.argv[1])
for split, name in [('test', 'workload_eval.json'), ('train', 'workload_dev.json')]:
    w = make_workload(out / f'raw/live_{split}', 0)
    (out / name).write_text(json.dumps(w))
    print(name, len(w['tasks']), 'tasks,', sum(len(t['steps']) for t in w['tasks']), 'steps')
EOF

# ---- 2. micro and cache (P=2, 24K: 12K per slot) ------------------------------------
[ -s "$OUT/raw/micro_requests.jsonl" ] || bench "$OUT" micro --lengths 1024,4096,8192 --outputs 64,512 --k 1,2 --repeats 2
[ -s "$OUT/raw/cache.jsonl" ] || bench "$OUT" cache --lengths 2048,8192 --repeats 2
if [ ! -s "$OUT/restore_validation.json" ] && in_time; then $PY -m phase1.check_restore --results-dir "$OUT" || log "RESTORE CHECK FAILED"; fi
report

# ---- 3. macro rounds on the eval workload ---------------------------------------------
EVAL="macro --workload $OUT/workload_eval.json --num-jobs 8 --repeats 1 --skip-done"
MAIN="--gap lognormal:2:1 --arrival poisson:1"

round() {  # round <repeat index> <full|core>
  local r=$1 scope=$2
  log "=== round $r ($scope)"
  server --parallel 2 --ctx-size 24576 && {
    bench "$OUT" $EVAL --repeat-start $r $MAIN --k 2 --jobs-active 0
    bench "$OUT" $EVAL --repeat-start $r $MAIN --k 1 --jobs-active 1
    bench "$OUT" $EVAL --repeat-start $r $MAIN --k 1 --jobs-active 0
    bench "$OUT" $EVAL --repeat-start $r $MAIN --k 2 --jobs-active 2
    if [ "$scope" = full ]; then
      bench "$OUT" $EVAL --repeat-start $r --arrival poisson:1 --gap zero --k 2
      bench "$OUT" $EVAL --repeat-start $r --arrival poisson:1 --gap fixed:15 --k 2
      bench "$OUT" $EVAL --repeat-start $r --gap lognormal:2:1 --arrival poisson:0.5 --k 2
      bench "$OUT" $EVAL --repeat-start $r --gap lognormal:2:1 --arrival poisson:2 --k 2
    fi
  }
  in_time && server --parallel 1 --ctx-size 24576 && bench "$OUT" $EVAL --repeat-start $r $MAIN --k 1
  in_time && server --parallel 4 --ctx-size 24576 && bench "$OUT" $EVAL --repeat-start $r $MAIN --k 4
  in_time && server --parallel 4 --ctx-size 24576 --kv-unified && bench "$OUT" $EVAL --repeat-start $r $MAIN --k 4
  in_time && server --parallel 2 --ctx-size 24576 --cache-ram 3072 && bench "$OUT" $EVAL --repeat-start $r $MAIN --k 2
  report
}

round 0 full
round 1 full

# ---- 4. dev workload sweep (input to Phase 2 baseline selection) ---------------------
if in_time; then
  server --parallel 2 --ctx-size 24576 && \
    bench "$DEV" macro --workload "$OUT/workload_dev.json" --num-jobs 8 --repeats 1 --skip-done $MAIN --k 1,2 --jobs-active 0,2
fi

# ---- 5. third repeat and a larger KV budget if time and memory allow -------------------
round 2 core
avail=$($PY -c 'import psutil; print(int(psutil.virtual_memory().available / 2**30))')
if in_time && [ "$avail" -ge 7 ]; then
  for r in 0 1; do
    in_time && server --parallel 4 --ctx-size 36864 && bench "$OUT" $EVAL --repeat-start $r $MAIN --k 4
  done
else
  log "skip P4/36K budget (available ${avail} GiB or deadline)"
fi

$PY -m phase1.server stop
report
log "=== pipeline done"
if [ "$STOP_WHEN_DONE" = 1 ]; then
  log "stopping the machine in 2 minutes"
  sudo shutdown -h +2
fi
