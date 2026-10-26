#!/bin/bash
# Full Phase 1 pipeline: live capture -> micro -> cache -> macro rounds -> dev sweep -> report.
# Runs sequentially (never two experiments on the server at once). Each experiment is
# skipped, not truncated, once HOURS have elapsed; the report is regenerated after every round.
# Usage: nohup caffeinate -dims bash scripts/overnight.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/final}
DEV=${DEV:-runs/final-dev}
PY=.venv/bin/python
HOURS=${HOURS:-9}
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
mkdir -p "$OUT/logs" "$OUT/raw" "$DEV/raw"
exec >> "$OUT/logs/pipeline.log" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }
in_time() { [ "$(date +%s)" -lt "$DEADLINE" ]; }

server() {  # server <args...>: restart llama-server with this configuration
  $PY -m phase1.server stop
  log "server $*"
  $PY -m phase1.server --results-dir "$OUT" "$@" || { log "SERVER START FAILED $*"; return 1; }
  pmset -g batt | head -1
}

bench() {  # bench <results-dir> <bench args...>
  local dir=$1; shift
  if ! in_time; then log "SKIP (deadline) $*"; return; fi
  log "bench $*"
  $PY -m phase1.bench "$@" --results-dir "$dir" --timeout 600 || log "BENCH FAILED $*"
}

report() {
  $PY -m phase1.summarize --results-dir "$OUT" > "$OUT/logs/summary.txt" 2>&1 || true
  $PY -m phase1.final --results-dir "$OUT" || log "REPORT FAILED"
}

log "=== pipeline start, deadline $(date -r $DEADLINE '+%F %T'), commit $(git rev-parse --short HEAD)"

# ---- 1. live capture (resumes: existing task files are skipped) -------------------
for split in test train; do
  mkdir -p "$OUT/raw/live_$split"
  cp -n results/raw/live_$split/task_*.json "$OUT/raw/live_$split/" 2>/dev/null
done
server --parallel 2 --ctx-size 24576
log "capture test 20"
$PY -m phase1.capture --split test --count 20 --max-steps 40 --results-dir "$OUT" || log "CAPTURE TEST FAILED"
log "capture train 10"
$PY -m phase1.capture --split train --count 10 --max-steps 40 --results-dir "$OUT" || log "CAPTURE TRAIN FAILED"

$PY - "$OUT" <<'EOF'
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
bench "$OUT" micro --lengths 1024,4096,8192 --outputs 64,512 --k 1,2 --repeats 2
bench "$OUT" cache --lengths 2048,8192 --repeats 2
in_time && { $PY -m phase1.check_restore --results-dir "$OUT" || log "RESTORE CHECK FAILED"; }
report

# ---- 3. macro rounds on the eval workload ---------------------------------------------
EVAL="macro --workload $OUT/workload_eval.json --num-jobs 8 --repeats 1"
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
    bench "$DEV" macro --workload "$OUT/workload_dev.json" --num-jobs 8 --repeats 1 $MAIN --k 1,2 --jobs-active 0,2
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
