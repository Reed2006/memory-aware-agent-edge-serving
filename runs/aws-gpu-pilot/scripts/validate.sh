#!/bin/bash
# Calibration check at one rate: FIFO on R0, R1, R2 and J2 on R2 (pilot directory, not counted in the sweep).
cd ~/memory-aware-agent-edge-serving || exit 1
export PATH="$PWD/third_party/llama.cpp/build/bin:$PATH"
OUT=runs/aws-gpu-pilot; RATE=${RATE:-3.5}
.venv/bin/python -m phase1.server --parallel 2 --ctx-size 24576 --results-dir $OUT || exit 1
for case in "0 0" "2 0" "2 2" "1 0"; do
  set -- $case
  echo "=== [$(date "+%T")] R$1 J$2 scaled:$RATE"
  .venv/bin/python -m phase1.bench macro --workload runs/final/workload_eval.json --results-dir $OUT --num-jobs 8 \
    --repeat-start $1 --repeats 1 --k 2 --jobs-active $2 --gap lognormal:2:1 --arrival scaled:$RATE \
    --timeout 600 --skip-done 2>&1 | grep -cE "jobs_per_minute"
done
.venv/bin/python -m phase1.server stop
.venv/bin/python -m phase1.curves --results-dir $OUT
