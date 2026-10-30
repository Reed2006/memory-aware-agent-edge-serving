#!/bin/bash
# Calibration pilot: speed benchmark, then FIFO P2 K2 on R0 at several scaled arrival rates.
cd ~/memory-aware-agent-edge-serving || exit 1
export PATH="$PWD/third_party/llama.cpp/build/bin:$PATH"
OUT=runs/aws-gpu-pilot
mkdir -p $OUT/logs
( while true; do echo "[$(date "+%F %T")] $(nvidia-smi --query-gpu=uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader)"; sleep 60; done ) >> $OUT/logs/machine.log 2>&1 &
LOGGER=$!; trap "kill $LOGGER 2>/dev/null" EXIT
echo "=== [$(date "+%T")] llama-bench"
llama-bench -m models/Qwen3-4B-Q4_K_M.gguf -p 1024,4096 -n 64 -fa 1 -r 2 2>&1 | grep -E "^\||build" | tee $OUT/logs/llama-bench.txt
echo "=== [$(date "+%T")] server"
.venv/bin/python -m phase1.server --parallel 2 --ctx-size 24576 --results-dir $OUT || exit 1
for rate in ${RATES:-1 2 4 6}; do
  echo "=== [$(date "+%T")] FIFO R${REPEAT:-0} scaled:$rate"
  .venv/bin/python -m phase1.bench macro --workload runs/final/workload_eval.json --results-dir $OUT --num-jobs 8 \
    --repeat-start ${REPEAT:-0} --repeats 1 --k 2 --jobs-active ${JOBS:-0} --gap lognormal:2:1 --arrival scaled:$rate \
    --timeout 600 --skip-done 2>&1 | grep -E "jobs_per_minute|skip|Error|error" | cut -c1-400
done
.venv/bin/python -m phase1.server stop
echo "=== [$(date "+%T")] contention"
.venv/bin/python -m phase1.curves --results-dir $OUT
