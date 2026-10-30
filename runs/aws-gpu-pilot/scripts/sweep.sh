#!/bin/bash
# The GPU sweep: 3 trials of every experiment at the calibrated rate, plus J3/J4 (see README "AWS GPU").
cd ~/memory-aware-agent-edge-serving || exit 1
export PATH="$PWD/third_party/llama.cpp/build/bin:$PATH"
TRIALS=3 ARRIVAL_KIND=scaled ARRIVAL_RATE=3.5 EXTRA_J="3 4" WORKLOADS=runs/final OUT=runs/aws-gpu DEV=runs/aws-gpu-dev \
  HOURS=22 STOP_WHEN_DONE=1 STOP_DELAY_MIN=20 bash scripts/overnight.sh
echo "SWEEP EXIT $?" >> runs/aws-gpu/logs/pipeline.log
