#!/bin/bash
# Used once: stopped the first launch (no EXTRA_J) at its first macro run and relaunched sweep.sh.
cd ~/memory-aware-agent-edge-serving
until grep -q "=== round 0 (full), trial 1 of 3" runs/aws-gpu/logs/pipeline.log; do sleep 2; done
tmux kill-session -t sweep; sleep 2
.venv/bin/python -m phase1.server stop
sleep 2
echo "=== [$(date "+%F %T")] restarted with EXTRA_J=3 4 at commit $(git rev-parse --short HEAD) (stopped at the first macro run)" >> runs/aws-gpu/logs/pipeline.log
pgrep -fa "phase1.bench|llama-server|overnight.sh" || echo "no leftover processes"
tmux new -d -s sweep "bash ~/sweep.sh"
