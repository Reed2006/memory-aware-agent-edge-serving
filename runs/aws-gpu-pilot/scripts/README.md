# Scripts run on the g4dn.xlarge (home directory), in order

1. `setup.sh`: environment, model (sha256 checked), llama.cpp 620d2225c built with CUDA for sm_75, tests.
2. `pilot.sh`: `llama-bench`, then FIFO P2 K2 on R0 at scaled:1, 2, 4, 6 (`logs/llama-bench.txt`, runs in `raw/`).
3. `validate.sh`: at scaled:3.5, FIFO on R0, R2, R1 and J2 on R2; `python -m phase1.curves` compares contention.
4. `sweep.sh`: the sweep in `runs/aws-gpu` and `runs/aws-gpu-dev` (first launched without `EXTRA_J`).
5. `restart.sh`: stopped that first launch at its first macro run and relaunched `sweep.sh` with `EXTRA_J="3 4"`;
   micro, cache and restore results were kept (see the "restarted" line in `runs/aws-gpu/logs/pipeline.log`).

Automatic package updates were disabled before setup (`systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer`).
