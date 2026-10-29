# Memory-Aware Agent Edge Serving — Phase 1

CSE 291A (Fall 2026), Option 1. This project asks how a single memory-constrained edge
machine should serve several multi-step LLM agents at once. Phase 1 measures fixed-concurrency
FIFO baselines on a real agent benchmark and identifies the dominant bottleneck.

- **Report:** [`runs/final/PHASE1_REPORT.md`](runs/final/PHASE1_REPORT.md), with figures in
  `runs/final/figures/` (PNG and PDF).
- **Method:** [`docs/DESIGN.md`](docs/DESIGN.md).
- **Course material:** `Requirements/`, kept as received.

**Main finding.** At a fixed KV budget, the cost FIFO fails to control is the agent history
that gets evicted and then has to be prefilled again.
- Capping active tasks at the number of slots (P2 K2 J2) nearly doubles throughput on long
  tasks, but it makes short tasks wait.
- llama.cpp's slot placement is not agent-aware. With one request in flight it reuses a single
  slot and leaves the other empty.
- The built-in host prompt cache does not help.
- Saving a slot to SSD takes under 1 s, while re-prefilling 8K tokens takes 44 s.

## Layout

| Path | Contents |
|---|---|
| `phase1/core.py` | FIFO dispatcher, HTTP client, JSONL logging, append-only token helpers |
| `phase1/server.py` | Starts and stops a pinned `llama-server` (P, KV budget, unified KV, host cache) |
| `phase1/capture.py` | Live τ²-bench Retail runs: local Qwen agent, DeepSeek user, environment evaluator |
| `phase1/adapters.py` | Strict tool-call JSON normalization; never invents a tool or an argument |
| `phase1/bench.py` | `micro`, `cache` and `macro` (multi-step replay) suites with resource sampling |
| `phase1/check_restore.py` | Checks that greedy output is identical after slot save → overwrite → restore |
| `phase1/sweep.py` | Restarts the server for each server-side configuration and runs one bench suite |
| `phase1/summarize.py` | Per-configuration tables: completion, tail latency, wasted prefill, fairness |
| `phase1/startup.py` | Model load, listen and first-request time per server launch, parsed from server logs |
| `phase1/noise.py` | Run-to-run noise: spread of repeated trials on identical inputs (`--trial`) |
| `phase1/final.py` | Figures F1–F11 and the English report |
| `phase1/package.py` | Handoff archive without credentials or model weights |
| `scripts/overnight.sh` | The full, resumable Phase 1 pipeline |
| `runs/final/` | Eval results: `raw/` logs, `summary/` CSVs, `figures/`, report, workloads, manifest |
| `runs/final-dev/` | Dev-workload runs used to choose baselines |
| `tests/` | Unit and end-to-end tests (mock server) |

## Setup

Run from the project root. The model file is about 2.3 GiB.

```bash
uv venv --python 3.13 .venv
git clone --depth 1 --branch v1.0.1 https://github.com/sierra-research/tau2-bench third_party/tau2-bench
uv pip install --python .venv/bin/python -r requirements-lock.txt
mkdir -p models
curl -fL --retry 3 -o models/Qwen3-4B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf
(cd models && shasum -a 256 -c ../runs/final/model.sha256)
.venv/bin/python -m pytest tests -q
```

`llama-server` must be build 11669 / commit `620d2225c`. Other versions may change the
server flags and cache behavior.

**DeepSeek key.** The key is needed only for live capture. It is read from
`DEEPSEEK_API_KEY` or from a one-line `API.md`, which is git-ignored. Never put it in
command-line arguments, configs, logs or reports.

## Reproduce

The full pipeline runs live capture → micro → cache probe → restore check → macro rounds → dev
sweep → report:

```bash
HOURS=9 nohup caffeinate -ims bash scripts/overnight.sh > /dev/null 2>&1 &
```

- It is resumable: captured tasks, finished suites and logged macro runs are skipped.
- Progress is logged to `runs/final/logs/pipeline.log`.
- Run it on AC power with other apps closed (including VMs such as Colima/Docker). `caffeinate -ims`
  keeps the Mac awake but lets the display sleep, so the window server does not compete for the GPU.
- `TRIALS=3` runs every macro experiment three times on identical inputs (trial 1 keeps the plain
  label, so existing results count as trial 1). The report averages trials per repeat and shows
  their spread as run-to-run noise. On the M4 this adds trials 2 and 3 to `runs/final` (~19 h):
  `TRIALS=3 WORKLOADS=runs/final HOURS=10 nohup caffeinate -ims bash scripts/overnight.sh > /dev/null 2>&1 &`
- After round 1 it re-runs FIFO P2 K2 J∞ and P2 K2 J2 on R0 three times (`NOISE_TRIALS`,
  `NOISE_REPEAT`) into `runs/final-noise`. Identical inputs, so the spread is run-to-run noise.
  `SCOPE=noise` runs only these trials (~1 h on the M4) and never captures:

```bash
SCOPE=noise WORKLOADS=runs/final HOURS=2 nohup caffeinate -ims bash scripts/overnight.sh > /dev/null 2>&1 &
.venv/bin/python -m phase1.noise --results-dir runs/final-noise   # also run by the pipeline
```

Individual steps:

```bash
.venv/bin/python -m phase1.server --parallel 2 --ctx-size 24576 --results-dir runs/final
.venv/bin/python -m phase1.bench macro --workload runs/final/workload_eval.json --num-jobs 8 \
  --k 2 --jobs-active 0,2 --gap lognormal:2:1 --arrival poisson:1 --results-dir runs/final --skip-done
.venv/bin/python -m phase1.server stop
.venv/bin/python -m phase1.summarize --results-dir runs/final
.venv/bin/python -m phase1.startup --results-dir runs/final    # also run by phase1.final
.venv/bin/python -m phase1.final --results-dir runs/final
```

New experiments should write to their own directory (`--results-dir runs/<name>`; the default
is `runs/scratch`). Logs are append-only, so never append a second full rerun to an existing
directory and count it as extra repeats.

### Main options

- **`phase1.server`:**
  - `--parallel P`
  - `--ctx-size` (total KV tokens; each slot gets ctx/P unless `--kv-unified`)
  - `--cache-ram MiB` (llama.cpp host prompt cache)
  - `stop` signals only the recorded process, and only if it is `llama-server`.
- **`phase1.bench macro`:**
  - `--k` (request concurrency)
  - `--jobs-active` (task admission limit J; 0 = unlimited)
  - `--gap` (`recorded | zero | fixed:S | exp:MEAN | lognormal:MEDIAN:SIGMA`)
  - `--arrival` (`batch | poisson:TASKS_PER_MIN | scaled:TASKS_PER_MIN`); `scaled` compresses the
    repeat's `poisson:1` pattern, so a rate sweep changes only the rate (`poisson:R` draws a new pattern)
  - `--trial N` re-runs a repeat on identical inputs under the label suffix `-T<N>`
  - `--num-jobs`, `--repeats`, `--repeat-start`, `--skip-done`
  - Each repeat index fixes the task mix, arrivals and delays, so configurations stay paired.
- **`phase1.summarize`:** slowdown needs a `--k 1 --jobs-active 1` run in the same directory,
  which serves as the isolated-compute reference.
- **`phase1.curves`:** per-run contention (mean active tasks, time above the slot count, hit rate),
  used to calibrate the arrival rate on a new machine.

## Moving to AWS (CPU)

Keep the same model SHA-256, workload files, runtime commit, KV precision and code. The
pipeline reuses the frozen workloads in `runs/final` (`WORKLOADS=runs/final`), so replays never
call DeepSeek and no key goes to the instance.

**1. Copy the repo to the instance** (from the laptop). Include `.git` so every run records its
commit; never copy credentials:

```bash
rsync -az --exclude .venv --exclude models --exclude third_party --exclude API.md --exclude '.env*' \
  ./ ubuntu@<host>:memory-aware-agent-edge-serving/
```

**2. Build `llama-server` and set up** (on the instance, Ubuntu 24.04), then follow
[Setup](#setup); on Linux check the model with `sha256sum -c` instead of `shasum -a 256 -c`:

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake git curl libcurl4-openssl-dev tmux
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
cd ~/memory-aware-agent-edge-serving
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp && git -C third_party/llama.cpp checkout 620d2225c
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/llama.cpp/build -j --target llama-server
```

**3. Pilot, then the full sweep**, inside tmux (`tmux new -s sweep`; detach with `Ctrl-b d`):

```bash
export PATH="$PWD/third_party/llama.cpp/build/bin:$PATH"
# Pilot (~1 h, own directory): sets TIMEOUT and threads, and estimates the full run.
WORKLOADS=runs/final OUT=runs/aws-pilot DEV=runs/aws-pilot-dev SERVER_ARGS="--cpu --threads 8" \
  TIMEOUT=1800 HOURS=1 bash scripts/overnight.sh
# Full sweep. Resumable: experiments skipped at the deadline run on the next invocation.
WORKLOADS=runs/final OUT=runs/aws DEV=runs/aws-dev SERVER_ARGS="--cpu --threads 8" \
  TIMEOUT=1800 HOURS=9 STOP_WHEN_DONE=1 bash scripts/overnight.sh
tail -f runs/aws/logs/pipeline.log
```

**Single experiment** (one configuration, e.g. a quick check before the sweep). These options
reproduce the sweep's FIFO baseline on repeat R0 (label
`macro-P2-C24576-K2-Jall-lognormal:2:1-poisson:1-R0`); keep ad-hoc runs in their own directory:

```bash
export PATH="$PWD/third_party/llama.cpp/build/bin:$PATH"
.venv/bin/python -m phase1.server --cpu --threads 8 --parallel 2 --ctx-size 24576 --results-dir runs/aws-single
.venv/bin/python -m phase1.bench macro --workload runs/final/workload_eval.json --results-dir runs/aws-single \
  --num-jobs 8 --repeat-start 0 --repeats 1 --k 2 --jobs-active 0 \
  --gap lognormal:2:1 --arrival poisson:1 --timeout 1800 --skip-done
.venv/bin/python -m phase1.server stop
.venv/bin/python -m phase1.summarize --results-dir runs/aws-single
```

- Pick another configuration with the bench's `--k`, `--jobs-active` (0 = no limit), `--repeat-start`,
  `--gap`, `--arrival`, and the server's `--parallel`, `--ctx-size`, `--kv-unified`, `--cache-ram`.
- Every value in a comma list adds a run: `--k 1,2 --jobs-active 0,2` is 4 runs.
- Leave out `--num-jobs 8 --repeats 1 --k 2` and the defaults run 6 runs of all 20 tasks
  (K 1 and 2 × 3 repeats), which do not match the sweep's 8-task runs.
- Slowdown and fairness need a `--k 1 --jobs-active 1` run in the same directory.

**4. Copy results back** (from the laptop) and commit them there:
`rsync -az ubuntu@<host>:memory-aware-agent-edge-serving/runs/aws/ runs/aws/` (same for `runs/aws-dev/`).

- `--threads` = physical cores: on Graviton every vCPU is a core; on Intel/AMD it is half the
  vCPUs (check `lscpu`).
- Budget: (HOURS + 0.5) × the hourly price must stay under the daily limit. `STOP_WHEN_DONE=1`
  powers the instance off when the pipeline ends, and at the latest 30 min after the deadline;
  a stopped instance still pays for its disk.
- Calibrate arrival rates to the instance after the pilot: a k× slower machine at the same
  arrival rate runs a more heavily loaded experiment.
- Compare configurations within one platform; never mix Mac and AWS absolute speeds.

## AWS GPU (g4dn.xlarge): 3 trials of every experiment

A T4 (16 GB) is the cheapest course-listed GPU that fits the model plus a 24K–36K KV budget
(~6.3–7.7 GB); `--ctx-size` still enforces the edge-sized budget. At ~$0.53/hr the whole sweep runs
in one session under a $20 daily limit. It is a new baseline: run Phase 2 on the same instance type.

**1. Launch** (console, us-west-2): g4dn.xlarge, AMI "Deep Learning Base OSS Nvidia Driver GPU AMI
(Ubuntu 24.04)" (AWS-provided, no Marketplace fees), 100 GB gp3 root volume, your key pair, a
security group allowing SSH from your IP and all outbound traffic. If the launch fails on a vCPU
limit, the account needs quota for "Running On-Demand G and VT instances" (≥ 4). Keep results on the
root (EBS) volume: the instance-store NVMe disk is wiped whenever the instance stops.

**2. Copy the repo and set up** (copy as in the CPU section, user `ubuntu`), then on the instance:

```bash
# No package updates mid-sweep: keeps CPU/disk quiet and the NVIDIA driver fixed between trials.
sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer
sudo apt-get update && sudo apt-get install -y build-essential cmake git curl tmux
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
cd ~/memory-aware-agent-edge-serving    # then follow Setup (sha256sum -c instead of shasum)
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp && git -C third_party/llama.cpp checkout 620d2225c
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build third_party/llama.cpp/build -j --target llama-server llama-bench
nvidia-smi && .venv/bin/python -m pytest tests -q
```

**3. Pilot: calibrate the arrival rate** (~1 h, own directory). A GPU serves steps several times
faster than the M4, so at 1 task/min tasks barely overlap and the eviction effect disappears. Tool
delays stay in real seconds; only the arrival rate changes. `scaled:R` compresses each repeat's
1/min pattern, so R0/R1/R2 keep their relative burstiness.

```bash
export PATH="$PWD/third_party/llama.cpp/build/bin:$PATH"
mkdir -p runs/aws-gpu-pilot
llama-bench -m models/Qwen3-4B-Q4_K_M.gguf -p 1024,4096 -n 64 -fa 1 -r 2 | tee runs/aws-gpu-pilot/llama-bench.txt
.venv/bin/python -m phase1.server --parallel 2 --ctx-size 24576 --results-dir runs/aws-gpu-pilot
for rate in 1 2 4; do
  .venv/bin/python -m phase1.bench macro --workload runs/final/workload_eval.json --results-dir runs/aws-gpu-pilot \
    --num-jobs 8 --repeat-start 0 --repeats 1 --k 2 --jobs-active 0 --gap lognormal:2:1 --arrival scaled:$rate --skip-done
done
.venv/bin/python -m phase1.curves --results-dir runs/aws-gpu-pilot
```

Pick the rate at which FIFO on R0 averages ~2.4 active tasks (the M4's R0: 2.4 active, 41% of the
time above 2 slots, 69% full-history hits), then check R2 (FIFO and J2) at that rate: the M4's R2 had
4.4 active, 87%, 7%, and J2 ~2× FIFO. Stop the server before the sweep.

**4. Full sweep** in tmux (`tmux new -s sweep`), ~10–20 h for 114 macro runs:

```bash
TRIALS=3 ARRIVAL_KIND=scaled ARRIVAL_RATE=<rate> WORKLOADS=runs/final OUT=runs/aws-gpu DEV=runs/aws-gpu-dev \
  HOURS=22 STOP_WHEN_DONE=1 STOP_DELAY_MIN=20 bash scripts/overnight.sh
tail -f runs/aws-gpu/logs/pipeline.log       # GPU clocks/power/temperature: runs/aws-gpu/logs/machine.log
```

Resumable: after a stop or crash, the same command re-runs only what is missing. Each run records
the GPU UUID, so a restart onto a different card shows in `raw/runs_meta.jsonl` and the report.

**5. Copy results back** (from the laptop, during the run and at the end, before the power-off):

```bash
for d in aws-gpu aws-gpu-dev aws-gpu-pilot; do
  rsync -az -e "ssh -i <key.pem>" ubuntu@<host>:memory-aware-agent-edge-serving/runs/$d/ runs/$d/
done
```

**6. Clean up:** terminate the instance once the copies are verified, then check EC2 → Volumes,
Snapshots and Elastic IPs for leftovers (a volume, even detached, bills until deleted).
