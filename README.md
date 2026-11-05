# Memory-Aware Agent Edge Serving: Phase 1

CSE 291A (Fall 2026), Option 1. Several multi-step LLM agents share one inference server with a
fixed KV-cache budget. Phase 1 implements a FIFO baseline with fixed request concurrency and an
optional cap on active tasks, drives it with replays of τ²-bench Retail trajectories, and measures
which effect dominates: idle time from admitting too few agents, or lost histories from admitting
too many.

**Report:** [`report/phase1.pdf`](report/phase1.pdf) (source [`report/phase1.tex`](report/phase1.tex)).
The main text is two pages; the appendix holds the full tables, figures and artifact map.

## Main results

| | MacBook Air M4 (Metal) | AWS g4dn.xlarge, T4 (CUDA) |
|---|---|---|
| Re-prefill of an 8K history | 44 s | 5.1 s |
| Save / restore one 8K slot | 0.37 s / 0.35 s | 8.95 s / 0.47 s |
| Full-history hit, 2 slots, 1 / 2 / 3 / 4 active tasks | 100 / 83 / 11 / 0 % | 98 / 81 / 37 / 15 % |
| Cap J = 2 vs FIFO, per repeat (R0 / R1 / R2) | 1.03 / 0.97 / **1.97×** | 0.91 / 1.00 / **0.79×** |
| Best fixed cap on dev tasks | J = 2 (2.02× FIFO) | J = 4 (1.04× FIFO) |
| Run-to-run noise (same inputs) | 11 % FIFO, 2 % J = 2 (M5, 3 trials) | ≤ 1.3 % (3 trials of every run) |

- At a fixed budget, the cost FIFO fails to control is the lost history: another agent overwrites a
  waiting agent's slot, and the returning agent re-prefills its history.
- Whether capping active tasks pays off depends on how expensive a lost history is compared with
  tool time. The cap nearly doubles throughput on the M4 and loses on the T4.
- Caps are unfair to short tasks (slowdown 9.5 vs 1.6 for long tasks on the M4).
- llama.cpp's slot placement is not agent-aware: with one request in flight, every request goes to
  the same slot.
- llama.cpp's host prompt cache (`--cache-ram`) does not help.

## Verify the reported numbers (no model, GPU or API key)

Every table and figure is generated from the committed raw logs. With Python 3.13 and
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Reed2006/memory-aware-agent-edge-serving && cd memory-aware-agent-edge-serving
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r <(grep -v tau2-bench requirements-lock.txt)
.venv/bin/python -m pytest tests -q   # 39 unit and end-to-end tests against a mock server
bash scripts/verify.sh                # regenerates both generated reports and diffs them
```

`scripts/verify.sh` copies `runs/` to a temporary directory, reruns `phase1.final` on the M4 and T4
results, and checks that the regenerated reports match the committed ones. Expected output:

```
OK    runs/final/PHASE1_REPORT.md regenerates identically from raw/
OK    runs/aws-gpu/PHASE1_REPORT.md regenerates identically from raw/
```

To rebuild the PDF: `cd report && latexmk -pdf phase1.tex`.

## Repository layout

| Path | Contents |
|---|---|
| `report/` | Phase 1 report (LaTeX, PDF, figures) |
| `phase1/core.py` | FIFO queue with concurrency K, HTTP client, JSONL logging, append-only replay |
| `phase1/server.py` | Starts and stops a pinned `llama-server` (P slots, KV budget, unified KV, host cache) |
| `phase1/capture.py`, `phase1/adapters.py` | Live τ²-bench Retail capture (local Qwen agent, DeepSeek user); strict tool-call normalization |
| `phase1/bench.py` | `micro`, `cache` and `macro` suites; active-task cap J; trials; resource sampling |
| `phase1/check_restore.py` | Checks that greedy output is identical after slot save → overwrite → restore |
| `phase1/sweep.py` | Restarts the server per server-side configuration and runs one suite |
| `phase1/summarize.py`, `noise.py`, `curves.py`, `startup.py` | Per-configuration metrics, run-to-run noise, contention curves, cold start |
| `phase1/final.py` | Figures and the generated report of one results directory |
| `scripts/overnight.sh` | Full resumable pipeline (capture → micro → cache → restore check → macro → dev → report) |
| `scripts/verify.sh` | Regenerates and diffs the generated reports |
| `docs/DESIGN.md` | Definitions, metrics, experiment matrix and validity rules |
| `REPORT.md` | Detailed write-up of the T4 sweep |
| `tests/` | Unit and end-to-end tests |

| Results | Contents |
|---|---|
| `runs/final/` | M4 eval results: frozen workloads, live capture, `raw/` logs, `summary/`, `figures/`, generated report |
| `runs/final-dev/` | M4 dev runs used to choose baselines |
| `runs/aws-gpu/`, `runs/aws-gpu-dev/` | T4 eval and dev sweeps, three trials of every run |
| `runs/aws-gpu-pilot/` | T4 arrival-rate calibration pilot and the scripts that ran setup and sweep |
| `runs/m5-noise/`, `runs/m5/` | Run-to-run noise trials on a second Mac (M5) and their pipeline log |
| `runs/aws-single/` | One exploratory FIFO run on a CPU instance (c7g.2xlarge); overloaded, not used in the report |

Each run directory has `raw/` (append-only request, task, resource and run-metadata logs),
`summary/` (CSV tables), `figures/` (PNG and PDF) and `PHASE1_REPORT.md`. Every run records its git
commit, server command, power state and, on the T4, GPU identity in `raw/runs_meta.jsonl`.

## Pinned versions

| Item | Version |
|---|---|
| Runtime | llama.cpp `llama-server` build 11669, commit `620d2225c` |
| Model | Qwen3-4B Q4_K_M, revision `bc640142` (SHA-256 in `runs/final/model.sha256`), f16 KV, thinking disabled |
| Benchmark | τ²-bench v1.0.1 Retail; 20 eval and 10 dev tasks sampled with seed 291 |
| Python | 3.13, dependencies in `requirements-lock.txt` |

## Rerun the experiments

Full setup (model about 2.3 GiB):

```bash
git clone --depth 1 --branch v1.0.1 https://github.com/sierra-research/tau2-bench third_party/tau2-bench
uv pip install --python .venv/bin/python -r requirements-lock.txt
mkdir -p models
curl -fL --retry 3 -o models/Qwen3-4B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf
(cd models && shasum -a 256 -c ../runs/final/model.sha256)   # sha256sum -c on Linux
```

Build `llama-server` at commit `620d2225c` and put it on `PATH`. The replays reuse the frozen
workloads in `runs/final`, so they never call the user simulator and need no API key. Only a new
live capture needs a DeepSeek key, read from `DEEPSEEK_API_KEY` or a one-line, git-ignored `API.md`.

**M4 configuration** (into a new directory; the committed `runs/final` would be skipped as done):

```bash
WORKLOADS=runs/final OUT=runs/m4-rerun DEV=runs/m4-rerun-dev HOURS=9 \
  nohup caffeinate -ims bash scripts/overnight.sh > /dev/null 2>&1 &
```

**T4 configuration** on an EC2 g4dn.xlarge with a CUDA build of `llama-server` (the setup, pilot
and sweep scripts that were used are in `runs/aws-gpu-pilot/scripts/`):

```bash
TRIALS=3 ARRIVAL_KIND=scaled ARRIVAL_RATE=3.5 EXTRA_J="3 4" WORKLOADS=runs/final \
  OUT=runs/t4-rerun DEV=runs/t4-rerun-dev HOURS=22 STOP_WHEN_DONE=1 bash scripts/overnight.sh
```

The pipeline is resumable: finished suites and logged runs are skipped, and an interrupted run is
moved to `raw/interrupted/` before it is rerun. Progress goes to `<OUT>/logs/pipeline.log`. Write
new experiments to their own directory (`--results-dir runs/<name>`).

**One configuration by hand** (FIFO P2 K2 on repeat R0):

```bash
.venv/bin/python -m phase1.server --parallel 2 --ctx-size 24576 --results-dir runs/scratch
.venv/bin/python -m phase1.bench macro --workload runs/final/workload_eval.json --results-dir runs/scratch \
  --num-jobs 8 --repeat-start 0 --repeats 1 --k 2 --jobs-active 0 --gap lognormal:2:1 --arrival poisson:1
.venv/bin/python -m phase1.server stop
.venv/bin/python -m phase1.summarize --results-dir runs/scratch
```

Main options:
- `phase1.server`: `--parallel P`, `--ctx-size` (total KV tokens, split across slots unless
  `--kv-unified`), `--cache-ram MiB`, `--cpu --threads N`.
- `phase1.bench macro`: `--k` (request concurrency), `--jobs-active` (cap J; 0 = none),
  `--gap` (`recorded | zero | fixed:S | exp:MEAN | lognormal:MEDIAN:SIGMA`),
  `--arrival` (`batch | poisson:RATE | scaled:RATE`), `--repeat-start`, `--repeats`,
  `--trial N` (rerun a repeat on identical inputs), `--skip-done`.
- A repeat index fixes the task mix, arrivals and tool delays, so configurations are compared on
  the same inputs. Slowdown needs a `--k 1 --jobs-active 1` run in the same directory.
- Calibrate the arrival rate on a new machine (`phase1.curves`); compare configurations only within
  one platform.
