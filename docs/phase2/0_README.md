# Phase 2: start here

CSE 291A Option 1, Phase 2: memory-aware scheduling for agent serving on a 2-slot
llama.cpp server. This page maps every Phase 2 file. Read the docs in number order.

**Status (2026-11-24).**
- **Main M4 campaign:** complete.
- **Frontier supplement:** complete.
- **Arrival-rate / workload-mix supplement:** complete (48 runs); all five predictions
  hold.
- **Second-platform (T4) campaign:** not started.
- What is left, and who does it: [7_TODO.md](7_TODO.md).

## The question, the system, the answer so far

**The question.**
- Agent tasks alternate between model calls and tool waits.
- While a task waits, its slot holds its KV history; the task needs it back when it
  returns.
- With only two slots, the scheduler must decide when to admit a new task, whether to keep
  a waiting task's slot, and whether to save that slot to SSD or recompute it later.

**The system** (`phase2/`) is an event-driven scheduler outside llama-server. It owns slot
assignment through `id_slot`. Its levers are:
- an admission cap J;
- a continuation-first queue with aging;
- a cost lease that keeps a slot while `P(owner returns soon) × recompute cost` exceeds the
  queue's blocked time;
- optional KV checkpoints to SSD.

**The answer so far (M4).**
- **Pre-registered result: negative.** The frozen final policy `joint` beats FIFO (1.33×
  throughput), but ties the re-tuned simple baselines:
  - JCT ratio 1.05 [0.96, 1.17] vs best-fixed;
  - JCT ratio 1.11 [0.97, 1.33] vs affinity + J=2.
- **Not wasting prefill improves efficiency and fairness together.**
- **Under contention a fairness–efficiency trade-off appears.** On three new pre-registered
  heavy seeds:
  - checkpointed sharing (`joint-checkpoint`) gives up about 5% JCT (not significant);
  - in return its P95 slowdown is 37–52% lower than admission-capped run-to-completion
    (B2F);
  - B2F keeps the lower absolute tail delay;
  - sharing without checkpoints is dominated.
- **The benefit of scheduling follows queueing.** Across arrival rates and job mixes
  (pre-registered, 48 runs):
  - at half rate or with a short-heavy mix on light seeds, FIFO ties the best policy
    (fifo / B2F JCT 0.95–1.11);
  - at double rate or with a long-heavy mix it is 2.8–6.4× slower;
  - the jc vs B2F trade-off reappears at double rate;
  - `joint`'s deficit vs B2F grows with load.

Details: [4_FAIRNESS_EFFICIENCY.md](4_FAIRNESS_EFFICIENCY.md).

## Docs (`docs/phase2/`)

| File | What it is |
| --- | --- |
| `0_README.md` | This map |
| [1_PLAN.md](1_PLAN.md) | Design and evaluation plan, with the design history to revision 2 (section 13) |
| [2_RUNBOOK.md](2_RUNBOOK.md) | How to run probes, dev, freeze, eval and sensitivity; the overnight campaign |
| [3_PILOT_REVIEW_2026-11-18.md](3_PILOT_REVIEW_2026-11-18.md) | Review of the pilot that led to revision 2 |
| [4_FAIRNESS_EFFICIENCY.md](4_FAIRNESS_EFFICIENCY.md) | Analysis of every completed run: where fairness and efficiency trade off |
| [5_FRONTIER_PREREG.md](5_FRONTIER_PREREG.md) | Pre-registration of the frontier supplement (3 new heavy seeds) |
| [6_RATE_MIX_PREREG.md](6_RATE_MIX_PREREG.md) | Pre-registration of the arrival-rate and workload-mix supplement |
| [7_TODO.md](7_TODO.md) | Handoff: the second server, what this Mac still lacks, expected conclusions, the cross-platform method |

## Code

The Python package and the campaign launchers keep their names. Python cannot import
modules whose names start with a digit. The campaign lock
(`runs/phase2/2_m4_campaign/campaign/lock.json`) also pins the sha256 of `phase1/*.py`,
`phase2/*.py`, `scripts/phase2.sh`, `scripts/phase2_night.sh` and `requirements-lock.txt`.
**Do not edit those files.** A change there invalidates the lock that every driver checks
before running.

| File | Purpose |
| --- | --- |
| `phase2/state.py`, `phase2/scheduler.py` | Policy configs, task/slot state, dispatch, leases, aging |
| `phase2/costs.py`, `phase2/checkpoints.py` | Cost model (prefill/decode/save/restore, wait distribution), checkpoint store and quota |
| `phase2/bench.py`, `phase2/server.py`, `phase2/probes.py` | Replay benchmark, backend launcher, mechanism probes and calibration |
| `phase2/plan.py` | Manifests: run lists with pinned input hashes; dry-run unless `--execute` |
| `phase2/freeze.py` | Rules (as code) that turn the dev summary into the frozen eval config |
| `phase2/campaign.py` | Resumable overnight campaign: lock, preflight, stages, watchdog |
| `phase2/summarize.py`, `phase2/plot.py` | Summaries, paired ratios with seed-cluster bootstrap, fairness, memory, figures |
| `tests/test_phase2_*.py` | 273 model-free tests: `.venv/bin/python -m pytest -q tests` |

## Scripts

| Stage | Script | What it does |
| --- | --- | --- |
| 2 | `scripts/phase2_night.sh` (locked); wrapper `scripts/phase2/2_campaign.sh` | The main campaign: `init`, `start --until`, `status`, `stop`, `approve` |
| 2 | `scripts/phase2.sh` (locked) | Manual `create / show / run / summarize / plot / freeze` |
| 3 | `scripts/phase2/3_frontier_seeds.py` | A priori load screen that picked the frontier seeds |
| 3 | `scripts/phase2/3_frontier_run.sh` | Frontier supplement driver |
| 3 | `scripts/phase2/3_frontier_check.py` | Prints the frontier tables and the P1–P5 verdicts |
| 4 | `scripts/phase2/4_rate_mix_run.sh` | Arrival-rate / mix supplement driver (`--until HH:MM`; `touch runs/phase2/4_m4_rate_mix/STOP` to stop; finished runs are skipped) |
| 4 | `scripts/phase2/4_rate_mix_check.py` | Prints the rate/mix tables and the R1–R5 verdicts |

## Results (`runs/phase2/`)

| Folder | What | Key files |
| --- | --- | --- |
| `1_m4_pilot/` | Pilot, superseded by revision 2; exploratory evidence only. `invalid/` holds retired runs, each with a README | `dev/summary/REPORT.md`, `campaign/` |
| `2_m4_campaign/` | Main campaign (revision 2): probes, profiles, waits, dev 57, eval 43, sensitivity 16 | `eval/summary/REPORT.md`, `sensitivity/summary/REPORT.md`, `dev/summary/`, `manifests/frozen.json`, `campaign/lock.json` |
| `3_m4_frontier/` | Frontier supplement: 9 runs on seeds 3116, 3129, 3120 | `eval/summary/REPORT.md`, `frontier.log` |
| `4_m4_rate_mix/` | Arrival-rate / mix supplement: 48 runs on seeds 1101–1103 | `sensitivity/summary/REPORT.md`, `sensitivity/figures/`, `manifests/`, `workloads/`, `rate_mix.log` |

**Inside each run folder** (e.g. `2_m4_campaign/eval/eval/main/<policy>/S<seed>-T<trial>/raw/`):
- `macro_runs.jsonl`: one terminal record per run;
- `macro_jobs.jsonl` and `macro_requests.jsonl`: task and request records;
- `policy_decisions.jsonl`: every lease, evict, save and restore decision;
- `resources.jsonl`: memory samples;
- `runs_meta.jsonl`: command, versions, power state.

**Summaries** (`summary/`):
- `run_summary.csv` has one row per run;
- `jobs.csv` has one row per task, with `slowdown` and `isolated_s`;
- `paired_*.csv` hold the paired ratios.

**Path notes.**
- Old manifests and metadata contain absolute paths from the M4 (`/Users/...`). They are
  provenance only. Every result is also identified by its sha256 and parameter hash.
- On the M4, the old folder names (`runs/phase2-m4-air-16g*`) are local symlinks, ignored by
  git.
- The default `--results-dir` of `phase2.server` and `phase2.bench` is `runs/phase2`, which is
  now the parent folder. Always pass `--results-dir`.

## Fixed setup of every M4 run

| Setting | Value |
| --- | --- |
| Model | Qwen3-4B Q4_K_M (sha256 in `runs/final/model.sha256`) |
| Runtime | llama.cpp build 11669 / commit `620d2225c` |
| KV | f16, P = 2 slots, ctx 24576 (12288 per slot), no host cache |
| Workloads | `runs/final/workload_dev.json` (10 tasks, dev); `runs/final/workload_eval.json` (20 tasks, eval) |
| Timing | Arrival `scaled:1` on the M4; tool gaps `lognormal:2:1` unless a condition changes them |
| Cost profile | `runs/phase2/2_m4_campaign/profiles/profile.json` |
| Frozen policies | `runs/phase2/2_m4_campaign/manifests/frozen.json` (final = `joint`; short cutoff 782.5 output tokens) |
