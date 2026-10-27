# Phase 1 Report — Memory-Aware Agent Edge Serving (Option 1)

_Generated 2026-10-26T21:23 from `runs/final` by `phase1.final`. All numbers come from the raw logs in this directory._

## 1. Question and baseline

Multi-step agents alternate model calls with tool calls and grow their context at every step. On a memory-constrained edge machine the KV cache that holds this context is the scarce resource. Phase 1 asks: **under a fixed KV memory budget, how do request concurrency, the number of server slots, the number of simultaneously active agent tasks, tool idle time and arrival rate affect completed agent work, tail latency, memory and fairness — and what is the dominant bottleneck?**

The baseline is a FIFO queue with fixed concurrency K in front of one llama-server: every ready agent step (first step or continuation after a tool) is dispatched in enqueue order; tool waits do not hold K. An optional task-level admission limit J caps how many agent tasks are active at once (waiting counts toward latency).

## 2. Setup

| Item | Value |
|---|---|
| Machine | arm64 macOS 15.3, Apple M4, 16 GB unified memory |
| Runtime | llama.cpp llama-server build 11669 (commit 620d2225c), Metal, flash-attn, continuous batching |
| Model | Qwen3-4B Q4_K_M (sha256 7485fe6f…), f16 KV, thinking disabled; fixed in all comparisons |
| KV budget | --ctx-size tokens in total; partitioned per slot unless "u" (unified pool); f16 KV = 144 KiB/token |
| Agent benchmark | τ²-bench v1.0.1 Retail (official tools, orchestrator, ENV evaluator) |
| User simulator | DeepSeek V4 flash (`deepseek-flash`), live capture only; never called during timed replays |
| Power during runs | Now drawing from 'AC Power'; Now drawing from 'Battery Power' |
| Code commit(s) | 0d53ae07, 7a489316, b51b99e1 |


## 3. Workloads (micro → macro)

**Real application (live).** 20 eval tasks (test split) and 10 dev tasks (train split), sampled with seed 291 and kept regardless of outcome, were executed end-to-end: local Qwen3-4B agent, official Retail tools and DeepSeek user simulator. Eval tasks take 2–20 agent calls (median 11); the longest prompt per task is 5,030–9,677 tokens. ENV checks pass on 3/20 eval tasks: the 4B model is weak, and quality is reported, not filtered (F1).

**Macro (benchmark-derived).** Each live trajectory is converted to an append-only replay that keeps the exact number of steps, per-step output tokens and input growth, and retains tokens actually generated during the replay. A run draws N tasks from the frozen eval set (balanced, seeded), with Poisson arrivals and controlled tool delays (default log-normal, median 2 s, σ=1, heavy-tailed). Every configuration sees identical arrivals, task order and delays for the same repeat index. The dev split is kept separate for selecting baselines.

**Micro (controlled).** Exact token counts with prompt cache disabled; K requests submitted together. **Cache probe.** Measures continuation cost when an agent's slot is retained, idle, overwritten, erased, or saved to SSD and restored.

**Metrics.** Completed tasks/min, P50/P95/P99 task completion time (arrival → last step, admission and queue included), FIFO and admission waits, failures and their reason, peak server RSS / min system available memory, prefill tokens that recompute already-seen history ("wasted prefill"), full-history hit rate, and fairness: slowdown = completion time / (isolated compute time + own tool delay), Jain index of 1/slowdown, and slowdown of long vs short tasks.

## 4. Results

### 4.1 Micro: prefill dominates long contexts (F2)

| input tokens | output | K | prefill s | round trip s | aggregate tok/s |
|---|---|---|---|---|---|
| 1,024 | 64 | 1 | 4.56 | 6.59 | 9.7 |
| 1,024 | 64 | 2 | 4.36 | 12.24 | 10.4 |
| 1,024 | 512 | 1 | 4.37 | 23.58 | 21.7 |
| 1,024 | 512 | 2 | 4.36 | 40.36 | 25.3 |
| 4,096 | 64 | 1 | 19.11 | 21.50 | 3.0 |
| 4,096 | 64 | 2 | 24.15 | 42.34 | 3.0 |
| 4,096 | 512 | 1 | 19.01 | 41.20 | 12.4 |
| 4,096 | 512 | 2 | 19.28 | 76.11 | 13.4 |
| 8,192 | 64 | 1 | 43.56 | 46.47 | 1.4 |
| 8,192 | 64 | 2 | 43.96 | 92.25 | 1.4 |
| 8,192 | 512 | 1 | 43.83 | 70.53 | 7.3 |
| 8,192 | 512 | 2 | 44.04 | 133.27 | 7.7 |


Prefill runs at roughly 184 tokens/s, so re-processing an 8K history costs 43.6 s — longer than generating several hundred tokens.

### 4.2 Cache mechanisms (F3)

| input | condition | continuation prefill s | input reused |
|---|---|---|---|
| 2,048 | erase | 9.277 | 0.0% |
| 2,048 | overwrite | 9.389 | 0.0% |
| 2,048 | retain | 0.108 | 99.5% |
| 2,048 | save_restore | 0.107 | 99.5% |
| 2,048 | wait | 0.098 | 99.5% |
| 8,192 | erase | 44.110 | 0.0% |
| 8,192 | overwrite | 44.752 | 0.0% |
| 8,192 | retain | 0.196 | 99.9% |
| 8,192 | save_restore | 0.176 | 99.9% |
| 8,192 | wait | 0.174 | 99.9% |


| input | state MiB | save s | restore s | save+erase+restore+prefill s | re-prefill s |
|---|---|---|---|---|---|
| 2,048 | 292 | 0.156 | 0.072 | 0.336 | 9.28 |
| 8,192 | 1,156 | 0.365 | 0.353 | 0.896 | 44.11 |


Restore correctness: after save → overwrite → restore, 64 greedy tokens identical to the uninterrupted run: **True**.

### 4.3 Macro: FIFO baselines under a fixed 24K-token KV budget (F4–F6)

| config | repeats | completed | tasks/min | P50 s | P95 s | wasted prefill | history hit | mean slowdown |
|---|---|---|---|---|---|---|---|---|
| P1·24K K1 J∞ | 3 | 24/24 | 0.483 ± 0.116 | 382 | 575 | 78.0% | 26.4% | 4.61 |
| P2·24K K1 J1 | 3 | 24/24 | 0.677 ± 0.197 | 174 | 280 | 0.0% | 100.0% | 7.49 |
| P2·24K K1 J∞ | 3 | 24/24 | 0.450 ± 0.051 | 412 | 638 | 78.0% | 25.1% | 5.15 |
| P2·24K K2 J2 | 3 | 24/24 | 0.736 ± 0.228 | 152 | 256 | 20.0% | 95.7% | 5.53 |
| P2·24K K2 J∞ | 3 | 24/24 | 0.587 ± 0.203 | 318 | 479 | 45.4% | 57.8% | 3.98 |
| P2·24K+RAM3G K2 J∞ | 3 | 24/24 | 0.532 ± 0.113 | 363 | 532 | 63.6% | 45.0% | 4.70 |
| P4·24K K4 J∞ | 3 | 10/24 | 0.424 ± 0.205 | 38 | 68 | 15.6% | 87.5% | 4.23 |
| P4·36K K4 J∞ | 2 | 13/16 | 0.528 ± 0.141 | 50 | 169 | 7.1% | 97.8% | 1.52 |
| P4u·24K K4 J∞ | 3 | 16/24 | 0.465 ± 0.052 | 111 | 203 | 31.5% | 77.3% | 4.42 |


Highest completed-task throughput on the eval workload: **P2·24K K2 J2**. (Phase 2 must re-select the best fixed baseline on the dev workload; this ranking is descriptive.)

Failures (25 tasks across all runs) are kept in the denominators. Reasons seen: Context size has been exceeded.; request (N tokens) exceeds the available context size (N tokens), try increasing it.

Mean time per completed task (F5):

| config | admission wait | FIFO wait | model round trip | tool delay | total |
|---|---|---|---|---|---|
| P1·24K K1 J∞ | 0 | 196 | 100 | 34 | 330 |
| P2·24K K1 J1 | 101 | 0 | 40 | 34 | 175 |
| P2·24K K1 J∞ | 0 | 226 | 106 | 34 | 367 |
| P2·24K K2 J2 | 52 | 0 | 70 | 34 | 156 |
| P2·24K K2 J∞ | 0 | 109 | 136 | 34 | 280 |
| P2·24K+RAM3G K2 J∞ | 0 | 129 | 156 | 34 | 319 |
| P4·24K K4 J∞ | 0 | 3 | 35 | 7 | 45 |
| P4·36K K4 J∞ | 0 | 0 | 45 | 31 | 75 |
| P4u·24K K4 J∞ | 0 | 9 | 77 | 29 | 115 |


### 4.4 Memory (F7)

| server | theoretical KV GiB | peak server RSS GiB | min system available GiB |
|---|---|---|---|
| P1·24K | 3.38 | 3.59 | 1.86 |
| P2·24K | 3.38 | 3.61 | 1.86 |
| P2·24K+RAM3G | 3.38 | 3.59 | 1.97 |
| P4·24K | 3.38 | 3.58 | 2.18 |
| P4·36K | 5.06 | 5.21 | 2.14 |
| P4u·24K | 3.38 | 3.61 | 2.16 |


llama-server preallocates the whole KV pool at start, so resident memory is set by the budget, not by load; "too much work" therefore shows up as slot contention (evicted histories) or as hard context overflows, not as growing RSS.

### 4.5 Idle time between steps (F10)

| tool delay | tasks/min | P50 s | P95 s | wasted prefill | history hit | slowdown |
|---|---|---|---|---|---|---|
| zero | 0.603 | 147 | 289 | 46.0% | 66.7% | 6.05 |
| lognormal:2:1 | 0.587 | 318 | 479 | 45.4% | 57.8% | 3.98 |
| fixed:15 | 0.456 | 323 | 510 | 66.0% | 43.1% | 1.76 |


### 4.6 Arrival rate (F11)

| arrival | tasks/min | P50 s | P95 s | wasted prefill | history hit | slowdown |
|---|---|---|---|---|---|---|
| poisson:0.5 | 0.467 | 81 | 188 | 40.0% | 92.1% | 1.34 |
| poisson:1 | 0.587 | 318 | 479 | 45.4% | 57.8% | 3.98 |
| poisson:2 | 0.801 | 185 | 416 | 47.1% | 55.1% | 3.77 |


### 4.7 Fairness (F8)

| config | short-task slowdown | long-task slowdown |
|---|---|---|
| P1·24K K1 J∞ | 5.01 | 4.20 |
| P2·24K K1 J1 | 13.44 | 1.53 |
| P2·24K K1 J∞ | 5.57 | 4.73 |
| P2·24K K2 J2 | 9.50 | 1.57 |
| P2·24K K2 J∞ | 4.53 | 3.43 |
| P2·24K+RAM3G K2 J∞ | 5.46 | 3.95 |
| P4·24K K4 J∞ | 4.05 | n/a |
| P4·36K K4 J∞ | 1.72 | 1.16 |
| P4u·24K K4 J∞ | 5.34 | 1.25 |


### 4.8 Dev workload (for Phase 2 baseline selection)

| P | K | J | tasks/min | runs |
|---|---|---|---|---|
| 2.0 | 1.0 | ∞ | 0.437 | 1.0 |
| 2.0 | 1.0 | 2.0 | 0.425 | 1.0 |
| 2.0 | 2.0 | ∞ | 0.453 | 1.0 |
| 2.0 | 2.0 | 2.0 | 0.914 | 1.0 |


## 5. Dominant bottleneck

Across baselines the share of prefill spent recomputing history ranges from 0.0% (P2·24K K1 J1) to 78.0% (P2·24K K1 J∞); full-history hit rates range 25%–100%. Across configurations, wasted-prefill share and completed tasks/min correlate at r = -0.43. Resident memory stays flat because the pool is preallocated, so the edge-memory limit acts through the number and size of slots: more slots per budget means smaller per-slot contexts (overflow failures for long Retail histories), fewer slots means agents evict one another's histories while their tools run.


## 6. Phase 2 direction

Phase 1 shows that, at a fixed KV budget, the cost that FIFO fails to control is **history eviction followed by re-prefill**, which grows with context length and with the number of agent tasks that share the slots. The Phase 2 mechanism is therefore a memory-aware admission and retention policy: (1) admit new agent tasks only while the KV budget can hold their growing context (estimated from observed growth), (2) give ready continuations priority with slot affinity, with aging to protect long tasks, and (3) when a tool is slow, decide between keeping the slot, saving its state to SSD (measured above) or evicting it. Required baselines: FIFO, the best fixed K/J selected on the dev workload, and llama.cpp's built-in host prompt cache (`--cache-ram`) as the simple fixed cache policy; plus queue-only and cache-only ablations, policy overhead, and low-load / zero-delay cases where little benefit is expected.

## 7. Limitations

- Replay is append-only and benchmark-derived, not a byte-faithful trace; it does not re-execute customer-service semantics. Completion means "all recorded steps executed", not task success.
- Small number of tasks per run and few repeats; P95/P99 are descriptive. Error bars are SD across repeats.
- One laptop: thermal state and background processes are not fully controlled; RSS under-counts Metal allocations.
- The 600 s per-request HTTP deadline is a client limit, not a production SLO.
- The user simulator is DeepSeek, not TritonAI; it only shapes the captured trajectories.

## 8. Reproduce

```bash
bash scripts/overnight.sh         # capture → micro → cache → macro rounds → report
.venv/bin/python -m phase1.final --results-dir runs/final
```

Raw logs: `raw/`; tables: `summary/`; figures: `figures/` (PNG + PDF); pipeline log: `logs/pipeline.log`.
