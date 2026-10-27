# Phase 1 Report — Memory-Aware Agent Edge Serving (Option 1)

_Generated 2026-10-26T21:44 from `runs/final` by `phase1.final`. All numbers come from the raw logs in this directory._

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
| Power during runs | AC power for 58 of 59 recorded launches; the 1 battery launch was the run stopped by hand, quarantined in `raw/interrupted/` and re-run on AC |
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


Tasks/min is mean ± SD across repeats; P50/P95, wasted prefill and hit rate are means of per-run values. Latency is over completed tasks only, so configurations with failures (P4) look faster than they are: the tasks that fail are the long ones.

**Repeats are paired, not noise.** Each repeat index draws a different seeded mix of 8 tasks, and every configuration sees the same mix for the same index. The SD above is therefore mostly workload variation: FIFO P2 K2 J∞ has a median completion time of 158 s in R0, 50 s in R1, 747 s in R2. Throughput per repeat, and relative to FIFO P2 K2 J∞ on the same tasks:

| config | R0 tasks/min | R1 tasks/min | R2 tasks/min | R0 vs FIFO | R1 vs FIFO | R2 vs FIFO |
|---|---|---|---|---|---|---|
| P1·24K K1 J∞ | 0.617 | 0.428 | 0.405 | 0.75× | 0.88× | 0.90× |
| P2·24K K1 J1 | 0.830 | 0.455 | 0.745 | 1.01× | 0.93× | 1.65× |
| P2·24K K1 J∞ | 0.507 | 0.437 | 0.406 | 0.62× | 0.89× | 0.90× |
| P2·24K K2 J2 | 0.843 | 0.474 | 0.891 | 1.03× | 0.97× | 1.97× |
| P2·24K K2 J∞ | 0.820 | 0.488 | 0.452 | 1.00× | 1.00× | 1.00× |
| P2·24K+RAM3G K2 J∞ | 0.661 | 0.487 | 0.448 | 0.81× | 1.00× | 0.99× |
| P4·24K K4 J∞ | 0.346 | 0.269 | 0.656 | 0.42× | 0.55× | 1.45× |
| P4·36K K4 J∞ | 0.628 | 0.429 | n/a | 0.77× | 0.88× | n/a |
| P4u·24K K4 J∞ | 0.408 | 0.477 | 0.510 | 0.50× | 0.98× | 1.13× |


Limiting active tasks (J = P) wins only when histories are long (R0, R2) and is neutral on short tasks (R1). Throughput also counts completed tasks only: P4·24K looks fast in R2 because its long tasks failed early.


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
| lognormal:2:1 | 0.654 | 104 | 246 | 28.8% | 83.0% | 2.13 |
| fixed:15 | 0.456 | 323 | 510 | 66.0% | 43.1% | 1.76 |


Compared on the repeats every delay has. Only a long fixed idle time (15 s) clearly raises eviction and lowers throughput; the gap between no delay and log-normal delay is within repeat-to-repeat variation.


### 4.6 Arrival rate (F11)

| arrival | tasks/min | P50 s | P95 s | wasted prefill | history hit | slowdown |
|---|---|---|---|---|---|---|
| poisson:0.5 | 0.467 | 81 | 188 | 40.0% | 92.1% | 1.34 |
| poisson:1 | 0.654 | 104 | 246 | 28.8% | 83.0% | 2.13 |
| poisson:2 | 0.801 | 185 | 416 | 47.1% | 55.1% | 3.77 |


Compared on the repeats every rate has. The lowest rate (0.5/min) gives the best fairness and hit rate; between 1 and 2/min, throughput rises with offered load while P95 latency and eviction grow.


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

| P | K | J | runs | completed | tasks/min | P95 s | wasted prefill | history hit |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | ∞ | 1 | 8/8 | 0.437 | 863 | 80.7% | 12.8% |
| 2 | 1 | 2 | 1 | 8/8 | 0.425 | 773 | 81.0% | 15.4% |
| 2 | 2 | ∞ | 1 | 8/8 | 0.453 | 887 | 77.0% | 14.1% |
| 2 | 2 | 2 | 1 | 8/8 | 0.914 | 263 | 0.0% | 100.0% |


On the dev tasks the best fixed baseline is again P2 K2 J2 (J = number of slots). K1 J2 gains nothing over K1 J∞; §4.9 explains why.


### 4.9 How llama.cpp assigns slots (P = 2)

| K | J | requests | share on busiest slot | full-history hit | misses that kept only the shared prefix |
|---|---|---|---|---|---|
| 1 | ∞ | 251 | 100% | 22.5% | 100% |
| 1 | 1 | 251 | 100% | 100.0% | n/a |
| 2 | ∞ | 1150 | 58% | 56.0% | 100% |
| 2 | 2 | 251 | 60% | 95.6% | 100% |


All Retail tasks start with the same 4,891-token system prompt and tool schema. llama.cpp picks the idle slot whose cached prompt shares the longest prefix with the new request, and with one request in flight (K = 1) every slot is idle, so it keeps choosing the same slot: **with K = 1 the second slot is never used**, and P2 K1 behaves like P1 K1. Two active tasks (J = 2) then overwrite each other in one slot while the other stays empty. When a history is lost, the shared prefix survives and only the task-specific part is recomputed. Slot choice is not agent-aware: pinning each task to a slot (`id_slot`) is a cheap first step for Phase 2.


## 5. Dominant bottleneck

Across baselines the share of prefill spent recomputing history ranges from 0.0% (P2·24K K1 J1) to 78.0% (P2·24K K1 J∞); full-history hit rates range 25%–100%. Across configurations, wasted-prefill share and completed tasks/min correlate at r = -0.43. The paired comparison is clearer: on the same tasks, P2·24K K2 J2 reaches 1.97× the throughput of FIFO P2 K2 J∞ in the long-task repeat R2 but only 0.97× in the short-task repeat R1. Removing eviction pays off in proportion to how much history there is to lose. Resident memory stays flat because the pool is preallocated, so the edge-memory limit acts through the number and size of slots: more slots per budget means smaller per-slot contexts (overflow failures for long Retail histories), fewer slots means agents evict one another's histories while their tools run. Raising the budget from 24K to 36K tokens at P4 cuts failures sharply, but P4 still does not beat P2 with a task limit.


## 6. Phase 2 direction

Phase 1 shows that, at a fixed KV budget, the cost that FIFO fails to control is **history eviction followed by re-prefill**, which grows with context length and with the number of agent tasks that share the slots. Four findings shape Phase 2:

1. **A fixed task limit J = P is already a strong baseline** (best on eval and dev, up to ~2× FIFO on long tasks). Phase 2 has to beat it, not FIFO.
2. **That baseline is unfair to short tasks**: they wait for admission behind long ones (§4.7). A policy that admits by predicted KV footprint and remaining steps can keep J's cache hits while cutting short-task slowdown.
3. **Slot placement is not agent-aware** (§4.9): with one request in flight llama.cpp reuses one slot and leaves the other empty. Slot affinity per task is required.
4. **Simple fixed caching does not help**: llama.cpp's host prompt cache (`--cache-ram 3072`) is no better than none on the same tasks, while SSD save/restore of one slot costs < 1 s against 44 s to re-prefill 8K tokens (§4.2). Per-slot capacity, not slot count, decides whether long tasks fit (P4 fails at 6K per slot, mostly recovers at 9K).

The Phase 2 mechanism is therefore a memory-aware admission and retention policy: (1) admit new agent tasks only while the KV budget can hold their growing context (estimated from observed growth), (2) give ready continuations priority with slot affinity, with aging so neither short nor long tasks starve, and (3) when a tool is slow, decide between keeping the slot, saving its state to SSD or evicting it. Required baselines: FIFO, the best fixed K/J selected on the dev workload (P2 K2 J2), and `--cache-ram` as the simple fixed cache policy; plus queue-only and cache-only ablations, policy overhead, and low-load / zero-delay cases where little benefit is expected.

## 7. Limitations

- Replay is append-only and benchmark-derived, not a byte-faithful trace; it does not re-execute customer-service semantics. Completion means "all recorded steps executed", not task success.
- 8 tasks per run and 3 repeats (2 for P4·36K and for the tool-delay / arrival sweeps); P95/P99 are descriptive. Error bars are SD across repeats, which mostly reflects the different task mix of each repeat (§4.3).
- Latency of configurations with failures is over completed tasks only.
- One laptop: thermal state and background processes are not fully controlled; RSS under-counts Metal allocations.
- The 600 s per-request HTTP deadline is a client limit, not a production SLO.
- The user simulator is DeepSeek, not TritonAI; it only shapes the captured trajectories.

## 8. Reproduce

```bash
bash scripts/overnight.sh         # capture → micro → cache → macro rounds → report
.venv/bin/python -m phase1.final --results-dir runs/final
```

Raw logs: `raw/`; tables: `summary/`; figures: `figures/` (PNG + PDF); pipeline log: `logs/pipeline.log`.
