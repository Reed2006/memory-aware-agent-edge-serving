# Phase 1 Report — Memory-Aware Agent Edge Serving (Option 1)

_Generated 2026-10-30T10:38 from `runs/aws-gpu` by `phase1.final`. All numbers come from the raw logs in this directory._

## 1. Question and baseline

Multi-step agents alternate model calls with tool calls and grow their context at every step. On a memory-constrained edge machine the KV cache that holds this context is the scarce resource. Phase 1 asks: **under a fixed KV memory budget, how do request concurrency, the number of server slots, the number of simultaneously active agent tasks, tool idle time and arrival rate affect completed agent work, tail latency, memory and fairness — and what is the dominant bottleneck?**

The baseline is a FIFO queue with fixed concurrency K in front of one llama-server: every ready agent step (first step or continuation after a tool) is dispatched in enqueue order; tool waits do not hold K. An optional task-level admission limit J caps how many agent tasks are active at once (waiting counts toward latency).

## 2. Setup

| Item | Value |
|---|---|
| Machine | EC2 g4dn.xlarge, Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz, 4 logical CPUs, 15.4 GiB memory, Tesla T4 (15360 MiB, driver 595.112.07) |
| Runtime | llama.cpp llama-server build 11669 (commit 620d2225c), CUDA, flash-attn, continuous batching |
| Model | Qwen3-4B Q4_K_M (sha256 7485fe6f…), f16 KV, thinking disabled; fixed in all comparisons |
| KV budget | --ctx-size tokens in total; partitioned per slot unless "u" (unified pool); f16 KV = 144 KiB/token |
| Agent benchmark | τ²-bench v1.0.1 Retail (official tools, orchestrator, ENV evaluator) |
| User simulator | DeepSeek V4 flash (`deepseek-flash`), live capture only; never called during timed replays |
| Power during runs | not recorded |
| Code commit(s) | 05fa9950, 22a1abb9 |


## 3. Workloads (micro → macro)

**Real application (live).** 20 eval tasks (test split) and 10 dev tasks (train split), sampled with seed 291 and kept regardless of outcome, were executed end-to-end: local Qwen3-4B agent, official Retail tools and DeepSeek user simulator. Eval tasks take 2–20 agent calls (median 11); the longest prompt per task is 5,030–9,677 tokens. ENV checks pass on 3/20 eval tasks: the 4B model is weak, and quality is reported, not filtered (F1). The capture is reused from `runs/final` as a frozen workload, so no replay here calls the user simulator.

**Macro (benchmark-derived).** Each live trajectory is converted to an append-only replay that keeps the exact number of steps, per-step output tokens and input growth, and retains tokens actually generated during the replay. A run draws N tasks from the frozen eval set (balanced, seeded), with each repeat's 1 task/min Poisson pattern compressed to 3.5 tasks/min (`scaled:3.5`), a rate chosen in a calibration pilot (`runs/aws-gpu-pilot`) for comparable contention, and controlled tool delays (default log-normal, median 2 s, σ=1, heavy-tailed). Every configuration sees identical arrivals, task order and delays for the same repeat index, and every experiment was run 3 times on these identical inputs (trials). The dev split is kept separate for selecting baselines.

**Micro (controlled).** Exact token counts with prompt cache disabled; K requests submitted together. **Cache probe.** Measures continuation cost when an agent's slot is retained, idle, overwritten, erased, or saved to SSD and restored.

**Metrics.** Completed tasks/min, P50/P95/P99 task completion time (arrival → last step, admission and queue included), FIFO and admission waits, failures and their reason, peak server RSS / min system available memory, prefill tokens that recompute already-seen history ("wasted prefill"), full-history hit rate, and fairness: slowdown = completion time / (isolated compute time + own tool delay), Jain index of 1/slowdown, and slowdown of long vs short tasks.

## 4. Results

### 4.1 Micro: prefill dominates long contexts (F2)

| input tokens | output | K | prefill s | round trip s | aggregate tok/s |
|---|---|---|---|---|---|
| 1,024 | 64 | 1 | 0.49 | 1.45 | 44.1 |
| 1,024 | 64 | 2 | 0.54 | 2.19 | 57.7 |
| 1,024 | 512 | 1 | 0.50 | 8.28 | 61.8 |
| 1,024 | 512 | 2 | 0.53 | 10.19 | 100.2 |
| 4,096 | 64 | 1 | 2.21 | 3.27 | 19.6 |
| 4,096 | 64 | 2 | 2.91 | 5.89 | 21.5 |
| 4,096 | 512 | 1 | 2.22 | 10.85 | 47.2 |
| 4,096 | 512 | 2 | 2.34 | 15.55 | 65.5 |
| 8,192 | 64 | 1 | 5.08 | 6.28 | 10.2 |
| 8,192 | 64 | 2 | 6.52 | 11.88 | 10.6 |
| 8,192 | 512 | 1 | 5.08 | 14.84 | 34.5 |
| 8,192 | 512 | 2 | 5.32 | 23.62 | 43.0 |


Prefill runs at roughly 1,565 tokens/s, so re-processing an 8K history costs 5.1 s, as long as generating about 334 output tokens at the measured 66 tokens/s.

### 4.2 Cache mechanisms (F3)

| input | condition | continuation prefill s | input reused |
|---|---|---|---|
| 2,048 | erase | 1.075 | 0.0% |
| 2,048 | overwrite | 1.068 | 0.0% |
| 2,048 | retain | 0.027 | 99.5% |
| 2,048 | save_restore | 0.024 | 99.5% |
| 2,048 | wait | 0.024 | 99.5% |
| 8,192 | erase | 5.132 | 0.0% |
| 8,192 | overwrite | 5.120 | 0.0% |
| 8,192 | retain | 0.030 | 99.9% |
| 8,192 | save_restore | 0.028 | 99.9% |
| 8,192 | wait | 0.028 | 99.9% |


| input | state MiB | save s | restore s | save+erase+restore+prefill s | re-prefill s |
|---|---|---|---|---|---|
| 2,048 | 292 | 0.848 | 0.116 | 0.989 | 1.07 |
| 8,192 | 1,156 | 8.947 | 0.472 | 9.449 | 5.13 |


Restore correctness: after save → overwrite → restore, 64 greedy tokens identical to the uninterrupted run: **True**.

### 4.3 Macro: FIFO baselines under a fixed 24K-token KV budget (F4–F6)

| config | repeats | completed | tasks/min | P50 s | P95 s | wasted prefill | history hit | mean slowdown |
|---|---|---|---|---|---|---|---|---|
| P1·24K K1 J∞ | 3 × 3 trials | 72/72 | 1.915 ± 0.547 | 84 | 147 | 79.0% | 24.9% | 1.75 |
| P2·24K K1 J1 | 3 × 3 trials | 72/72 | 1.229 ± 0.086 | 155 | 241 | 0.0% | 100.0% | 11.74 |
| P2·24K K1 J∞ | 3 × 3 trials | 72/72 | 1.914 ± 0.546 | 85 | 147 | 79.0% | 24.9% | 1.76 |
| P2·24K K2 J2 | 3 × 3 trials | 72/72 | 1.875 ± 0.455 | 73 | 138 | 38.6% | 91.1% | 4.88 |
| P2·24K K2 J3 | 3 × 3 trials | 72/72 | 2.096 ± 0.627 | 59 | 128 | 59.9% | 69.0% | 1.72 |
| P2·24K K2 J4 | 3 × 3 trials | 72/72 | 2.086 ± 0.620 | 64 | 129 | 60.4% | 61.9% | 1.56 |
| P2·24K K2 J∞ | 3 × 3 trials | 72/72 | 2.133 ± 0.662 | 65 | 132 | 62.2% | 56.9% | 1.44 |
| P2·24K+RAM3G K2 J∞ | 3 × 3 trials | 72/72 | 2.135 ± 0.663 | 66 | 132 | 63.9% | 56.6% | 1.45 |
| P4·24K K4 J∞ | 3 × 3 trials | 30/72 | 1.246 ± 0.459 | 16 | 19 | 35.7% | 82.0% | 1.55 |
| P4·36K K4 J∞ | 2 × 3 trials | 39/48 | 1.507 ± 0.429 | 29 | 102 | 46.7% | 90.2% | 1.18 |
| P4u·24K K4 J∞ | 3 × 3 trials | 48/72 | 1.511 ± 0.430 | 36 | 92 | 43.4% | 79.3% | 1.85 |


Tasks/min is mean ± SD across repeats (each repeat first averaged over its trials); P50/P95, wasted prefill and hit rate are means of per-run values. Latency is over completed tasks only, so configurations with failures (P4) look faster than they are: the tasks that fail are the long ones.

**Repeats are paired, not noise.** Each repeat index draws a different seeded mix of 8 tasks, and every configuration sees the same mix for the same index. The SD above is therefore mostly workload variation: FIFO P2 K2 J∞ has a median completion time of 58 s in R0, 28 s in R1, 111 s in R2. Throughput per repeat, and relative to FIFO P2 K2 J∞ on the same tasks:

| config | R0 tasks/min | R1 tasks/min | R2 tasks/min | R0 vs FIFO | R1 vs FIFO | R2 vs FIFO |
|---|---|---|---|---|---|---|
| P1·24K K1 J∞ | 2.257 | 1.284 | 2.204 | 0.91× | 0.94× | 0.86× |
| P2·24K K1 J1 | 1.313 | 1.141 | 1.231 | 0.53× | 0.83× | 0.48× |
| P2·24K K1 J∞ | 2.255 | 1.284 | 2.203 | 0.91× | 0.94× | 0.86× |
| P2·24K K2 J2 | 2.249 | 1.368 | 2.008 | 0.91× | 1.00× | 0.79× |
| P2·24K K2 J3 | 2.473 | 1.372 | 2.443 | 1.00× | 1.00× | 0.96× |
| P2·24K K2 J4 | 2.472 | 1.371 | 2.416 | 1.00× | 1.00× | 0.94× |
| P2·24K K2 J∞ | 2.472 | 1.371 | 2.557 | 1.00× | 1.00× | 1.00× |
| P2·24K+RAM3G K2 J∞ | 2.472 | 1.371 | 2.563 | 1.00× | 1.00× | 1.00× |
| P4·24K K4 J∞ | 1.092 | 0.883 | 1.762 | 0.44× | 0.64× | 0.69× |
| P4·36K K4 J∞ | 1.810 | 1.204 | n/a | 0.73× | 0.88× | n/a |
| P4u·24K K4 J∞ | 1.196 | 1.337 | 2.000 | 0.48× | 0.98× | 0.78× |


Highest throughput per repeat: R0 P2·24K K2 J3 (1.00×; within 1%: P2·24K K2 J4, P2·24K K2 J∞, P2·24K+RAM3G K2 J∞); R1 P2·24K K2 J3 (1.00×; within 1%: P2·24K K2 J2, P2·24K K2 J4, P2·24K K2 J∞, P2·24K+RAM3G K2 J∞); R2 P2·24K+RAM3G K2 J∞ (1.00×; within 1%: P2·24K K2 J∞). A task limit equal to the slot count (P2 K2 J2) reaches 0.91× FIFO in R0, 1.00× FIFO in R1, 0.79× FIFO in R2. Throughput counts completed tasks only, so a configuration whose long tasks fail (P4) can look fast on the tasks that remain.


Highest completed-task throughput on the eval workload: **P2·24K+RAM3G K2 J∞**. (Phase 2 must re-select the best fixed baseline on the dev workload; this ranking is descriptive.)

Failures (75 tasks across all runs) are kept in the denominators. Reasons seen: Context size has been exceeded.; request (N tokens) exceeds the available context size (N tokens), try increasing it.

Mean time per completed task (F5):

| config | admission wait | FIFO wait | model round trip | tool delay | total |
|---|---|---|---|---|---|
| P1·24K K1 J∞ | 0 | 25 | 21 | 34 | 80 |
| P2·24K K1 J1 | 108 | 0 | 14 | 34 | 157 |
| P2·24K K1 J∞ | 0 | 25 | 21 | 34 | 80 |
| P2·24K K2 J2 | 29 | 0 | 18 | 34 | 81 |
| P2·24K K2 J3 | 10 | 2 | 22 | 34 | 69 |
| P2·24K K2 J4 | 5 | 4 | 24 | 34 | 68 |
| P2·24K K2 J∞ | 0 | 8 | 26 | 34 | 68 |
| P2·24K+RAM3G K2 J∞ | 0 | 8 | 26 | 34 | 68 |
| P4·24K K4 J∞ | 0 | 0 | 6 | 7 | 13 |
| P4·36K K4 J∞ | 0 | 0 | 15 | 31 | 46 |
| P4u·24K K4 J∞ | 0 | 1 | 19 | 29 | 50 |


### 4.4 Memory (F7)

| server | theoretical KV GiB | peak server RSS GiB | min system available GiB |
|---|---|---|---|
| P1·24K | 3.38 | 0.86 | 13.94 |
| P2·24K | 3.38 | 0.86 | 13.92 |
| P2·24K+RAM3G | 3.38 | 0.86 | 13.92 |
| P4·24K | 3.38 | 0.85 | 13.92 |
| P4·36K | 5.06 | 0.86 | 13.91 |
| P4u·24K | 3.38 | 0.87 | 13.92 |


llama-server preallocates the whole KV pool at start, so resident memory is set by the budget, not by load; "too much work" therefore shows up as slot contention (evicted histories) or as hard context overflows, not as growing RSS. On a CUDA GPU the weights and the KV pool live in GPU memory, which server RSS does not include: nvidia-smi reported 2,481–7,739 MiB in use over the sweep (`logs/machine.log`), set by the server configuration rather than by load.

### 4.5 Idle time between steps (F10)

| tool delay | tasks/min | P50 s | P95 s | wasted prefill | history hit | slowdown |
|---|---|---|---|---|---|---|
| zero | 2.444 | 14 | 35 | 1.4% | 97.7% | 1.33 |
| lognormal:2:1 | 1.922 | 43 | 118 | 54.0% | 77.3% | 1.19 |
| fixed:15 | 1.098 | 156 | 281 | 74.6% | 18.9% | 1.06 |


Compared on the repeats every tool delay has (zero → lognormal:2:1 → fixed:15). Throughput falls from 2.444 to 1.922 to 1.098 tasks/min, P95 completion time rises from 35 s to 118 s to 281 s, the full-history hit rate falls from 98% to 77% to 19% and mean slowdown falls from 1.33 to 1.19 to 1.06.


### 4.6 Arrival rate (F11)

| arrival | tasks/min | P50 s | P95 s | wasted prefill | history hit | slowdown |
|---|---|---|---|---|---|---|
| scaled:1.75 | 1.144 | 40 | 108 | 33.5% | 92.5% | 1.09 |
| scaled:3.5 | 1.922 | 43 | 118 | 54.0% | 77.3% | 1.19 |
| scaled:7 | 2.461 | 58 | 130 | 68.4% | 47.6% | 1.39 |


Compared on the repeats every arrival rate has (scaled:1.75 → scaled:3.5 → scaled:7). Throughput rises from 1.144 to 1.922 to 2.461 tasks/min, P95 completion time rises from 108 s to 118 s to 130 s, the full-history hit rate falls from 92% to 77% to 48% and mean slowdown rises from 1.09 to 1.19 to 1.39.


### 4.7 Fairness (F8)

| config | short-task slowdown | long-task slowdown |
|---|---|---|
| P1·24K K1 J∞ | 1.87 | 1.63 |
| P2·24K K1 J1 | 21.22 | 2.26 |
| P2·24K K1 J∞ | 1.87 | 1.64 |
| P2·24K K2 J2 | 8.49 | 1.27 |
| P2·24K K2 J3 | 2.15 | 1.29 |
| P2·24K K2 J4 | 1.80 | 1.31 |
| P2·24K K2 J∞ | 1.50 | 1.38 |
| P2·24K+RAM3G K2 J∞ | 1.51 | 1.38 |
| P4·24K K4 J∞ | 1.51 | n/a |
| P4·36K K4 J∞ | 1.24 | 1.07 |
| P4u·24K K4 J∞ | 2.03 | 1.17 |


### 4.8 Dev workload (for Phase 2 baseline selection)

| P | K | J | runs | completed | tasks/min | P95 s | wasted prefill | history hit |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | ∞ | 3 | 24/24 | 2.091 | 201 | 80.3% | 14.1% |
| 2 | 1 | 2 | 3 | 24/24 | 1.730 | 175 | 75.6% | 43.6% |
| 2 | 2 | ∞ | 3 | 24/24 | 2.367 | 171 | 78.3% | 29.5% |
| 2 | 2 | 2 | 3 | 24/24 | 2.096 | 128 | 23.9% | 92.3% |
| 2 | 2 | 3 | 3 | 24/24 | 2.284 | 141 | 71.6% | 56.4% |
| 2 | 2 | 4 | 3 | 24/24 | 2.461 | 158 | 75.9% | 36.8% |


On the dev tasks the highest-throughput fixed baseline is **P2 K2 J4** (2.461 tasks/min, 1.04× FIFO P2 K2 J∞). With K = 1, a task limit of 2 reaches 0.83× K1 J∞: with one request in flight, llama.cpp keeps using a single slot (§4.9).


### 4.9 How llama.cpp assigns slots (P = 2)

| K | J | requests | share on busiest slot | full-history hit | misses that kept only the shared prefix |
|---|---|---|---|---|---|
| 1 | ∞ | 753 | 100% | 23.8% | 100% |
| 1 | 1 | 753 | 100% | 100.0% | n/a |
| 2 | ∞ | 3450 | 55% | 59.5% | 100% |
| 2 | 2 | 753 | 55% | 90.7% | 100% |
| 2 | 3 | 753 | 58% | 67.5% | 100% |
| 2 | 4 | 753 | 55% | 59.8% | 100% |


All Retail tasks start with the same 4,891-token system prompt and tool schema. llama.cpp picks the idle slot whose cached prompt shares the longest prefix with the new request, and with one request in flight (K = 1) every slot is idle, so it keeps choosing the same slot: **with K = 1 the second slot is never used**, and P2 K1 behaves like P1 K1. Two active tasks (J = 2) then overwrite each other in one slot while the other stays empty. When a history is lost, the shared prefix survives and only the task-specific part is recomputed. Slot choice is not agent-aware: pinning each task to a slot (`id_slot`) is a cheap first step for Phase 2.


### 4.10 Server startup (cold start)

| server | launches | model load s, median (min–max) | listening after s, median | warm-up request ms, median |
|---|---|---|---|---|
| P1·24K | 9 | 1.32 (1.31–1.33) | 1.37 | 245 |
| P2·24K | 23 | 1.31 (1.28–1.32) | 1.36 | 245 |
| P4·24K | 9 | 1.32 (1.30–1.35) | 1.37 | 247 |
| P4u·24K | 9 | 1.32 (1.30–1.43) | 1.37 | 244 |
| P4·36K | 6 | 1.35 (1.33–1.37) | 1.40 | 246 |


One row per `llama-server` launch is in `summary/startup.csv`, parsed from `logs/server-*.log` (`--cache-ram` is not visible in the log, so those launches are grouped by slots and budget). Loading the model takes 1.32 s at the median and at most 1.43 s: the weights are memory-mapped, so a launch whose model file is not already in the OS page cache (for example the first launch after a reboot) is slower. The first request after each launch is the bench warm-up (5 prompt + 16 generated tokens). Neither model load nor the warm-up enters any measurement above, so all results describe a warm server; on an edge device that starts the server on demand, both would add to the first task.


### 4.11 Eviction cliff: hit rate vs active tasks (F12)

| server | 1 active | 2 active | 3 active | 4 active | 5 active | 6+ active |
|---|---|---|---|---|---|---|
| P1·24K | 88% (78) | 27% (222) | 14% (169) | 10% (90) | 0% (53) | 0% (69) |
| P2·24K | 98% (1639) | 81% (2137) | 37% (1053) | 15% (538) | 1% (228) | 1% (243) |
| P2·24K+RAM3G | 94% (64) | 93% (243) | 40% (162) | 16% (119) | 0% (27) | 3% (66) |
| P4·24K | 97% (114) | 81% (156) | 72% (76) | 69% (29) | 40% (15) | – |
| P4·36K | 96% (77) | 94% (217) | 81% (79) | 77% (56) | – | – |
| P4u·24K | 98% (123) | 93% (215) | 84% (55) | 41% (74) | 0% (10) | 59% (32) |

Full-history hit rate of continuations (number of continuations), by how many agent tasks were admitted and unfinished when the step was dispatched. Pooled over all runs of each server, so it is descriptive. With two slots, a returning task finds its history 98% of the time when it is alone and 81% with two active tasks, but only 37% with three, and almost never with more. More slots move the cliff to more tasks, at the price of smaller per-slot contexts (§4.3). Eviction is set by how many tasks overlap relative to the slots, which is what an admission limit controls.


### 4.12 Context growth and the cost of a miss (F13)

| context tokens | median prefill s, history reused | median prefill s, history re-prefilled |
|---|---|---|
| 5,000–5,999 | 0.1 | 0.3 |
| 6,000–6,999 | 0.1 | 1.0 |
| 7,000–7,999 | 0.1 | 2.0 |
| 8,000–8,999 | 0.1 | 2.8 |
| 9,000–9,999 | 0.5 | 3.5 |

P2·24K K2, main condition: 2724 continuations, 868 misses. A step that reuses its history prefills only its new tokens (median 0.1 s overall). A miss usually keeps the 4,891-token shared prefix in the slot and recomputes the task-specific history (median 1,123 tokens), at about 0.7 s per 1,000 tokens under contention. The cost of each eviction therefore grows roughly linearly as the task's context grows, while a hit stays a matter of its new tokens and of sharing the batch. This is observational: long contexts are also late steps of long tasks, which overlap more other tasks; a controlled growth sweep is future work.


### 4.13 Memory pressure in tokens (F14)

| config | repeat | peak demand (tokens) | time above the 24K budget | mean active tasks | full-history hit | tasks/min |
|---|---|---|---|---|---|---|
| P2·24K K2 J2 | R0 | 14,839 | 0% | 1.9 | 89% | 2.249 |
| P2·24K K2 J∞ | R0 | 25,779 | 7% | 2.3 | 63% | 2.472 |
| P2·24K K2 J2 | R1 | 17,261 | 0% | 1.2 | 95% | 1.368 |
| P2·24K K2 J∞ | R1 | 20,247 | 0% | 1.2 | 91% | 1.371 |
| P2·24K K2 J2 | R2 | 17,853 | 0% | 1.8 | 89% | 2.008 |
| P2·24K K2 J∞ | R2 | 39,717 | 54% | 3.9 | 16% | 2.557 |

Demand is the KV the admitted tasks would need to keep their whole histories resident. Resident memory cannot show this (F7: the pool is preallocated), but the demand can: under FIFO it exceeds the budget whenever several tasks overlap, and every token above the budget belongs to a history that must be evicted and later re-prefilled. With partitioned slots eviction starts even earlier, as soon as more tasks than slots are active (F12). With J = 2 the demand never exceeds two histories, which always fit.


### 4.14 Admission limit (F15)

| J | tasks/min (mean of repeats) | P95 s | full-history hit | server busy | short-task slowdown |
|---|---|---|---|---|---|
| 1 (K1) | 1.229 | 241 | 100% | 29% | 21.22 |
| 2 | 1.875 | 138 | 91% | 45% | 8.49 |
| 3 | 2.096 | 128 | 69% | 55% | 2.15 |
| 4 | 2.086 | 129 | 62% | 57% | 1.80 |
| ∞ | 2.133 | 132 | 57% | 60% | 1.50 |

P2 under the main condition, J = 1 with K = 1 and the rest with K = 2. As J grows, the full-history hit rate falls from 100% to 91% to 69% to 62% to 57%, while the share of each run with the server busy rises from 29% to 45% to 55% to 57% to 60%; the highest mean throughput is at J = ∞.


### 4.15 Arrival-rate calibration (F16)

| run | tasks/min offered | mean active tasks | time above slots | full-history hit | tasks/min completed |
|---|---|---|---|---|---|
| R0 J∞ | 1.00 | 0.73 | 2% | 91% | 0.896 |
| R0 J∞ | 2.00 | 1.36 | 5% | 88% | 1.612 |
| R0 J∞ | 3.50 | 2.30 | 36% | 65% | 2.472 |
| R0 J∞ | 4.00 | 2.68 | 53% | 46% | 2.607 |
| R0 J∞ | 6.00 | 3.43 | 72% | 27% | 2.887 |
| R1 J∞ | 3.50 | 1.25 | 4% | 89% | 1.372 |
| R2 J∞ | 3.50 | 3.93 | 85% | 17% | 2.558 |
| R2 J2 | 3.50 | 1.78 | 0% | 89% | 2.008 |

Target from the initial run (M4, FIFO P2 K2 on R0 at 1 task/min): 2.39 active tasks, 41% of the time above the slot count, 69% hits. The main rate `scaled:3.5` was chosen from these pilot runs, which are not part of the sweep.


### 4.16 Run-to-run noise (every experiment repeated as trials)

| config | trials per repeat | tasks/min CV, mean over repeats | max CV | vs FIFO per trial, range per repeat |
|---|---|---|---|---|
| P1·24K K1 J∞ | 3 | 0.1% | 0.1% | R0 0.91–0.91× (all below 1); R1 0.94–0.94× (all below 1); R2 0.86–0.86× (all below 1) |
| P2·24K K1 J1 | 3 | 0.0% | 0.0% | R0 0.53–0.53× (all below 1); R1 0.83–0.83× (all below 1); R2 0.48–0.48× (all below 1) |
| P2·24K K1 J∞ | 3 | 0.0% | 0.1% | R0 0.91–0.91× (all below 1); R1 0.94–0.94× (all below 1); R2 0.86–0.86× (all below 1) |
| P2·24K K2 J2 | 3 | 0.0% | 0.0% | R0 0.91–0.91× (all below 1); R1 1.00–1.00× (all below 1); R2 0.78–0.79× (all below 1) |
| P2·24K K2 J3 | 3 | 0.1% | 0.3% | R0 1.00–1.00×; R1 1.00–1.00× (all above 1); R2 0.95–0.96× (all below 1) |
| P2·24K K2 J4 | 3 | 0.3% | 0.8% | R0 1.00–1.00× (all below 1); R1 1.00–1.00×; R2 0.94–0.95× (all below 1) |
| P2·24K K2 J∞ | 3 | 0.1% | 0.1% | – |
| P2·24K+RAM3G K2 J∞ | 3 | 0.1% | 0.3% | R0 1.00–1.00×; R1 1.00–1.00×; R2 1.00–1.01× |
| P4·24K K4 J∞ | 3 | 0.0% | 0.1% | R0 0.44–0.44× (all below 1); R1 0.64–0.64× (all below 1); R2 0.69–0.69× (all below 1) |
| P4·36K K4 J∞ | 3 | 0.4% | 0.7% | R0 0.73–0.74× (all below 1); R1 0.88–0.88× (all below 1) |
| P4u·24K K4 J∞ | 3 | 0.5% | 1.3% | R0 0.48–0.48× (all below 1); R1 0.97–0.98× (all below 1); R2 0.78–0.79× (all below 1) |

Trials replay a repeat with identical tasks, arrivals and tool delays, so their spread is run-to-run noise; the tables above average them first. Each trial restarted the servers and ran the configurations in a rotated order. The largest CV of tasks/min is 1.3%: a paired difference smaller than about 3% in a single run would not be distinguishable from noise. A comparison is consistent when every trial falls on the same side of FIFO.


## 5. Dominant bottleneck

Across baselines the share of prefill tokens spent recomputing history ranges from 0.0% (P2·24K K1 J1) to 79.0% (P1·24K K1 J∞); full-history hit rates range 25%–100%. Across configurations, wasted-prefill share and completed tasks/min correlate at r = 0.76. Under FIFO, re-prefill takes 15% of model time (prefill + decode) and the server is busy 60% of each run; with J = 2 the server is busy 45% of the time. On the same tasks, a task limit equal to the slot count (P2 K2 J2) reaches 0.79× FIFO in the most contended repeat (R2, FIFO median completion 111 s) and 1.00× in the least contended (R1). Even where tasks overlap most, the limit costs throughput: a lost history is cheap to rebuild (0.7 s per 1,000 tokens, §4.12), while the admitted tasks spend much of their time in tool calls and leave the server idle. Resident memory stays flat because the pool is preallocated, so the memory limit acts through the number and size of slots: more slots per budget means smaller per-slot contexts (overflow failures for long Retail histories), fewer slots means agents evict one another's histories while their tools run. Raising the budget from 24K to 36K tokens at P4 raises completed tasks from 42% to 81%.


## 6. Phase 2 direction

The measured findings that set the Phase 2 baselines and mechanism:

1. **The best fixed baseline on the dev workload is P2 K2 J4** (2.461 tasks/min, 1.04× FIFO). Phase 2 has to beat it, not only FIFO, on the eval workload.
2. **A tight task limit is unfair to short tasks**: their mean slowdown is 8.49 under P2 K2 J2 against 1.50 under FIFO, because they wait for admission behind long tasks (§4.7).
3. **Slot placement is not agent-aware** (§4.9): with one request in flight, 100% of requests go to the same slot. Slot affinity per task is a cheap first mechanism.
4. **Retention by save/restore**: saving one slot's 8,192-token state takes 8.95 s and restoring it 0.47 s, against 5.13 s to re-prefill (§4.2), so writing a checkpoint costs more than rebuilding, and a checkpoint helps only if it is written while the task waits on a tool, off the critical path.
5. **llama.cpp's host prompt cache** (`--cache-ram 3072`) reaches 1.00× FIFO on average: a fixed cache does not address eviction.

The Phase 2 mechanism is therefore an adaptive admission and retention policy: (1) admit agent tasks while the KV budget can hold their growing histories and while the server would otherwise idle during tool calls, (2) give ready continuations priority with per-task slot affinity, with aging so neither short nor long tasks starve, and (3) when a tool call is long, decide between keeping the slot, checkpointing it, or evicting it. Required baselines: FIFO, P2 K2 J4 (best fixed on dev), and `--cache-ram` as the simple fixed cache policy; plus queue-only and cache-only ablations, policy overhead, and low-load / zero-delay cases where little benefit is expected.

## 7. Limitations

- Replay is append-only and benchmark-derived, not a byte-faithful trace; it does not re-execute customer-service semantics. Completion means "all recorded steps executed", not task success.
- 8 tasks per run and 3 repeats (2 for P4·36K and for the tool-delay / arrival sweeps); P95/P99 are descriptive. Error bars are SD across repeats, which mostly reflects the different task mix of each repeat (§4.3). Run-to-run noise on identical inputs is in §4.16.
- Latency of configurations with failures is over completed tasks only.
- Each macro run is 8 tasks starting from erased slots (an empty KV cache), so it includes a ramp-up and a drain; there is no separate steady-state window, and tasks/min includes the drain tail. Server startup is reported in §4.10.
- One dedicated EC2 GPU instance with nothing else running; the GPU ran at its board power cap under load; its SM clock ranged 1,065–1,590 MHz (`logs/machine.log`). Server RSS excludes GPU memory (§4.4).
- The main arrival `scaled:3.5` was calibrated on this machine; absolute numbers are not comparable with runs on other machines.
- The 600 s per-request HTTP deadline is a client limit, not a production SLO.
- The user simulator is DeepSeek, not TritonAI; it only shapes the captured trajectories.

## 8. Reproduce

```bash
# pipeline settings for this directory (logs/pipeline.log): commit 05fa995, trials 3, arrival scaled:3.5
.venv/bin/python -m phase1.final --results-dir runs/aws-gpu --main-arrival scaled:3.5   # figures + this report
```

Setup, options and the AWS procedure are in the project `README.md`; definitions and validity rules in `docs/DESIGN.md`.

Raw logs: `raw/`; tables: `summary/`; figures: `figures/` (PNG + PDF); pipeline log: `logs/pipeline.log`.
