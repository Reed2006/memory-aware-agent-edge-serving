# Phase 1 Design and Methodology

This document records how the Phase 1 experiments are defined and why. Results are in
[`runs/final/PHASE1_REPORT.md`](../runs/final/PHASE1_REPORT.md).

## Research question

Several multi-step agents share one memory-constrained machine. An agent waiting on a tool
does not need the model, but its KV state is still useful for its next step. The server must
trade off two things: freeing capacity for other work, and sparing a returning agent from
recomputing its history. Phase 1 implements and measures a fixed-concurrency FIFO baseline to
find out whether this conflict is real and how much it costs. Phase 2 designs a policy around
the bottleneck Phase 1 finds. Phase 1 assumes neither that recomputation dominates nor that a
joint policy beats a tuned FIFO.

## Platform

| Item | Choice | Reason |
|---|---|---|
| Machine | MacBook Air M4, 16 GB unified memory, macOS 15.3 | The course allows a laptop for Option 1; CPU and GPU share physical memory |
| Runtime | llama.cpp `llama-server` build 11669 (commit `620d2225c`), Metal | Exposes per-slot KV reuse, erase, save and restore over HTTP; the version is pinned |
| Model | Qwen3-4B Q4_K_M (official GGUF), thinking disabled | Small enough to leave room for a multi-GiB KV pool; full attention makes KV cost easy to reason about |
| KV precision | f16, 144 KiB/token (36 layers × 8 KV heads × 128 head dim × K,V × 2 bytes) | Fixed in every comparison |
| Python | 3.13 in `.venv` | τ²-bench v1.0.1 requires `>=3.12,<3.14` |

The KV pool is preallocated from `--ctx-size` when the server starts, so resident memory is
set by the budget, not by load. Unified memory means RSS and Metal buffers must not be added
together.

## Terms

- **P**: server slots (`--parallel`). In partitioned mode each slot owns `ctx-size / P`
  tokens. `--kv-unified` makes the slots share one pool.
- **K**: model requests in flight, enforced by the external FIFO. Tool waits release the permit.
- **J**: agent tasks admitted at once (task-level admission, 0 = unlimited). Time spent
  waiting for admission counts toward latency.
- **FIFO**: every ready step (a first step, or a continuation after a tool returns) is
  dispatched in enqueue order. A task waiting on a tool never blocks ready work, so FIFO order
  over steps is not the same as permanent priority by task arrival.
- **Simple fixed cache policy**: llama.cpp's host prompt cache (`--cache-ram MiB`). It is off
  (`0`) in every other configuration, together with `--no-cache-idle-slots`, so that slot
  overwrites stay visible.

## Workloads

1. **Live (real application).** τ²-bench v1.0.1 Retail, using the official tools,
   orchestrator and environment evaluator. The local Qwen3-4B model plays the agent and
   DeepSeek V4 flash (`deepseek-flash`) plays the user simulator.
   - Tasks: 20 eval (test split) and 10 dev (train split), sampled with seed 291 and kept
     whether they pass or fail.
   - Every task gets its own copy of the initial database.
   - DeepSeek is called only during capture, never during timed replays.
2. **Macro (benchmark-derived replay).** Each live trajectory is turned into an append-only
   token replay:
   - The next prompt is `P_{s+1} = P_s ‖ G_s ‖ Δ_{s+1}`, where `G_s` is the set of tokens
     actually generated in this replay and `Δ` is the observation, calibrated to the captured
     input growth. The cached prefix therefore matches the next input exactly.
   - Output length is fixed per step (`n_predict`, `ignore_eos`) and checked on every response.
   - A run draws 8 tasks from the frozen eval set, with seeded Poisson arrivals and seeded
     tool delays (default: log-normal, median 2 s, σ = 1).
   - Every configuration sees the same arrivals, task order and delays for a given repeat
     index. Different repeat indices draw different task mixes.
   - Completion means "all recorded steps executed", not semantic task success.
3. **Micro.** Exact token counts with the prompt cache disabled: inputs of 1K / 4K / 8K tokens,
   outputs of 64 / 512 tokens, K = 1 / 2.
4. **Cache probe.** Measures the cost of a continuation after the slot is:
   - retained,
   - left idle,
   - overwritten by another request,
   - erased,
   - or saved to SSD and restored.

   A separate check confirms that greedy output is identical after save → overwrite → restore.

The dev split is used to choose baselines; the eval split is used for reported comparisons.

## Metrics

| Level | Metrics |
|---|---|
| Task | completed tasks/min, P50/P95/P99 completion time (arrival → last step), admission wait, failures and their reason |
| Request | FIFO wait, round trip, `prompt_n`, `cache_n`, `prompt_ms`, slot |
| Cache | wasted prefill = prompt tokens evaluated beyond the step's new tokens; full-history hit = `cache_n ≥ len(previous prompt + output) − 1` |
| Memory | server RSS and system available memory, sampled every 250 ms; theoretical KV pool size |
| Startup | per server launch, from `logs/server-*.log`: model load time, time until listening, first (warm-up) request time |
| Fairness | slowdown = completion time / (isolated compute + own tool delay), where isolated compute is the median over J = 1, K = 1 runs; Jain index of 1/slowdown; short- vs long-task slowdown |

Notes on interpreting these:
- Latency percentiles cover completed tasks only; failures are reported separately.
- Request round trip includes queueing inside `llama-server` when K > 1, so it is not an
  exclusive service time.

## Experiment matrix

| Experiment | Varied | Held fixed |
|---|---|---|
| Main | P ∈ {1, 2, 4}, K, J ∈ {∞, 1, 2}, unified KV, `--cache-ram` 3 GiB | 24K-token budget, log-normal delay, Poisson 1 task/min |
| Budget | P4 at 36K tokens (9K per slot) | everything else |
| Tool delay | none / log-normal median 2 s / fixed 15 s | P2 K2, Poisson 1/min |
| Arrival rate | Poisson 0.5 / 1 / 2 tasks per minute | P2 K2, log-normal delay |
| Dev | P2, K ∈ {1, 2}, J ∈ {∞, 2} | dev workload |

There are 3 repeats per configuration (2 for the budget, tool-delay and arrival sweeps).
Configurations are compared within the same repeat, because each repeat is a different task
mix.

## Validity rules

- Run one experiment on the server at a time: live capture and timed replays never overlap.
- Pin the runtime, model hash, KV precision, workload files and code commit. Record the git
  commit, server command and power state with every run (`raw/runs_meta.jsonl`).
- Keep failures and timeouts in the denominators. Record context-overflow errors with the
  server's message.
- Logs are append-only and resumable:
  - `--skip-done` skips runs whose label is already logged.
  - An interrupted run is moved to `raw/interrupted/` before it is re-run, so it is never
    double-counted.
- Run on AC power with no swap. If memory is short, skip a configuration (the P4/36K run
  needs ≥ 7 GiB available) rather than run it degraded.
- Mac and AWS results each stand alone; absolute speeds are never mixed.

## Known limitations

- The replay is benchmark-derived and append-only, not a byte-faithful production trace.
- Runs are small (8 tasks), so P95/P99 are descriptive.
- On one laptop, thermal state and background load are only partly controlled.
- The 4B agent passes the environment check on only 3 of 20 eval tasks. Quality is reported,
  not filtered.
