# Phase 2 plan: coordinating agent scheduling and KV retention by measured cost

> Translated from the Chinese original on 2026-11-22. This is the design history. The
> current state is in [0_README.md](0_README.md), the results are in
> [4_FAIRNESS_EFFICIENCY.md](4_FAIRNESS_EFFICIENCY.md), and the next steps are in
> [7_TODO.md](7_TODO.md). Folder names are updated to the numbered layout:
> `runs/phase2/1_m4_pilot` was `runs/phase2-m4-air-16g`, and `runs/phase2/2_m4_campaign` was
> `runs/phase2-m4-air-16g-r2`.

Written 2026-11-14. Final submission is due 2026-12-10. This document is the plan for
implementation and evaluation. When it was written, the new policy was neither
implemented nor validated.

## Updates

**2026-11-16.**
- The core scheduler, cost model, checkpoints, replay, probes, experiment manifests and
  analysis tools described below are implemented. How to run them is in
  [2_RUNBOOK.md](2_RUNBOOK.md).
- Validation so far is limited to mock and model-free checks. Real-model compatibility,
  cost calibration, quality and performance were still to be measured.
- Where later sections say "to be added", they record the original design; the code in
  `phase2/` is authoritative.
- Section 13 records the gaps found by the second review on 11-16 and the changes made.

**2026-11-18.**
- The M4 pilot finished calibration, mechanism probes, 60 dev runs and 10 eval runs. The
  review is in [3_PILOT_REVIEW_2026-11-18.md](3_PILOT_REVIEW_2026-11-18.md).
- Revision 2 changes the procedure:
  - Screen joint aging first. Then confirm Q/C/CP and TTL 10/15/30 at that same aging.
    Then freeze the complete configuration.
  - A dev score is never replaced by a parameter that was not measured.
- Checkpoints use the same cost judgment as every other executable action.
- Online prefill updates, by default, only use samples with at least 128 actually
  computed tokens.
- The all-policy T0 table and the three-trial core table are counted separately. The
  core includes the strongest affinity fixed baseline.
- New seeds: eval 1101–1103, sensitivity 2101–2102.
- The old pilot is kept as exploratory evidence. The new version's performance still
  had to be confirmed.
- The early sections below keep the design history. For execution details, the runbook
  and the source code are authoritative.

## Research question

**With a fixed KV capacity, and using only information the runtime has already observed,
can we decide when a new task starts, when a waiting task keeps its slot, and when its
history is saved or recomputed, so that the system balances task completion speed and
waiting fairness?**

The core is a small event-driven Python management layer:
- it tracks slot ownership explicitly;
- it coordinates execution with time-limited reservations and aging-based scheduling;
- it enables checkpoints only after their full cost is counted.

The M4 is the platform for full mechanism validation. The T4 tests whether the policy
makes different choices when recompute is cheap and saving is expensive.

## 1. Materials read and course requirements

All five files in `Requirements/` were read; the old discussion documents total 2,082
lines. We also read the Phase 1 PDF, the full LaTeX source and appendix, README, DESIGN,
the detailed REPORT, and the core runner, server, cache probe and test structure.

| File | What it actually requires of Phase 2 |
| --- | --- |
| `CSE 291A Fall 2026 Course Projects.md` | Submit by Dec 10. Design on dev workloads and evaluate on eval workloads. Pin versions. Record failures, timeouts, repeats and variability. Submit reproducible materials. |
| `CSE 291A Fall 2026 Course Projects (1).md` | Option 1: focus on one mechanism supported by Phase 1. Compare against FIFO and the best fixed baseline chosen on dev. A cache design also needs a simple fixed cache baseline. Check fairness, overhead and conditions with no benefit. |
| `Professor feedback.md` | Do not rely only on made-up micro benchmarks. Changing FIFO requires measuring fairness. Tiered memory and KV compression are suggestions for depth; not every feature must be implemented. |
| `edge_agent_queue_cache_memo.pdf` | The original claim is to coordinate ready-step scheduling with keep/save/evict during waits. It requires queue-only, cache-only and joint ablations. |
| `ChatGPT-Phase 1 plan-20261025-1703.md` (a Chinese chat log; `Requirements/` is not in the repository) | Historical discussion, not a description of the current implementation. Its CPU instance, model size and early timeline cannot override the later M4/T4 measurements. |

The memo's RSS high/low watermarks should not be copied. The runtime preallocates the KV
pool, so it does not grow with the number of tasks. The main control variables should be:
- whether a slot is occupied, and who owns its cache;
- the potential recompute;
- the ready backlog and waiting time.

Physical memory sets the experiment boundary and guards against anomalies.

The course does not require a paper or model training. The most convincing outcome is a
small working system, a set of well-controlled comparisons, and evidence that explains
why the policy makes different decisions under different conditions.

## 2. What Phase 1 supports, and what this investigation found

### 2.1 Main existing evidence

The setup is fixed:
- Qwen3-4B Q4_K_M, f16 KV, llama.cpp build 11669 / commit `620d2225c`;
- main configuration P=2, K=2, total KV 24,576 tokens, so 12,288 tokens per slot;
- theoretical KV pool about 3.38 GiB.

| Observation | M4 | T4 | Design implication |
| --- | --- | --- | --- |
| Recompute of 8K cold history | ~44 s | ~5.1 s | One retention rule cannot be assumed to fit both platforms. |
| Save / restore an 8K slot | 0.37 / 0.35 s | 8.95 / 0.47 s | Save cost must be counted, not just restore vs prefill. |
| J=2 vs FIFO, R0/R1/R2 | 1.03 / 0.97 / 1.97× | 0.91 / 1.00 / 0.79× | A high hit rate can come with lower utilization. |
| Historical best fixed J on dev | 2 | 4 | The comparison must be against a tuned fixed policy. |
| Short-task slowdown under J=2 | ~9.5 | ~8.5 | New tasks waiting a long time for admission must be addressed. |

These are values from the historical reports, not results of the new policy. The near-2×
gain on the M4 appears only on R2; it must not be written as a general doubling. The M5
noise trials are not repeats of the M4 runs.

### 2.2 The shared prefix changes the value of checkpoints

A read-only look at `runs/final/workload_eval.json`:
- The 20 eval tasks have 205 model steps. Steps per task are min/median/max 2/11/20.
- All 20 tasks share a 4,891-token prefix. Summed over all step inputs, the prefix is
  about 79.61% of input tokens.
- The private part of each task's last prompt has a median of 1,837 tokens and a maximum
  of 4,786.

Below are FIFO runs only: P2/C24576/K2/Jall, no host cache, main arrival,
`lognormal:2:1`.

| Platform and scope | Continuations / misses | Median wasted prefill tokens per miss | Median whole prefill time per miss |
| --- | --- | --- | --- |
| M4, R0–R2 once each | 227 / 102 | 671.5 | 6.047 s |
| T4, R0–R2 × three trials | 681 / 310 | 964 | 0.857 s |

How to read the table:
- **Definitions** follow Phase 1. A miss is `cache_n < reusable_tokens`. Waste is
  `max(prompt_n - new_tokens, 0)`.
- **The last column** includes computing the step's new tokens, so it **is not the
  recompute cost that could be saved**.
- **On the T4**, only 6 of the 310 misses had `cache_n=0`. The rest usually kept the
  shared prefix.

An 8K checkpoint, by contrast, saves the whole slot: about 1,212,663,176 bytes measured.
So the real decision is often "recompute a few hundred to a thousand private tokens" vs
"read and write a GB-sized state". The incremental cost must be recalibrated; the 8K cold
prefill table cannot be applied directly.

### 2.3 What the waits mean must be corrected

- **Real tool execution time is tiny.** In `raw/live_test/task_*.json`, the slowest of
  100 real Retail tool executions took 2.984 ms. The slowest of 64 train executions took
  0.257 ms.
- **The frozen workload's `gap_s`** is the interval between two agent calls, including
  the user simulator and orchestration. On eval its median is about 0.963 s, P95
  1.656 s and maximum 2.186 s.
- **The formal replay uses synthetic waits** drawn from an independent log-normal. The
  227 waits of the three eval mixes have:
  - median about 2.056 s, mean 3.625 s;
  - P95 10.114 s, maximum about 63.42 s;
  - about 7.93% above 9 s.

These synthetic waits can be used to study controlled external waits. They cannot be
called a measured distribution of slow Retail tools, and they cannot show that a tool's
name predicts a random wait.

### 2.4 Baseline selection and sample size need strengthening

The existing dev has only one 8-task mix, R0, covering 8 of 10 tasks. The three T4
trials replay the same input, so they reflect only runtime noise. The three eval mixes
have 24 task occurrences covering 17 of 20 tasks.

Phase 2 should cover all dev tasks and add independent permutations and arrival/wait
seeds before freezing rules.

Phase 1 analysis already used the eval set, so new seeds only provide new load
permutations, not a never-seen task set. Claiming generalization across tasks would
require freezing unused benchmark tasks. Otherwise the conclusions apply to the current
task pool only.

### 2.5 The old host-cache configuration needs a validity control

The existing launcher also passed `--no-cache-idle-slots` to the host-cache group. The
pinned source has a further condition: when a slot is chosen by longest common prefix,
the RAM cache save/load on that path triggers only if more than half of the old context
would be lost. With the long shared prefix, that path may rarely fire.

So the historical finding that host cache did not help covers only the tested
configuration. It cannot be generalized to "RAM offload does not work".

Phase 2 must test a working RAM-cache control:
- `--cache-ram` together with `--cache-idle-slots`;
- confirmation from the server log and from real private-history hits that state was
  actually saved and restored;
- the extra RAM and the foreground copy cost counted.

Reference: [slot selection and cache update in the pinned
build](https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/server-context.cpp#L1696).

## 3. Related work: what to borrow, and how to position the contribution

Status as checked on 2026-11-14. Preprints are not described as accepted. Read the first
six closely; do not reproduce every system.

| Paper and primary source | Mechanism to borrow | Decision for this project |
| --- | --- | --- |
| [InferCept, ICML 2024, Abhyankar et al.](https://proceedings.mlr.press/v235/abhyankar24a.html) | Coordinates Preserve / Swap / Discard on tool interrupts, counting the cost to interfered foreground work. | keep/save/recompute has direct prior work. This project's whole-slot SSD API is not the paper's chunked GPU↔CPU pipeline. |
| [Agentix, NSDI 2026, Luo et al.](https://www.usenix.org/conference/nsdi26/presentation/luo) (formerly Autellix) | Schedules by the service a program's completed calls have received. | Build an attained-service baseline that reads no future lengths. An external scheduler cannot do the paper's token-level preemption, so label it an inspired adaptation. |
| [Continuum, arXiv:2511.02230, v7 2026-09-08, Li et al.](https://arxiv.org/abs/2511.02230) | TTL retention combined with program scheduling, trading rebuild against queueing. | A fixed TTL must be a strong simple baseline. This project does not claim the first tool-aware TTL. |
| [AgServe, NeurIPS 2025, Ren et al.](https://papers.neurips.cc/paper_files/paper/2025/hash/04132e28265a355456f86a1c3fec3bcd-Abstract-Conference.html) | Session-aware ETA eviction, plus quality and model selection. | Borrow session/return-time awareness. The main comparison keeps the model and precision fixed. |
| [CacheWise, 2026-06-15 preprint, Tiwari et al.](https://arxiv.org/abs/2606.16824) | Prefix-aware scheduling; tool metadata guiding reuse-aware eviction. | Use historical distributions first. Do not train a complex predictor on only 10 dev tasks. |
| [EfficientAgent, 2026-09-27 preprint, Shao et al.](https://arxiv.org/abs/2609.33762) | The value of offload depends on hardware and on storage that can hold the reuse working set. | Charge save, restore, wasted writes, secondary capacity and foreground interference together. |
| [TOPAS, 2026-08-26 preprint, Ni et al.](https://arxiv.org/abs/2608.25523) | Joint prefix retention, request scheduling and aging. | Joint design is itself studied. Position the contribution on the constrained slot interface, no future information, and platform cost differences. |
| [KVFlow, 2025-07-10 preprint, Pan et al.](https://arxiv.org/abs/2507.07400) | Workflow graphs guide caching and prefetch. | Dynamic single-chain tasks cannot read the replay's full future graph. Future information is only allowed in a clearly labeled oracle. |
| [Preble, ICLR 2025, Srivatsa et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html) | Considers prefix locality and compute load together. | Hit rate is not task benefit. Do not port its multi-GPU architecture. |
| [SGLang, NeurIPS 2024, Zheng et al.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) | RadixAttention manages prefix reuse across calls. | Explain the boundary between slot-level management and radix/block caches; do not over-generalize. |

Suggested contribution statement: **on a unified-memory edge device and a discrete GPU,
measure the costs of retention, checkpointing and incremental recompute; build a slot
management and scheduling policy that uses no future information; and characterize the
limits of its benefit with fairness measurements and ablations.**

## 4. Machine conditions and run arrangements

Read-only check on 2026-11-14:
- Apple M4, 10 CPU cores, 10 GPU cores, 16 GiB unified memory, macOS 15.3;
- the existing `llama-server` is exactly build 11669 / `620d2225c`;
- Python 3.13.12; the GGUF model is local, about 2.3 GiB;
- the repository was clean and the 39 existing tests passed.

At that check the machine was on battery, with about 24 GiB of disk free and about
2.50 GiB of system swap in use. No inference server was started. This is the state of
the interactive environment at the time, not of the historical experiments.

**On the M4:**
- First finish mock tests and minutes-long mechanism probes, then run formal experiments
  on AC power.
- Before timing, leave free memory, keep background load stable, and record initial swap
  and new swap-in/out during the run.
- A run with clear system paging is listed separately and rerun; its log is kept.
- Do not look at server RSS alone: Metal and the file page cache also use unified memory.

The main model, precision, 24K KV and P2 configuration continue from Phase 1. Do not
download a larger model or switch inference frameworks.

**SSD checkpoints:**
- total budget 4 GiB;
- each task keeps only its current valid version, deleted promptly when the task ends;
- record page cache, bytes written, and the fraction of checkpoints never used.

**On the T4**, reuse the verified g4dn environment and setup scripts from the course
account first. Whether an instance is available and how much credit remains were not
checked; do not assume it is online, and do not quote old prices as current. Measure the
time per run first, then set hour and cost caps for the eval sweep. If new T4 experiments
cannot run, the M4 does the whole new-policy evaluation. Historical T4 data then serves
only as motivation, not as cross-platform validation of the new policy.

## 5. Mechanism checks that must pass first

These gates come before the full scheduler. They answer whether resources can be
controlled correctly and whether costs are computed correctly.

| Gate | Experiment | Pass condition and consequence |
| --- | --- | --- |
| A: affinity | P2/K1, two alternating tasks: automatic placement vs explicit `id_slot`; then P2/K2 with waits. | Both slots are used, ownership is correct, and the hit gain reproduces. If this alone gives a large gain, keep the simple scheme as a strong baseline. |
| B: incremental recompute | Fixed ~4.9K shared prefix with private history of 0.5K/1K/2K/4K: keep, overwrite the private suffix, or erase completely. K1 and K2 separately. | A `prefill(total, reused, concurrency)` curve with its error, able to estimate the incremental cost of lost history. |
| C: checkpoint interference | While slot 0 saves/restores 2K/4K/8K, slot 1 decodes continuously; control without I/O. Record per-token stalls, API round trip, completion time and memory. | The global stall and effective cost are known. Writes are not treated as free background work just because Python is async. |
| D: restore correctness | save → overwrite → restore on the same and a different slot; compare 64 greedy tokens under equal conditions; several prompt lengths. | Tokens match, the prefix really hits, no stale state is used. On failure, state can be cleared and recomputed from the full prompt. |
| E: ownership safety | Mock backend injects timeouts, failures and cancellations of completion/save/restore, and early tool returns. | A busy / saving / restoring slot is never reused. No deadlock after failures, never above K, no reservation left after a task ends. |
| F: RAM-cache validity | Three tasks alternate over two slots; historical flags vs idle-slot caching enabled; check save/restore logs, private history hits and memory. | Separates "the policy gives no benefit" from "the mechanism never fired"; gives a fair runtime-cache baseline. |

**What the pinned source shows.** This was checked in the GitHub source:
- `process_single_task` in `server-context.cpp` calls `llama_state_seq_save_file`
  synchronously in the SLOT_SAVE branch.
- The main loop in `server-queue.cpp` processes tasks first, then runs the inference
  update.
- save/restore is also refused during a yield.

So the current implementation cannot be assumed to overlap a save with inference on
another slot. Gate C measures the actual effect on tasks instead of guessing whether
saves are asynchronous.

Sources: [slot operations](https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/server-context.cpp#L2992),
[event loop](https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/server-queue.cpp#L278),
[API docs of the pinned version](https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/README.md).

These APIs release a slot's reusable state and its right of use. The fixed KV pool
allocation does not shrink. Do not write a successful checkpoint as "physical memory
reduced".

## 6. Recommended policy: expiring slot reservations with fair dispatch

### 6.1 Keep task state and cache state separate

**Task states:** `WAITING_FIRST_STEP → READY → RUNNING → EXTERNAL_WAIT → READY … →
FINISHED/FAILED`.

**Cache states:** `RESIDENT / SAVING / CHECKPOINTED / RESTORING / ABSENT`.
- An unfinished task need not stay resident.
- A task waiting on a tool does not hold an execution permit K.

**Per-slot record:** owner, context version, busy flag, known token prefix and lease
expiry.
- Every save/restore is mutually exclusive with completion on that slot.
- `id_slot` is bound only at dispatch, because the slot may change between enqueue and
  execution.
- "Pinning" means the management layer does not hand the slot to another task. It is not
  a call to some nonexistent hardware pin API.

**Uncertain outcomes.** An HTTP timeout or caller cancellation does not mean the backend
operation stopped.
- When the outcome is unknown, mark the slot `UNCERTAIN` and isolate it.
- Reuse it only after confirming that the backend finished and state is consistent, or
  after a controlled restart that clears the mapping.
- Never release the lock on timeout, delete a file that may still be written, and
  immediately submit a recompute.

### 6.2 Keep the queue rule small

**The basic rule.** The first version has two classes of ready request: first steps and
continuations.
- Without a forced-reservation conflict, continuations go first.
- Within a class, requests are ordered by ready time.

**Aging.**
- A request whose ready wait exceeds A gets top priority; overdue requests run in wait
  order.
- A is chosen on dev on the scale of service time, from candidates `2/4/8 × median dev
  step time`, then frozen.
- Aging treats new and old tasks alike. Time in a tool does not count as ready wait.
- Each completed step requeues. The rule never assumes that a task which has run many
  steps is nearly done, and it never reads the true remaining steps.

**No preemption.** The external scheduler cannot preempt a running LLM call.
- A is a priority threshold, not a strict wall-clock bound on waiting.
- Measure the maximum wait and how often overdue requests are still bypassed.
- Report the residual blocking by long requests.

**A simple strong control.** Order by each task's completed model service, smallest
first, also with aging. Call it Agentix-inspired; do not claim a full reproduction.
Check on dev whether it is steadier than continuation-first. Freeze only one as the final
queue rule; report the other as a comparison.

### 6.3 Reservation deadlines come from observable costs

J is not fixed to P. Whenever there is free execution capacity and a ready request:
1. use the task's own still-resident slot;
2. otherwise use an empty slot;
3. compare keeping vs giving way only when a waiting task's slot would be overwritten.

**Probability of return.** If waiting task i has waited a seconds, the conditional
probability that it returns within h seconds is

`p_i(h | a) = [F(a+h) - F(a)] / [1 - F(a)]`.

How F is estimated:
- F comes only from dev waits, or from online waits that have already finished.
- With too few samples, use a pooled distribution and a conservative default.
- Never use the replay's pre-sampled true delay.
- Never estimate the remaining wait of a heavy-tailed distribution as
  `mean wait - waited`.
- When the empirical CDF's tail has too few surviving samples and the denominator nears
  zero, do not extrapolate a return time. Use a predefined conservative short lease or
  no lease, and log the fallback.

**Cost of losing the slot.** `C_loss(i,s)` is the extra prefill i's next call is expected
to pay if slot s goes to the candidate instead of staying with i.
- It is estimated from the known token prefix, from whether the shared prefix can stay in
  that slot, from K, and from the curve of section 5.
- It is not the total prompt length divided by one global speed.

**The decision rule.** A transparent first version compares a few candidate deadlines
`h ∈ {0, Q25, Q50, Q75}`:

`V_keep(i,h) = p_i(h | a) × C_loss(i,s) - L_ready(h)`.

- `L_ready(h)` is the extra wait, in seconds, that holding the slot for up to h seconds is
  expected to impose on the current runnable queue. Work that can run normally on another
  slot is not counted as blocked.
- The first version uses a myopic projection from the current slot availability and
  queue, with only the current request and observed service; no whole future trajectory.

**Acting on it.**
- With a positive value, grant a finite lease. Otherwise pick the victim with the
  smallest loss and let the ready request run.
- Without competition, never erase KV. An expired lease only means the slot may be
  overwritten.
- Re-decide on every new ready event, tool return, step completion or lease expiry.
- An overdue request may break a lease, but never interrupt an operation on a busy slot.

**QC execution order:**
1. serve aging-overdue requests first;
2. without overdue requests, scan candidates in queue-rule order and run the highest
   priority request that can get a slot now;
3. a request blocked by a protected lease stays in the queue, and lower-priority work
   that does not break the lease may run;
4. if every candidate is blocked, wait for the nearest lease expiry or a new event.

Overdue requests may break an idle slot's lease; busy or uncertain slots stay off limits.
If an overdue request cannot get any safe slot, other work may still run on safe slots,
and the reason and bypass count are logged. The C ablation keeps strict FIFO and does not
use this bypass rule.

**Expected effect.** Admission changes with the slot's opportunity cost:
- the M4 should protect expensive history more willingly;
- the T4 should let ready work run more willingly.

The system does not promise a high hit rate; it aims at end-to-end task benefit. The rule
is a heuristic to be tested, not a proven optimum.

### 6.4 A checkpoint is a costed eviction option

**When to consider a save.** Only when a slot must be given up and the current context
version has no valid snapshot yet, so not every tool wait writes a large file.
- The first version allows at most one I/O operation at a time.
- The slot may be overwritten only after the save succeeds.
- Once the task generates more tokens, the old snapshot no longer stands in for the
  latest history.

**Conservative admission condition:**

`p_use × (C_loss - C_restore - I_restore) > C_save + I_save + margin`.

What the terms mean:
- **p_use** is the probability that the snapshot is actually restored before it is
  deleted, invalidated or the task ends.
- **All terms** use the same myopic delay-loss accounting.
- **C_save** counts the direct wait of giving up the slot.
- **I_save and I_restore** count only the extra waiting of other affected tasks; the same
  pause of the same task is never counted twice.
- **Restore and its interference** are weighted by the probability of use.

How to charge the costs:
- I/O round trip and server stall are measured separately.
- Without reliable evidence of overlap, the global save stall is not written off because
  "the tool will wait 10 s anyway".
- A one-time recompute cost is not multiplied by the number of future hits.

**When the task returns,** compare again, including interference with other tasks:
- prefill directly in the selected slot; or
- restore and then compute the new tokens.

Having a snapshot does not force a restore. The save already happened, so it is a sunk
cost and is not added to this restore-vs-recompute decision. Unused snapshots count
toward the wasted-write rate.

**Platforms.**
- The M4 is the main candidate for checkpoints.
- On the T4 the usual private suffix recomputes in under 1 s while a whole-slot save takes
  about 9 s, so the sensible behavior is likely to refuse saving.
- Fall back to recompute, and log the reason, when the model is uncertain, the 4 GiB
  secondary budget is exceeded, memory pressure is too high, or storage fails.

If globally synchronous I/O does not pay off, keep the full main system (affinity +
finite leases + aging) and report the negative checkpoint result as a cost study. Real
chunked, asynchronous KV offload would require changing the runtime and is not a
precondition of this phase.

### 6.5 Online information and capacity checks

**What the policy may see:**
- the current request's prompt / token count and generation cap;
- the service of completed steps;
- the current ready queue and slot ownership;
- external wait events that have already happened.

**What stays with the harness / evaluator only:** the full future `steps`, the next exact
`gap_s`, future outputs, final context, true remaining steps and future arrivals.

Further rules:
- A task ID must not be used to look up its full trajectory.
- An oracle, if built, is listed separately as a non-deployable upper bound.
- Before dispatch, check per-slot capacity from the current prompt and generation cap. Do
  not reserve by reading a future maximum length from the frozen trajectory.
- If a task later exceeds capacity, record a failure or wait under a predefined capacity
  policy. Never truncate context, shorten outputs or skip steps.
- The main experiments keep 12K per slot, which fits every frozen task.

## 7. Implementation files and test boundaries

At planning time these were all **to be added or changed**. `phase1/Fifo` and historical
results stay unchanged, so the original baseline is not altered.

| Location | Work |
| --- | --- |
| New `phase2/state.py` | task/slot/cache state, versions, ownership and invariants |
| New `phase2/costs.py` | prefill/save/restore curves, online EWMA, conditional remaining-wait distribution |
| New `phase2/scheduler.py` | event-driven queue, aging, leases, slot choice at dispatch |
| New `phase2/checkpoints.py` | checkpoint life cycle, total byte budget, failure fallback and I/O mutual exclusion |
| New `phase2/bench.py` | reuses Phase 1 workload functions and ModelClient with a common policy interface; future data stays in the harness |
| New `phase2/summarize.py`, `phase2/plot.py` | paired comparisons, ablations, fairness, decision benefit and overhead |
| New `tests/test_phase2_*.py` | state transitions, error recovery, aging, budgets, oracle isolation, mock end-to-end completeness |

The existing `Fifo` operation only receives the payload, and its metadata is only logged;
a full task policy should not be forced into it. The new adapter keeps a reusable submit
style but passes online-visible events explicitly.

**Run identity.** Run labels and grouping keys must add `policy`, parameter hash, cost
model version, workload hash and seed/trial. Otherwise skip-done and summarize may merge
different policies.

**Decision log.** The decision JSONL records at least:
- time, and the candidate slot, owner and version;
- action, predicted cost and reason;
- actual token reuse;
- I/O bytes and duration;
- decision CPU time.

**Testing order.** Run the affected mock unit tests, then the full existing suite, then a
few real requests. Large performance runs do not replace correctness tests.

## 8. Baselines, ablations and fair comparison

| ID | Policy | Question it answers |
| --- | --- | --- |
| B0 | Native FIFO, P2/K2/J∞, automatic slot placement | Alignment with Phase 1 |
| B1 | Best fixed K/J re-chosen on dev | Is it better than a carefully tuned simple scheme? |
| B2 | FIFO + explicit affinity + idle-slot LRU, J∞ | Is the gain just fixing slot placement? |
| B2F | B2 + best fixed cap chosen on dev (merged with B2 if the best is J∞) | Is it better than the strong fixed baseline with affinity fixed? |
| B3 | FIFO + affinity + fixed TTL | Is cost adaptivity needed? TTL chosen on dev |
| Q | Queue rule + aging, with B2's simple cache/victim rule | Gain of changing only ready order |
| C | FIFO request order + cost lease/victim, no checkpoint | Gain of changing only residency; record the resource waits leases cause |
| QC | Queue rule + cost lease/victim, no checkpoint | Joint effect of queue and cache |
| QC+S | QC + checkpoints enabled by net benefit | Extra value of tiered state saving |

**What stays fixed across the ablations.** Q/C/QC and their common starting point B2 all
use J∞. Affinity, I/O settings, model, K, KV budget and instrumentation stay the same.

**Blocking in C.** If the FIFO head is blocked by a lease in C, report the blocking only;
never silently reorder to later requests. This is also where joint scheduling may add
value. Resource waits caused by the method are kept apart from ordinary queue waits.

**The RAM-cache row.** The runtime RAM cache validated in gate F is one control row, and
Phase 1 flags are kept for historical comparison.
- RAM-cache uses extra RAM, which must count toward total memory.
- Give the configuration and resource breakdown under the same physical cap, so a gain
  from extra secondary capacity is not credited to the policy.
- If the M4 cannot fit this configuration, mark it not comparable and evaluate it on the
  T4. Never fill the table with paging runs.

**How the simple baselines are chosen.**
- B1's first screen: `(K,J)={(1,1),(1,∞),(2,2),(2,3),(2,4),(2,∞)}`, with at least three dev
  mixes covering all 10 tasks.
- When B2F beats the native fixed scheme, it is also a baseline that must be beaten.
- B3's fixed TTL uses a small grid such as 0/1/3/10 s, combined with J∞ and the best fixed
  J, chosen on dev and then frozen. No large hyperparameter search.

**How the results are read.**
- If QC beats only native B0 and not B2/B3, write that affinity or a simple TTL is enough;
  do not claim the joint policy is necessary.
- QC beating Q and C does not by itself show superadditivity. Report a factorial
  interaction with its error on the same latency metric.

## 9. Experiment matrix and sample plan

### 9.1 Main experiments

**Setup.**
- The main model and precision are unchanged: P2/K2/C24576.
- The exception is B1, which may pick K=1 or 2 on dev; report it per platform.
- Keep M4 `scaled:1` (equivalent to `poisson:1`) and T4 `scaled:3.5`, for comparison with
  old results.
- When changing the arrival rate, use `scaled` so that a different rate does not also
  change the arrival pattern.

**Selection and evaluation.**
- **Dev:** screen with three seeds, then confirm two or three finalists with extra dev
  seeds and repeats.
- **Freeze:** after parameters and configuration are frozen, eval uses five new seeds
  written into the manifest in advance, with 20 tasks per run so every eval task appears.
- **Rows:** eight base comparisons plus a working RAM-cache and optional B2F, so 9–10 rows,
  each at least once on five seeds.
- **Repeats:** B0, the strongest fixed baseline and the final joint are repeated to three
  trials on the same seeds, to separate input variation from runtime noise.

That is about 900–1,000 task occurrences in the first round, with repeats extra; they are
never called the same number of independent tasks. If the gates show QC and QC+S behave
identically, report that the checkpoint branch never fires and merge the duplicate runs.
If compute is short, shrink and freeze the matrix before formal eval; never stop based on
whether eval shows a significant gain.

### 9.2 Boundary experiments that must be included

| Axis | Conditions | What to explain |
| --- | --- | --- |
| Arrival | 0.5×, 1×, 2× of the main rate; bursts on at least three pre-registered seeds | Is low load nearly lossless? The trade between cache and waiting under overload |
| Wait | `zero`, `recorded`, `lognormal:2:1`, `fixed:15` | Separate raw call intervals from synthetic waits; find the wait scale where gains appear |
| Mix | Original balanced, short-heavy, long-heavy | Does continuation-first slow some class? The class cutoff is defined on dev |
| Checkpoint | Off, by cost, always-save (small diagnostic only) | Guard against the prior that more caching is always better |
| Prediction error | Pooled distribution vs online update; preset perturbations of cost estimates | Does a simple cost error make the policy degrade? |

**Which policies run each condition.** Each sensitivity condition first compares B0, B1
and the final joint on at least three pre-fixed seeds. The full ablation is added only
when a mechanism explanation needs it. Zero wait usually has no retention window, but a
queue change may still help; zero benefit is not assumed.

**Finite streams.** The main experiments stay finite-stream replays and report the
makespan from first arrival to last completion.
- If budget allows, the T4 adds a 100-task long stream, repeating tasks from the frozen
  pool. This is a larger load, not a larger task source.
- Record warm-up, completions within a fixed window and backlog, then drain.
- Overloaded conditions do not claim steady state.

**More KV does not mean more slots.** With P2 and partitioned KV fixed, raising total C
from 24K to 36K does not add a resident slot. So that comparison cannot argue "with
enough cache it degrades to FIFO".
- A sufficient-capacity control could use P4/C49152/K2 on the T4, keeping 12K per slot.
  State clearly that it changes both the number of resident slots and the KV capacity.
- The M4 does not force a large budget that might page.

### 9.3 Realism and quality

The main body keeps the frozen benchmark-derived replay, so every recorded step and
output length is identical.
- Report completed replay tasks/min; do not call it semantic-success goodput.
- Every run checks expected_steps, actual output tokens, append-only prefixes, failures
  and timeouts.

**A small live Retail correctness check.** Same model, same tasks and tool environment,
same decoding settings: paired trials of FIFO vs the final policy.
- Report tool-argument validity, steps executed, environment success rate and failed
  trajectories.
- Use the course's TritonAI as the user simulator.
- Label new captures apart from the old DeepSeek capture; never silently overwrite the
  old workload.

**Limits of the quality check.** The historical success rate of 3/20 means quality
preservation cannot be claimed lightly.
- Fixed output lengths control only the replay's work, not live semantic equivalence.
- Different batching may also cause numerical differences.
- A small live sample can catch obvious regressions, not prove statistical equivalence.

**An optional realism source.** The [anonymized Claude Code traces released by the
CacheWise authors](https://github.com/cachewise-project/cachewise-coding-traces).
- First check schema, license, available prefix information and context length.
- Extracting only the wait distribution would still add external validity.
- If contexts exceed the current 12K per slot, do not call truncated results a full
  coding-agent reproduction.
- Porting a full coding benchmark is not required in this phase.

## 10. Metrics, statistics and acceptance

**Primary metrics:**
- completions and failure rate;
- makespan / completed replay tasks per minute;
- median and P95 task completion time;
- queueing and admission waits, which must always be reported.

P99 is only descriptive at this sample size, not a strong conclusion.

**Mechanism metrics:**
- unfinished owners overwritten, history reuse, wasted prefill tokens and time;
- real idle time and the delay the policy itself causes;
- save/restore counts and bytes, and the fraction of snapshots deleted unused;
- prediction error, policy CPU overhead and global I/O stall.

Busy time is measured as the union of running intervals or from server iterations and
metrics; overlapping request durations are never summed.

**Fairness:**
`slowdown = task completion time in this run / (isolated pure model time of the same task
+ the run's actual total external wait)`.
- If the isolated log includes tool waits, subtract them first. Platform, model and P/C
  stay the same.
- Report separately:
  - short and long tasks;
  - new-task admission wait and the oldest ready wait;
  - maximum slowdown and aging triggers.
- The short/long cutoff is frozen on dev. Total eval length is used only by the
  evaluator for offline grouping and is never a policy input.
- Jain's index is supplementary only and must not hide individual very slow tasks.

**Memory:**
- fixed theoretical KV budget;
- server/runner RSS, system available memory and swap growth;
- T4 VRAM;
- peak checkpoint files and page cache effects.

Give resource caps together with host-cache schemes. "RSS did not grow" is not "no
memory capacity pressure".

**Statistics.**
- The unit is the paired seed/run, with trials nested inside.
- First compute the same-seed difference or ratio, then summarize the median, range and
  seed-clustered intervals.
- Never treat three trials of the same task as three independent tasks.
- Five seeds still give limited intervals: show every seed, not one average speedup.

**One aggregation definition.** The main table computes metrics per run, averages trials,
then summarizes seeds. Request-level pooled statistics are listed separately. Some
differences between Phase 1 tables and summary CSVs came from exactly this, so Phase 2
states it explicitly.

**Suggested pre-registered engineering targets:**
- on the M4, a reproducible improvement in task latency or throughput over the re-tuned
  strong baseline;
- in the T4 main condition, a very small gain is acceptable, but higher hit rates must not
  buy a clear throughput loss;
- no new correctness failures on either platform;
- aging reduces extreme waits.

A 10% improvement and 5% regression tolerance can be dev-stage goals, but whether they
are met must be judged against the measured noise on this machine; they are not
predicted results.

If the final joint is no better than the best simple policy, deliver the cost model, the
simple scheme, the negative result and its boundaries. That still answers the course's
research question.

## 11. Timeline (26 days)

| Dates | Work and deliverables |
| --- | --- |
| 11/14–11/16 | Gates A–F: incremental cost curves, checkpoint interference, working RAM-cache control, online information boundary; decide whether checkpoints enter the main policy. |
| 11/17–11/20 | Slot state machine, affinity, aging, finite leases; mock and small dev end to end; keep the original FIFO. |
| 11/21–11/24 | Re-select baselines on multiple dev mixes, calibrate the cost model; add budgeted checkpoints; freeze algorithm, parameters, eval seeds and main matrix on 11/24. |
| 11/25–11/30 | M4 and T4 main comparisons and Q/C/QC ablations; automatically log all failures, timeouts and operation costs; finish paired runs. |
| 12/01–12/04 | Load/wait/mix boundaries, a little live correctness; add key-baseline trials as planned. |
| 12/05–12/07 | Figures, mechanism analysis, fairness and negative results; first draft of the final report. |
| 12/08–12/09 | Freeze the submission commit, regenerate all tables and figures, spot-check every number, collect reproducible commands and an artifact map. |
| 12/10 | Submit code, workloads, manifests, raw logs, analysis scripts and the final report. |

Time estimates must be calibrated with the first pilot. At historical rates of about 0.59
tasks/min (M4) and 2.13 (T4), 900–1,000 task occurrences need about 25–28 h and 7–8 h;
more trials and sensitivity add to that, and longer streams may run at different rates.
Reserve several AC-powered nights on the M4. On the T4, confirm course credit and
instance availability before setting a budget; do not commit to costs from unverified
historical prices.

**Priorities.**
1. Affinity, correct costs, strong baselines and the full ablation.
2. Then checkpoints.
3. Only then coding traces or runtime changes.

If the net benefit of checkpoints is still unproven by 11/24, freeze them as an
automatically disabled branch so they do not delay the main evaluation.

## 12. Order of work and final deliverables

**First day.** Run "4.9K shared prefix + private suffix recompute" and "does slot 1 stall
while slot 0 saves". Then do the affinity gate. These directly decide the policy's costs
and scope, so a complex scheduler is not needed first.

**Where runs go.** Every new model experiment writes to its own `runs/phase2-.../` folder
(now `runs/phase2/<n>_...`). `runs/final` and `runs/aws-gpu` stay unchanged. Every run
saves:
- the command, commit and versions;
- the model and workload hashes, and the parameters;
- power state, machine and budget.

Once the new CLI is written and tested, its exact commands go into the README; this plan
does not give `phase2` commands that do not yet exist.

**Report structure.** The final report is suggested to center on six figures:
1. platform incremental costs and I/O interference;
2. end-to-end results against strong baselines;
3. queue/cache/checkpoint ablations;
4. fairness distributions;
5. benefit boundaries across arrival and wait scales;
6. action overhead and misprediction.

4–6 pages is a writing target, not a hard limit derived from the course files.

**What this round delivered.** Only this document. The 39 existing tests passed in
3.57 s. The scheduler was not changed, no inference sweep was rerun, no model API was
called and no cloud resource was rented. The first implementation step should be the
observable mechanism probes and slot state management.

## 13. 2026-11-16 review: gaps found and changes made

This section was written by the second code and design review. It checks the `phase2/`
implementation against the course Option 1 requirements, the professor's feedback, the
Phase 1 report and the original memo. All changes below were implemented with model-free
tests (188 tests passed); **no real-model experiment had been run yet.**

### 13.1 Problems that would make comparisons unfair or conclusions wrong

| Problem | Consequence | Change |
| --- | --- | --- |
| Under FIFO order, when the head is blocked by some lease, the lease owner's own returning request queues behind the head and is blocked too. | The TTL baseline (B3) and the cache-only ablation (C) would wait for their own lease to expire, becoming artificially weakened baselines and exaggerating joint's advantage. | When the head is blocked, the lease owner may use the slot reserved for it (the head cannot use it anyway); logged as `reason=lease_owner`. |
| joint-checkpoint's lease compared only against recompute. | When saving is cheap (M4), the lease still protects the slot against expensive recompute, so checkpoints almost never fire; the memo's keep / save / evict choice was not really implemented. | The compared cost is now `min(recompute, save+restore)` (if the profile has save/restore and the checkpoint budget has room); the decision log records `eviction_alternative`. |
| `slot erase` does not clear llama.cpp's host prompt cache. | The second and later runs of the RAM-cache control would start from the previous run's warm cache, making the same tasks look faster. | The executor requires `--restart-server ram-cache` when there is more than one host-cache run, restarting the backend before each run. |
| Policy order rotated only by trial. | With 1 trial, every seed has the same order; fanless M4 throttling would systematically favor policies that run first. | Rotate by `seed_index + trial`. |
| Runs did not check the power source. | Phase 1 once had a battery run moved out and rerun; Phase 2 had no such guard, and the machine was on battery. | Real runs and probes refuse on battery. Metadata records `pmset -g batt/therm`, and on the T4 `nvidia-smi` memory and temperature. Battery runs are never paired. |
| Prefill calibration included the retain condition (about 2 tokens) in the per-token median. | The fixed overhead spread over 2 tokens could overestimate recompute and over-protect with leases. | Fit only the overwrite / erase conditions. |

### 13.2 Course requirements that were not covered

| Requirement | Previous state | Change |
| --- | --- | --- |
| Report peak memory | `resources.jsonl` had samples, but no summary table had a memory column. | Run/policy summaries add peak server RSS, minimum system available memory, swap-out growth and GPU memory. |
| Test whether long tasks are unfairly delayed | Slowdown needs the K=1/J=1 isolated reference, which eval/sensitivity manifests did not generate. | eval/sensitivity add an `isolated` run first (1 seed, 1 trial); mix workloads reuse the source tasks' reference through `reference_workload_hash`. |
| Vary the workload mix | Only arrival rate and wait conditions; no mix. | New `mix-short` / `mix-long`: classify by the frozen cutoff; the favored class appears 3× in the pool (about 75/25); the file hash is written into the manifest. |
| Gains must not come from lower quality | Only steps and output lengths were checked. | Every request and task records a digest of generated tokens; summaries report the fraction matching the isolated run (should match under greedy decoding; batching numerics make it a diagnostic, not a gate). |
| Explain results with observations | Summaries had only the total number of decisions. | Add lease / no_lease / evict / aged dispatch / lease-owner dispatch / save / restore counts, total lease time, mean and max ready wait, admission wait and unused snapshots. |
| Compare with the best fixed baseline chosen on dev | Pairing supported one baseline matched by policy name. | Run records carry the manifest label `policy_label`; `--baseline` can repeat (e.g. fifo, best-fixed, affinity). |

### 13.3 Parameter selection and budget

- **The aging threshold was not tuned.**
  - The default of 30 s means very different things on the M4 (step about 9.5 s, about
    300 s FIFO queue wait) and the T4 (step about 1.9 s).
  - In joint it effectively decides how long a new task can be blocked by leases, so it
    is the main knob between throughput and short-task fairness.
  - Dev adds `--aging-grid`: 20/40/80 s suggested for the M4, 4/8/16 s for the T4.
- **The selection rules are written as code.** `phase2/freeze.py` fixes the rules before
  eval, and its output includes the reasons for each choice:
  - best-fixed is the fixed `(K,J)` or FIFO with the highest dev throughput;
  - B2F is kept only if it beats best-fixed;
  - TTL is the fastest one;
  - attained replaces continuation only if it is faster;
  - aging is the fastest whose short-task slowdown is no worse than best-fixed's,
    otherwise the fairest;
  - the short cutoff is the dev median.
- **The dev matrix was deduplicated and strengthened.**
  - Removed `fixed-k2-jinf` (identical to FIFO) and `ttl-0` (identical to affinity).
  - Added affinity-fixed `J=2,3`, which the plan defined as B2F but never generated.
- **Wait-distribution prior.** In the original flow, dev cost policies started with an
  empty wait distribution, so the first few contentions granted no lease, while eval's
  profile contained dev waits. Now one dev FIFO run on a separate seed collects waits
  before dev, and dev and eval use the same profile.
- **Duration.**
  - The original example (3 trials each for dev and eval, 5 seeds, every sensitivity
    condition) is about 90 h on the M4, which is not feasible on a daily-use 16 GB laptop.
  - New options: `--trial-labels` adds trials only to key rows; `--only-labels`,
    `--conditions` and `--minutes-per-task` give time estimates.
  - The staged plan is in [2_RUNBOOK.md](2_RUNBOOK.md) section 0: the M4 runs probes,
    dev, eval and key trials (about 45 h); the T4 runs everything (about 18–26 h).

### 13.4 Limits that remain and must be stated in the report

- The external scheduler cannot preempt a running request. Aging is only a priority
  threshold, not a wait bound.
- llama.cpp's slot save/restore runs synchronously in the main loop, so the whole server
  stalls during a save. Only "synchronous whole-slot saving" can be evaluated here, not
  the ceiling of chunked asynchronous offload.
- Releasing a slot does not shrink the KV pool. A checkpoint buys the right to use a slot,
  not lower physical memory.
- Eval tasks were already used in Phase 1. New seeds provide only new permutations,
  arrivals and waits. Cross-task generalization needs separately frozen unused tasks.
- The planned small live Retail correctness check is still not implemented. Replay with
  fixed output lengths plus output-digest agreement checks that replay work and state
  restore are correct, but cannot prove live semantic quality is unchanged.

### 13.5 2026-11-17, first night on the M4: measurements and fixes

**Gates A–D passed:**
- explicit `id_slot` uses both slots, with full hits on return;
- private-suffix recompute costs about 6.1 ms/token at K=1 and 6.6 ms/token at K=2;
- after same-slot and cross-slot restore, 64 greedy tokens all match (16/16);
- during save/restore the other slot's largest token gap equals the I/O time (up to about
  1.3 s), which confirms the synchronous global stall.

**Scheduler bug (fixed).**
- After a step ended, the dispatcher ran before replay called `tool_start()`, saw the owner
  as "not waiting", and evicted without pricing.
- In one dev run of cache, 19 of 23 evictions were like this, with only 2 lease decisions.
- The fix treats the end of a step as the start of the external wait. Regression test:
  `test_slot_is_protected_from_the_moment_its_step_ends`.
- Affected runs were moved to `runs/phase2/1_m4_pilot/invalid/2026-11-17-lease-race/`.

**Method fix (dev stage).**
- Lease candidate deadlines were the 25/50/75th percentiles of remaining wait, at most
  about 3.0 s on the M4. That missed the wait tail: on dev a fixed 10 s TTL hit 99%, the
  cost lease only 90%.
- Candidates now include the 90th percentile (`LEASE_QUANTILES`). For expensive history
  (≥20 s) the lease grows from about 3.0 s to 4.7 s; decisions on cheap history are
  unchanged.
- The 5 cost-lease runs of seed 101 were moved to `invalid/2026-11-17-lease-q90/` and
  rerun; the old results are recorded in that folder's README.

**Timing.** A measured dev run (10 tasks) averages 13.4 minutes, so the campaign estimate
changed from 1.7 to 1.4 minutes per task. This only affects whether the next run starts
before the deadline.

### 13.6 2026-11-17/18: revision after the three dev seeds (M4)

All 57 normal dev runs and 3 RAM-cache runs finished without failures. Mean task
completion time over the three seeds:

| Policy | Mean task completion time |
| --- | ---: |
| affinity-fixed-j2 | 129 s |
| fixed-k2-j2 | 136 s |
| joint-checkpoint | 139 s |
| ttl-10 | 173 s |
| cache | 180 s |
| joint-a80 | 206 s |
| fifo | 275 s |

Seeds 102 and 103 are much more contended: fifo recomputes 98K and 83K tokens.

**Finding 1: the cost lease is worse than the 10 s TTL tuned on dev.** The logs show two
reasons:
- **The holding cost was overcharged.** The formula `p(h) × loss − w × h` charges the whole
  h, but the owner usually returns soon (dev median wait 1.7 s), so the slot is idle only
  for E[min(remaining wait, h)]. Section 6.3 of the plan already required charging the
  waiting that is actually blocked.
- **The loss was undercharged.** It counted only the owner's recompute, not that the slot is
  busy during the recompute and the queue is blocked too. Result: on seed 103, cache made
  37 non-overdue evictions (48K tokens recomputed); ttl-10 made 1.

**Revision (v2, `phase2/costs.py`):**
- The value is now `p(h) × (1 + w) × loss − w × E[min(remaining wait, h)]`. Both sides
  count seconds of delay for the owner and the w waiting requests.
- Candidate deadlines add the 99th percentile. The tail is thin, so covering it adds
  almost no idle time.
- The lease cap for the cost policies (cache, joint, joint-checkpoint) goes from 10 s to
  60 s, as a safety cap only. The other policies' configs and param hashes are unchanged,
  so their dev runs stay valid.

With the dev wait distribution, v2 gives history costing ≥2 s to recompute a lease of
about 9 s (covering 99% of returns), and gives up history costing ≤1 s immediately. The 15
cost-lease runs of the three seeds were moved to
`runs/phase2/1_m4_pilot/invalid/2026-11-18-lease-v2/` and rerun; the old results stay as a
v1 ablation.

**Finding 2: under heavy load, evictions come mostly from overdue requests breaking
leases.**
- On seed 103, 38 of ttl-10's 39 evictions were triggered by requests waiting longer than
  the aging threshold (30 s).
- With J∞, queued requests become overdue and break leases.
- J=2 keeps extra tasks at admission, unaffected by aging, so it is fastest under heavy
  load but has the worst short-task slowdown (6.5–7.0, vs 3.8 for joint-checkpoint).

This fairness–throughput trade-off must be explained separately in the report.

**Finding 3: machine temperature.**
- The first run of each session is about 15–30% faster than later ones. The fanless M4 is
  cool after idling: the first prefill is about 3.4 ms/token, about 4.6–5.4 ms/token once
  thermally steady.
- freeze compares only candidates of the same kind, so it is not affected.
- From eval on, each session runs an unmeasured replay of about 6 minutes before the
  first timed run (`WARMUP_MINUTES` in `campaign.py`).

**New pre-registered rule and matrix changes:**
- freeze adds the rule **final**: among cache, joint at the chosen aging, and
  joint-checkpoint, take the fastest whose short-task slowdown is no worse than the best
  fixed baseline's. This is "our system" for the added eval trials and sensitivity.
- Sensitivity adds the frozen fixed TTL, and a gap-long condition (15 s per wait). The
  10 s TTL tuned on dev expires before every return there, so only this condition tests
  whether the cost lease adapts to waits.
- The sensitivity matrix becomes 6 conditions × 4 rows × 2 seeds, 48 runs.
- RAM-cache row: server RSS 6.2 GiB, minimum system available memory 1.5 GiB, a little
  swap, and slower decode too. Per section 8, it is marked as a resource-incomparable
  control on the M4.

### 13.7 2026-11-18 revision 2 (fixes after review, not yet measured at the time)

The three P1 issues of the [review](3_PILOT_REVIEW_2026-11-18.md) are fixed. The new
campaign uses `runs/phase2/2_m4_campaign` (originally `runs/phase2-m4-air-16g-r2`); pilot
results are exploratory evidence only.

- **The checkpoint quote matches its execution.**
  - The quote now includes the net-benefit condition with queue pressure, use
    probability and margin, the same judgment as the actual save. An existing snapshot
    is charged only for restore.
  - Like joint, each context version is priced once. It is repriced in only two cases:
    the version was priced as a checkpoint but can no longer be saved at execution
    (conditions, budget or owner state changed), or the save was refused.
  - When repriced, it is re-evaluated against recompute. If needed, the slot goes back to
    the returned owner first and the candidate request is deferred.
  - Without an available checkpoint, joint-checkpoint decides exactly like joint
    (regression test `test_checkpoint_policy_without_a_checkpoint_evicts_like_joint`).
- **The I/O price is calibrated online.** In the pilot, save+restore was quoted at about
  0.54 s but measured at about 0.99 s during runs: probes ran on an idle server, while
  under load the call also waits for the batch in progress. Now every successful
  save/restore updates its slope by EWMA from the store's measured backend call time
  (`online_io`). Uncalibrated curves stay empty.
- **Online prefill** updates only from samples with at least 128 computed tokens. The
  threshold came from the pilot's shadow analysis and was fixed before r2 dev.
- **Counting.**
  - Repriced lease/no_lease rows carry `reconsidered`.
  - no_lease is not double counted, but leases actually held are counted as usual.
  - Deferred dispatch attempts do not count toward dispatches, ready wait or aged counts,
    nor as a first dispatch.
- **Dev in two steps.**
  1. Baselines and the joint aging screen.
  2. At the chosen aging, confirm queue, cache, checkpoint and TTL 10/15/30.

  freeze accepts only measured rows whose full parameter hash matches the eval
  configuration. r2 no longer tests attained; the queue rule is fixed to continuation.
  TTL candidates change from 1/3/10 to 10/15/30.
- **Eval and sensitivity.**
  - The all-policy T0 table and the core repeat table are paired separately.
  - The core is fifo, best-fixed, affinity-fixed-best and the final system, with 1 extra
    trial on the M4 (trials 0, 1).
  - Sensitivity covers only gap-recorded and gap-long, with rows fifo,
    affinity-fixed-best (best-fixed if freeze did not keep it), the final system and the
    fixed TTL.
  - The new eval seeds 1101–1103 are only new permutations of the same 20 eval tasks.
  - A failed warm-up is retried once.
- **Reusing the pilot (`init --reuse-from`).**
  - **What is imported unchanged:** probes, wait distribution, profile, and the 30 dev runs
    of fifo, fixed K/J, affinity, affinity-fixed and RAM-cache.
  - **Why that is safe:** these policies never price a lease, never checkpoint and never
    read the cost model, and no code change after the pilot's first dev run touches
    their path. The import requires identical llama-server, model, Python, packages and
    workload, or it writes nothing. No imported run was a cold-machine start (checked
    against every run's start time).
  - **What r2 reruns:** only joint's aging grid and the confirmation group.
  - **The one cross-session comparison.** freeze's choices all compare like with like,
    except the fairness gate: r2's joint rows against the pilot's best-fixed slowdown.
    The report must say so.
  - **Time:** about 19 h for the required part, about 30 h for everything.
