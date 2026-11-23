# Phase 2 supplement: fairness–efficiency frontier (pre-registration)

Written 2026-11-21, about 00:50 PST, before any run of this supplement started.
(2026-11-22: file paths updated for the numbered `runs/phase2/` and `scripts/phase2/`
layout; nothing else changed.)
This is an exploratory supplement. It does not change the pre-registered Phase 2
analysis: the eval confirmatory table and the sensitivity conditions stay as they are,
and the frozen final policy stays `joint`.

## Why

The existing runs (dev 57, eval 43, sensitivity so far) show two things. Both were
noticed after looking at the data, on the same seeds that chose the configurations.
This supplement tests whether they hold on seeds that have never been run.

1. **Across policy families there is no trade-off.** Policies that waste prefill
   (fifo, affinity, queue, fixed or affinity-fixed with J ≥ 3) are worse on both
   efficiency and fairness. Avoiding recompute improves both at once.
2. **Among the efficient policies, a trade-off appears only under contention.**
   - On the contended seeds (dev 102 and 103, eval 1101, sensitivity 2102),
     admission-limited run-to-completion (`affinity-fixed-best`, J = 2) has the best
     throughput and mean JCT.
   - On those same seeds, all-admitted sharing with cost leases and KV checkpoints
     (`joint-checkpoint`) has a lower P95 slowdown, a lower short-task slowdown and a
     higher Jain index. Example: eval S1101, B2F JCT 161 s, P95 slowdown 7.3,
     Jain 0.728; joint-checkpoint JCT 186 s, P95 slowdown 6.0, Jain 0.750.
   - On light seeds, every efficient policy ties.

The proposed mechanism has two parts:
- J = 2 never preempts, so it wastes nothing, but later arrivals wait in the admission
  queue: the maximum admission wait is 100–500 s on contended seeds.
- Sharing admits everyone but pays for preemption. Without checkpoints that payment
  is recompute; with checkpoints it is a save/restore of about 0.8 s.

## Design

**Seeds: 3116, 3129, 3120.**
- `scripts/phase2/3_frontier_seeds.py` chose them before any run: they are the top 3 of
  the never-run seeds 3101–3130.
- The ranking uses an a priori load indicator: the mean admission wait of a 2-server
  FCFS run-to-completion queue serving each job for its cost-model isolated time.
  It uses arrivals, job order and the cost profile only, no measured outcome.
- The indicator separates the seeds already run: heavy 1101 (50 s) and 2102 (44 s);
  light 1102 (4 s), 1103 (8 s) and 2101 (9 s).
- The chosen seeds score 87 s (3116), 50 s (3129) and 48 s (3120).
- The selection picks a load regime, the one where a trade-off can exist. It does not
  pick a result.

**Policies.** All three are identical to the eval configurations, with the same
parameter hashes:

| Label | Param hash | What it is |
| --- | --- | --- |
| `affinity-fixed-best` (B2F) | 654f202a | J = 2, LRU, FIFO; never preempts |
| `joint` | 4ed3bc99 | Pre-registered final: all admitted, continuation queue, cost lease, aging 80 s |
| `joint-checkpoint` (jc) | a60b4703 | `joint` plus KV save/restore to SSD, 4 GiB budget |

**Everything else is as in eval:**
- workload `runs/final/workload_eval.json`, 20 jobs; every seed is a permutation of
  the same 20 tasks;
- gap `lognormal:2:1`, arrival `scaled:1`;
- P = 2, ctx 24576, cache-ram 0, no idle-slot cache, same model and server build;
- cost profile `runs/phase2/2_m4_campaign/profiles/profile.json`;
- one unmeasured warm-up replay before the first measured run;
- policy order rotated per seed, as the manifest writes it, with one trial per cell.

**Files.**
- Manifest: `runs/phase2/3_m4_frontier/manifests/frontier.json`, written by
  `phase2.plan create`.
- Results: `runs/phase2/3_m4_frontier/eval/`.
- Driver: `scripts/phase2/3_frontier_run.sh`.
- Runs start only after the main campaign session has ended.
- A run starts only if its 30-minute estimate ends before 09:00. Seeds run in order, so
  a cut leaves whole seeds.

**Slowdown references.** The isolated `fixed-k1-j1` run of the main eval
(`--reference-dir runs/phase2/2_m4_campaign/eval`). It covers the same 20 tasks.

## Metrics

- **Efficiency:** tasks/min and mean JCT.
- **Fairness:**
  - Jain index over 1/slowdown;
  - P95 slowdown;
  - mean short-task slowdown (frozen cutoff of 782.5 expected output tokens);
  - maximum admission wait.
- **Mechanism:** wasted prefill fraction, evictions, saves and restores.

## Predictions

Each prediction is checked per seed and holds if it is true in at least 2 of 3 seeds.

- **P1 (sharing costs efficiency):** B2F mean JCT < jc mean JCT.
- **P2 (sharing buys fairness):** jc P95 slowdown < B2F's, and jc Jain > B2F's.
- **P3 (checkpoints lower the price of sharing):** jc wasted prefill < joint's, and jc
  mean JCT < joint's.
- **P4 (eval negative result replicates):** joint mean JCT > B2F mean JCT.

How the outcomes are read:

| P1 | P2 | Reading |
| --- | --- | --- |
| holds | holds | A real trade-off between run-to-completion admission and checkpointed sharing |
| fails | holds | jc dominates; no trade-off |
| holds | fails | B2F dominates; no trade-off |
| fails | fails | No consistent ordering |

Pooled paired ratios over the 3 seeds, with a seed-cluster bootstrap, are reported as
description, not as a test.

## Limits

- **One trial per cell.** Repeats on eval S1101 put trial noise at about 10% for
  `joint` and about 3% for the J-limited policies. A single-seed difference smaller
  than about 10% is not read on its own.
- **Three heavy seeds.** The result is conditional on contention and is not an
  average-case claim. On light seeds all efficient policies tie.
- **No change to the final.** This supplement does not make jc the Phase 2 final.

## Offline analyses (no new runs)

- **A. Frontier.** For every seed already run, the Pareto set on (mean JCT, P95
  slowdown) and on (throughput, Jain).
- **B. Load dependence.** How the B2F-vs-sharing gaps grow with the a priori load
  indicator.
- **C. Reference sensitivity.** The isolated reference is warm for the shared
  4891-token prefix. Add a cold prefix prefill to each job's reference and check
  whether the fairness rankings change.

## Addendum (about 00:55 PST, still before any supplement run)

Analysis C on the existing runs changed how the fairness metrics should be read.

**The cold prefix is expensive.** Prefilling the 4891-token shared prefix cold costs
about 30 s under the cost model, which is comparable to a short job's whole isolated
time (35–60 s).

**Shifting every reference by δ = 30 s changes the fairness picture:**
- slowdowns compress by 3–5×;
- **Jain over 1/slowdown reverses rank** on several contended seeds. On eval S1101,
  jc is above B2F at δ = 0 but below it at δ = 30;
- the **order on P95 slowdown and short-task slowdown mostly survives, with smaller
  gaps**.

So the pre-registered metrics stay primary, read at δ = 0, the measured warm reference.
Within a run only the first job on each slot starts cold, and that is the same for
every policy. Three changes follow:

1. Jain is read only together with its δ = 30 value. A Jain difference that changes
   sign between δ = 0 and δ = 30 is reported as not robust.
2. A reference-free fairness metric is added: **contention delay**, latency minus
   isolated time (compute plus actual gaps).
   - A shift of the reference moves it by the same constant for every policy, so
     comparisons between policies do not depend on δ.
   - It is reported as the mean and P95 over all jobs, and the P95 over short jobs.
3. P2 is also checked with the P95 contention delay in place of P95 slowdown. It is
   reported separately. It does not replace the pre-registered check.

**Contention delay on the existing runs points the other way from slowdown.**

| Seed | B2F P95 delay | Sharing P95 delay | B2F max admission wait |
| --- | ---: | ---: | ---: |
| eval S1101 | 187 s | jc 327 s, joint 302 s | 177 s |
| sensitivity S2102 | 220 s | fixed-ttl 588 s, joint 769 s | 191 s |
| dev 103 | 135 s | 211–251 s | 111 s |

- Every short or long job under B2F waits about the same absolute time, in the
  admission queue. That costs short jobs proportionally more, so B2F loses on slowdown.
- Sharing finishes short jobs early and pushes the delay onto long jobs, so it wins on
  slowdown and loses on absolute tail delay.
- So part of the "fairness vs efficiency" trade-off is a trade-off between two
  fairness notions: proportional (slowdown) and absolute (delay).

One more prediction, written before any supplement run:

- **P5 (absolute-delay fairness favors run-to-completion):** B2F P95 contention delay <
  jc P95 contention delay, in at least 2 of 3 seeds.
