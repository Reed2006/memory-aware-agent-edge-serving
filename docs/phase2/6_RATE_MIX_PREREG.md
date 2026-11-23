# Phase 2 supplement: arrival rate and workload mix (pre-registration)

Written 2026-11-22 before any run of this supplement started. This file and the four
manifests listed below were committed in `308aa6a` at 20:33 PST; the driver started its
first (warm-up) replay at 20:34. (The first version said "about 20:50"; that estimate was
wrong and is corrected here. Paths and predictions are unchanged.)

This supplement does not change the pre-registered Phase 2 analysis:
- the eval confirmatory table stays as it is;
- the frozen final policy stays `joint`;
- `joint-checkpoint` (jc) stays exploratory.

## Why

The course's Option 1 asks Phase 2 to "evaluate changing arrival rates and workload
mixes, including cases where the policy provides little benefit". The original plan
(`1_PLAN.md`, section 9.2) listed both axes. Revision 2 dropped them to save time, so the
sensitivity stage only varied tool waits (gap-recorded, gap-long). Every Phase 2 run so
far uses the same arrival rate (`scaled:1`) and the balanced 20-task pool. This
supplement fills that gap.

## Design

**Seeds: the eval seeds 1101, 1102, 1103.**
- Their 1× results (`scaled:1`, balanced pool, `lognormal:2:1` gaps) already exist for
  all four policies as eval trial 0, with identical parameter hashes. Each seed therefore
  gets a three-point rate curve (0.5×, 1×, 2×) and a mix comparison (short-heavy,
  balanced, long-heavy) without rerunning the 1× cells.
- Three seeds meet the original plan's minimum for a sensitivity condition.

**Policies.** All four are the frozen eval configurations, with the same hashes:

| Label | Param hash | Role |
| --- | --- | --- |
| `fifo` | 42eadd78 | Phase 1 baseline |
| `affinity-fixed-best` (B2F) | 654f202a | Strongest simple baseline: J = 2, affinity, LRU, never preempts |
| `joint` | 4ed3bc99 | Pre-registered final |
| `joint-checkpoint` (jc) | a60b4703 | Exploratory: `joint` plus SSD checkpoints |

**Conditions.** Each condition changes one factor of the 1× eval setup:

| Condition | Change | Manifest |
| --- | --- | --- |
| arrival-double | `scaled:2`: the same arrival pattern compressed to twice the rate | `manifests/1_arrival_double.json` |
| mix-long | Long tasks appear 3× in the task pool, about 75% of draws | `manifests/2_mix_long.json` |
| mix-short | Short tasks appear 3× in the task pool, about 75% of draws | `manifests/3_mix_short.json` |
| arrival-half | `scaled:0.5`: the same arrival pattern stretched to half the rate | `manifests/4_arrival_half.json` |

More about the conditions:
- **Short and long** use the frozen dev cutoff of 782.5 expected output tokens. It splits
  the 20 eval tasks 10/10.
- **The mix workloads** are written by `phase2.plan create`:
  - `workloads/mix-short-f1eabdce66ec.json`, sha256 e3342197…
  - `workloads/mix-long-f1eabdce66ec.json`, sha256 61eb926e…
  - Each mix occurrence replays the identical recorded trajectory. Slowdown reuses the
    source task's isolated timing.
- **What the draws contain:**

  | Workload | Seed 1101 | Seed 1102 | Seed 1103 |
  | --- | ---: | ---: | ---: |
  | mix-short (short tasks of 20) | 17 | 17 | 15 |
  | mix-long (short tasks of 20) | 3 | 3 | 5 |

**Order.** The heavy conditions run first, so the first night yields the trade-off
evidence: arrival-double, mix-long, mix-short, arrival-half. Inside each manifest, the
policy order rotates by seed. A run starts only if its estimate ends before the
operator's deadline, and the runner skips finished runs on later nights.

**Everything else is as in eval:**
- workload `runs/final/workload_eval.json`, 20 jobs, gap `lognormal:2:1`;
- P = 2, ctx 24576, cache-ram 0, no idle-slot cache, same model and server build;
- cost profile `runs/phase2/2_m4_campaign/profiles/profile.json`;
- one unmeasured warm-up replay per session;
- AC power only; the 30 locked sources must match the campaign lock;
- one trial per cell.

**A priori load.** This is the indicator from `scripts/phase2/3_frontier_seeds.py`: the mean
admission wait of a 2-server FCFS run-to-completion queue, computed from arrivals, job
order and the cost model only. For reference, the earlier seeds scored heavy 1101 (50 s)
and 2102 (44 s), and light 1102 (4 s), 1103 (8 s) and 2101 (9 s).

| Seed | 0.5× | 1× | 2× | mix-short | mix-long |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1101 | 17.5 s | 49.6 s | 152.8 s | 12.6 s | 66.3 s |
| 1102 | 0.0 s | 4.0 s | 18.1 s | 2.3 s | 6.1 s |
| 1103 | 2.0 s | 7.9 s | 18.3 s | 3.6 s | 10.5 s |

So 2× makes seed 1101 very heavy and raises 1102 and 1103 to a moderate load. mix-long is
heavy only on seed 1101. Half rate and mix-short are light everywhere.

**Expected cost.** About 24–25 h for all 48 runs:
- arrival-half takes about 9.3 h, because its runs are bound by the stretched arrival
  span;
- the other three conditions take about 4–7 h each.

## Metrics

The metrics are those of the frontier supplement (`5_FRONTIER_PREREG.md`):
- **Efficiency:** tasks/min and mean JCT.
- **Fairness:**
  - P95 slowdown;
  - mean short-task and long-task slowdown;
  - Jain (read together with its δ = 30 value);
  - maximum admission wait;
  - contention delay (latency minus isolated time), mean and P95.
- **Mechanism:** wasted prefill, evictions, saves and restores.

The 1× cells come from `runs/phase2/2_m4_campaign/eval` (trial 0).

## Predictions

Each prediction holds if it is true in at least 2 of 3 seeds. Here R(c) is the fifo /
B2F mean-JCT ratio in condition c.

- **R1, low load gives little benefit.** At 0.5×:
  - B2F, joint and jc have mean JCTs within 10% of each other;
  - R(0.5×) < R(1×).
- **R2, the benefit of keeping KV grows with load.** R(2×) > R(1×).
- **R3, the frontier trade-off appears under load.** At 2×, jc's P95 slowdown is lower
  than B2F's. Its mean JCT is no lower than B2F's.
- **R4, the eval negative result holds under overload.** At 2×, joint's mean JCT is
  higher than B2F's.
- **R5, a long-heavy mix behaves like more load.** Both hold:
  - B2F's maximum admission wait is larger in mix-long than in mix-short;
  - R(mix-long) > R(mix-short).

These are descriptive, with no direction predicted:
- **Long tasks.** Long-task slowdown and long-task P95 contention delay, B2F vs jc, in
  each condition. This answers the course question of whether long tasks are unfairly
  delayed.
- **Absolute delay.** B2F vs jc P95 contention delay in the heavy cells (2×, and
  mix-long on seed 1101).
- **Pooled paired ratios.** These come from `phase2.summarize`, with a seed-cluster
  bootstrap, within each condition.

## How the outcomes are read

| Outcome | Reading |
| --- | --- |
| R1 holds | "Little benefit at low load" is shown directly, as the course asks. |
| R1 fails | Keeping KV still matters when lightly loaded. Report it as a benefit at low load. |
| R2 and R5 hold | The benefit scales with load, whether load comes from the arrival rate or the job mix. |
| R3 holds | The jc vs B2F trade-off from the frontier seeds generalizes from heavy seeds to heavy rates. |
| R3 fails with jc faster | jc dominates under overload. |
| R4 holds | The negative result for the final policy is robust to the arrival rate. |

## Limits

- **One trial per cell.** Trial noise is about 10% for the sharing policies and about 3%
  for the J-limited ones. A single-seed difference below about 10% is not read on its
  own.
- **The 1× cells ran in an earlier session** (2026-11-19 and 2026-11-20). Comparisons
  across conditions mix a condition effect with a session effect. Comparisons between
  policies within a condition do not.
- **Session splits.** If the 48 runs need more than one night, later conditions run in a
  later session. Each session starts with a warm-up.
- **Mix composition.** mix-long has only 3–5 short tasks per run, so its short-task
  slowdown is noisy. mix-short has only 3–5 long tasks, so its long-task numbers are
  noisy.
- **Same task pool.** These are new permutations and rates of the same 20 eval tasks.

## Files

| What | Where |
| --- | --- |
| Driver | `scripts/phase2/4_rate_mix_run.sh` (default deadline 09:00; `touch runs/phase2/4_m4_rate_mix/STOP` stops new runs) |
| Prediction check | `scripts/phase2/4_rate_mix_check.py` |
| Results | `runs/phase2/4_m4_rate_mix/sensitivity/<condition>/<policy>/S<seed>-T0` |
| Summary | `runs/phase2/4_m4_rate_mix/sensitivity/summary/` |
