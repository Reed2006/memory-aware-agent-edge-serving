# Phase 2: where fairness and efficiency trade off

Analysis of every completed Phase 2 run on the M4 Air (16 GB). Written 2026-11-21;
section 6 added 2026-11-24.

| Source | Runs |
| --- | --- |
| dev | 57 |
| eval | 43 (T0 + T1) |
| sensitivity | 16 (gap-recorded and gap-long) |
| frontier supplement | 9 ([5_FRONTIER_PREREG.md](5_FRONTIER_PREREG.md)) |
| arrival-rate / mix supplement | 48 ([6_RATE_MIX_PREREG.md](6_RATE_MIX_PREREG.md)) |

Sections 1–4 are post-hoc readings of data that also chose the configurations.
Sections 5 and 6 are pre-registered supplements: new seeds (5) and new arrival rates and
job mixes (6). Nothing here changes the pre-registered Phase 2 result: `joint` does not
beat the re-tuned strong baselines.

Notation:
- **B2F** is `affinity-fixed-best`: J = 2 active jobs, LRU, FIFO. It never preempts.
- **jc** is `joint-checkpoint`.
- **Slowdown** is latency divided by (isolated compute + actual tool gaps).
- **Contention delay** is latency minus that same isolated time.

## 1. Across policy families there is no trade-off

The policies that waste prefill are fifo, affinity, queue, and fixed or
affinity-fixed with J ≥ 3. Their wasted prefill fraction is 0.4–0.85 on contended
seeds, and they are worse on throughput, mean JCT, P95 slowdown and Jain together.

Avoiding recompute is a free lunch: it helps the job that keeps its KV and every
job queued behind it.

## 2. Among efficient policies, a trade-off appears only under contention

On light seeds (dev 101, eval 1102 and 1103, gap-recorded 2101) every efficient
policy is within trial noise on every metric.

On contended seeds the pattern is consistent. B2F has the best or tied-best
efficiency, and the worst proportional fairness among efficient policies:

| Seed / condition | Policy | Tasks/min | Mean JCT | P95 slowdown | Short slowdown | Max admission wait |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| eval 1101 | B2F (T0 + T1) | 0.957 | 161 s | 7.30 | 4.21 | 172 s |
| eval 1101 | jc | 0.917 | 186 s | 5.98 | 3.41 | 0 |
| eval 1101 | joint (T0 + T1) | 0.893 | 214 s | 8.81 | 4.85 | 0 |
| dev 103 | B2F (J2) | 1.241 | 153 s | 12.5 | 7.67 | 111 s |
| dev 103 | ttl-10 | 1.127 | 169 s | 10.2 | 6.30 | 0 |
| dev 102 | B2F (J2) | 1.212 | 129 s | 21.2 | 8.98 | 136 s |
| dev 102 | joint-a80 | 1.207 | 121 s | 5.84 | 3.78 | 0 |
| gap-recorded 2102 | B2F | 1.405 | 191 s | 33.8 | 15.5 | 191 s |
| gap-recorded 2102 | fixed-ttl | 1.043 | 328 s | 25.3 | 14.1 | 0 |
| gap-long 2101 | B2F | 0.639 | 360 s | 8.73 | 4.62 | 469 s |
| gap-long 2101 | joint | 0.575 | 439 s | 4.82 | 3.31 | 0 |
| gap-long 2102 | B2F | 0.627 | 763 s | 28.9 | 16.8 | 989 s |
| gap-long 2102 | joint | 0.542 | 879 s | 8.97 | 6.78 | 0 |

Dev 102 is the exception: `joint-a80` matches B2F's throughput and is fairer, so
there it dominates.

The mechanism has two sides:
- **B2F** never preempts, so it wastes nothing, and the price shows up as admission
  waiting. Later arrivals queue for up to 989 s while the two active jobs hold the
  slots, even through 15 s tool gaps.
- **Sharing** admits everyone, so nobody waits for admission, and the price shows up as
  preemption. Without checkpoints, preemption means recomputing about 5k-token
  contexts: wasted prefill is 0.15–0.83. jc pays a save/restore of about 0.8 s instead
  and has the lowest waste of the sharing policies (0.15 on eval 1101, against 0.44
  for joint).

**Long tool waits sharpen the trade-off.** Under gap-long (fixed 15 s gaps), B2F's
admission queue grows to 469–989 s, while the sharing policies thrash:
- `joint` evicts 141–183 times; its cost lease gives up the slot when the queue is
  long;
- pooled, `joint` is only 1.035× fifo throughput.

So under unlimited admission the cost lease collapses toward thrash at high load, and
admission control is the lever that protects efficiency.

## 3. "Fair" depends on which fairness

Contention delay is a reference-free view of the same runs, and it reverses the
fairness reading:

| Seed / condition | Policy | Mean delay | P95 delay | Short-job P95 delay |
| --- | --- | ---: | ---: | ---: |
| eval 1101 | B2F | 96 s | 187 s | 174 s |
| eval 1101 | jc | 120 s | 327 s | 180 s |
| eval 1101 | joint | 149 s | 302 s | 212 s |
| gap-recorded 2102 | B2F | 149 s | 220 s | 220 s |
| gap-recorded 2102 | joint | 360 s | 769 s | 684 s |
| gap-long 2101 | B2F | 186 s | 437 s | 482 s |
| gap-long 2101 | joint | 265 s | 640 s | 609 s |
| gap-long 2102 | B2F | 589 s | 977 s | 989 s |
| gap-long 2102 | joint | 704 s | 1347 s | 1332 s |

- **B2F** gives every job, short or long, a similar absolute wait. That is fair in
  absolute terms and has the smallest tail, but it costs short jobs proportionally
  more.
- **Sharing** finishes short jobs early and pushes the delay onto long jobs. It wins
  on slowdown and loses on tail delay.

This is the classic FCFS vs processor-sharing contrast. The observed "fairness vs
efficiency" trade-off is partly a trade-off between proportional fairness (slowdown)
and absolute fairness (delay). Absolute fairness and efficiency point the same way.

## 4. The slowdown reference is a measurement risk

Every job starts with the same 4891-token prefix. Prefilling it cold costs about 30 s
under the cost model, comparable to a short job's whole isolated time (35–60 s). The
isolated reference is warm. Within a run only the first job per slot starts cold, and
that is the same for every policy.

Re-reading the runs with every reference shifted by δ = 30 s gives three results:
- slowdowns compress 3–5×;
- **Jain over 1/slowdown reverses rank** on several contended seeds. On eval 1101,
  jc 0.750 vs B2F 0.725 at δ = 0 becomes 0.587 vs 0.562, and best-fixed jumps to 0.743;
- the **order on P95 slowdown and short-task slowdown survives, with smaller gaps**.

Jain should not be used alone as fairness evidence in the report. P95 and short-task
slowdown, together with contention delay, are the robust pair.

## 5. Frontier supplement (pre-registered, new seeds)

**How it ran.**
- 9 of 9 runs completed (05:13–08:18), with 0 failed jobs, on AC power.
- Configurations match the eval parameter hashes. The 30 locked sources matched the
  campaign lock before the first run.
- The snapshot directory was empty afterwards.
- Seeds were chosen before running, by the a priori load screen.

| Seed | Policy | Tasks/min | Mean JCT | P95 slowdown | Short slowdown | Jain (δ=0 / δ=30) | P95 delay | Max admission wait | Wasted prefill | Saves / restores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3116 | B2F | 1.170 | 267 s | 30.8 | 18.4 | 0.633 / 0.724 | 290 s | 288 s | 0 | – |
| 3116 | joint | 1.026 | 343 s | 30.9 | 15.7 | 0.596 / 0.695 | 518 s | 0 | 0.49 | – |
| 3116 | jc | 1.174 | 294 s | 19.3 | 11.9 | 0.711 / 0.849 | 443 s | 0 | 0.25 | 80 / 80 |
| 3129 | B2F | 0.827 | 159 s | 29.1 | 11.6 | 0.734 / 0.627 | 212 s | 208 s | 0 | – |
| 3129 | joint | 0.827 | 180 s | 26.1 | 10.1 | 0.691 / 0.579 | 263 s | 0 | 0.21 | – |
| 3129 | jc | 0.827 | 154 s | 14.1 | 5.7 | 0.752 / 0.656 | 229 s | 0 | 0.09 | 55 / 55 |
| 3120 | B2F | 1.056 | 167 s | 14.9 | 7.2 | 0.753 / 0.513 | 235 s | 246 s | 0 | – |
| 3120 | joint | 0.958 | 217 s | 17.3 | 8.4 | 0.665 / 0.492 | 301 s | 0 | 0.40 | – |
| 3120 | jc | 1.044 | 179 s | 9.2 | 5.7 | 0.734 / 0.533 | 213 s | 0 | 0.15 | 53 / 53 |

On seed 3129 the makespan is set by the last arrival, so all three policies have the
same tasks/min.

**Predictions.** A prediction holds if it is true in at least 2 of 3 seeds.

| Prediction | Result | Verdict |
| --- | --- | --- |
| P1: B2F JCT < jc JCT | 2/3 (+10%, +7%, −3%) | holds |
| P2: jc P95 slowdown and Jain better than B2F | 2/3 at δ = 0, 3/3 at δ = 30 | holds |
| P2-delay: jc P95 contention delay < B2F's | 1/3 | fails, as P5 predicted |
| P3: checkpoints lower waste and JCT vs joint | 3/3 | holds |
| P4: joint JCT > B2F JCT | 3/3 | holds |
| P5: B2F P95 contention delay < jc's | 2/3 | holds |

On P2, jc's P95 slowdown is lower in 3 of 3 seeds. The δ = 0 count is 2/3 only because
B2F's Jain is higher on seed 3120.

**Pooled paired ratios.** These are descriptive, from a seed-cluster bootstrap over 3
seeds:

| Comparison | Throughput | JCT |
| --- | --- | --- |
| jc vs B2F | 0.997 [0.988, 1.003] | 1.048 [0.969, 1.101] |
| joint vs B2F | 0.928 [0.877, 1.000] | 1.240 [1.133, 1.301] |
| jc vs joint | 1.078 [1.000, 1.144] | 0.845 [0.825, 0.856] |

**Reading.** Under the pre-registered rule, P1 and P2 both hold, so this is a real
trade-off. It is a very uneven one:
- **jc's efficiency price is small** and its interval includes zero: +4.8% JCT, with
  equal throughput.
- **jc's proportional-fairness gain is large.** P95 slowdown is 37%, 52% and 38% lower.
  Short-task slowdown is 35%, 51% and 21% lower.
- **B2F keeps the absolute-tail edge** in 2 of 3 seeds, consistent with section 3.
- **Sharing without checkpoints is dominated.** joint is 24% slower than B2F with no
  consistent fairness gain. Its P95 slowdown is about equal on 3116 and worse on 3120.
  Sharing pays off only when preemption is cheap.
- **The eval negative result replicates, and grows on heavy seeds.** joint vs B2F JCT
  is 1.240 here, against 1.135 in eval.

**Caveats.**
- One trial per cell. jc's per-seed JCT gaps vs B2F (+10%, +7%, −3%) are within the
  ~10% trial noise measured for the sharing policies on eval S1101. The efficiency price
  is best described as "small, possibly zero". The fairness gaps are several times
  larger than that noise.
- The `joint` S3120 run swapped out 688 MiB, so its JCT may be inflated somewhat. This
  does not touch the jc vs B2F comparison.
- jc remains exploratory and is not the pre-registered final. These seeds were picked
  for contention, so the result describes the heavy regime, not the average case.

## 6. Arrival-rate and workload-mix supplement (pre-registered)

The design and predictions are in [6_RATE_MIX_PREREG.md](6_RATE_MIX_PREREG.md), committed
in `308aa6a` before the first run. Each condition changes one factor of the eval setup on
the eval seeds 1101–1103:
- **arrival-half / arrival-double:** the same arrival pattern at 0.5× and 2× the rate;
- **mix-short / mix-long:** short or long tasks drawn about 75% of the time.

The 1× cells are eval trial 0. R(c) is the fifo / B2F mean-JCT ratio in condition c: how
much an efficient scheduler gains over FIFO.

**How it ran.** 48 of 48 runs completed with 0 failed jobs, in about 24.7 h (estimate:
24–25 h), over three sessions:

| Session (PST) | Runs |
| --- | --- |
| 11-22 20:34 – 11-23 08:37 | 28: arrival-double 12, mix-long 12, mix-short 4 |
| 11-24 00:30 – 08:56 | 15: mix-short 8, arrival-half 7 |
| 11-24 09:58 – 14:09 | 5: arrival-half |

- Every session first checked that the 30 locked sources matched the campaign lock, then
  ran one unmeasured warm-up replay.
- Configurations match the eval parameter hashes. The snapshot directory was empty
  afterwards.

Three things went off plan. None of them touches a verdict:
- **One run ended on battery.** The Mac was unplugged at 08:32 on 11-23, so the last ~5 of
  the 16 minutes of mix-short / jc / S1101 ran on battery. `phase2.summarize` flags the run
  (`on_battery`) and drops that seed from the mix-short jc paired ratios, which therefore
  have 2 seeds. No prediction uses this cell: R5 uses only fifo and B2F.
- **One run started under Spotlight indexing.** In the first run of session 2
  (mix-short / B2F / S1102), median decode speed in the first 8 minutes was 22.9 tok/s,
  against 26.2–26.4 tok/s for the other three policies on that seed in the same window.
  After that it was normal (25.1 tok/s). Its JCT (50 s, against 47–48 s for the other
  policies) may be inflated by a few percent. It enters only R5. Even with B2F 5% faster,
  R(mix-short) on S1102 would be about 1.0, against 6.2 for mix-long.
- **Some runs swapped.** The largest swap-outs were fifo 0.5× S1102 (767 MiB), B2F
  mix-long S1101 (582 MiB), fifo 0.5× S1101 (464 MiB), B2F mix-long S1103 (369 MiB) and B2F
  0.5× S1101 (299 MiB). Every other run was at or below 216 MiB.
  - Swap inflates fifo at 0.5×, which works against R1.
  - It inflates B2F in mix-long, which works against R5. It also makes jc's mix-long
    deficit vs B2F look smaller than it is.
  - So none of the swap flatters a reported result.

**R(c): fifo JCT / B2F JCT.** From `run_summary.csv`, trial 0. The a priori load of each
cell is in the pre-registration.

| Seed | 0.5× | 1× | 2× | mix-short | mix-long |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1101 | 3.63 | 4.92 | 2.77 | 4.04 | 4.06 |
| 1102 | 1.01 | 1.59 | 5.04 | 0.95 | 6.20 |
| 1103 | 1.11 | 2.27 | 5.07 | 1.37 | 6.41 |

**Predictions.** A prediction holds if it is true in at least 2 of 3 seeds.

| Prediction | Result | Verdict |
| --- | --- | --- |
| R1: at 0.5×, B2F, joint and jc within 10% on JCT, and R(0.5×) < R(1×) | 3/3 | holds |
| R2: R(2×) > R(1×) | 2/3 | holds |
| R3: at 2×, jc P95 slowdown < B2F's and jc JCT ≥ B2F's | 2/3 | holds |
| R4: at 2×, joint JCT > B2F's | 3/3 | holds |
| R5: in mix-long, B2F's max admission wait and R both larger than in mix-short | 3/3 | holds |

- **R1.** At 0.5× the three efficient policies are within 5% of each other: 98–103 s on
  S1101, 68–69 s on S1102, 70–72 s on S1103.
- **R2.** S1101 is the exception. FIFO is already saturated there at 1× (JCT 803 s), and
  doubling the rate moves it only to 854 s, while B2F goes from 163 s to 308 s. Past
  saturation the ratio stops growing.
- **R3.** S1103 is the seed where the conjunction fails, and it fails in jc's favor: jc is
  better on both axes (JCT 138 vs 148 s, P95 slowdown 5.40 vs 9.23). The pre-registration
  reads that outcome as "jc dominates under overload", for that seed.
- **R5.** B2F's maximum admission wait is 450/141/81 s in mix-long against 113/53/47 s in
  mix-short. The R comparison on S1101 is a near tie: 4.06 vs 4.04.

**Heavy cells, efficient policies.** On the light cells (0.5×, mix-short, and 1× on S1102
and S1103), B2F, joint and jc are within trial noise on every metric. The full tables are
printed by `scripts/phase2/4_rate_mix_check.py`.

| Cell | Policy | Mean JCT | P95 slowdown | Short / long slowdown | P95 delay | Long-task P95 delay | Max admission wait | Wasted prefill | Saves |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2× S1101 | B2F | 308 s | 16.37 | 9.53 / 3.95 | 474 s | 480 s | 456 s | 0 | – |
| 2× S1101 | joint | 498 s | 14.56 | 9.07 / 7.44 | 914 s | 914 s | 0 | 0.65 | – |
| 2× S1101 | jc | 373 s | 10.87 | 6.56 / 5.49 | 618 s | 646 s | 0 | 0.41 | 57 |
| 2× S1102 | B2F | 132 s | 14.81 | 5.01 / 1.99 | 139 s | 192 s | 179 s | 0 | – |
| 2× S1102 | joint | 137 s | 13.52 | 4.60 / 2.00 | 173 s | 177 s | 0 | 0.14 | – |
| 2× S1102 | jc | 138 s | 6.29 | 3.60 / 2.17 | 170 s | 177 s | 0 | 0.06 | 64 |
| 2× S1103 | B2F | 148 s | 9.23 | 6.53 / 1.94 | 145 s | 132 s | 168 s | 0 | – |
| 2× S1103 | joint | 177 s | 12.82 | 7.28 / 2.31 | 191 s | 139 s | 0 | 0.30 | – |
| 2× S1103 | jc | 138 s | 5.40 | 3.28 / 2.03 | 189 s | 205 s | 0 | 0.08 | 48 |
| mix-long S1101 | B2F | 316 s | 6.01 | 8.33 / 3.23 | 452 s | 476 s | 450 s | 0 | – |
| mix-long S1101 | joint | 482 s | 11.67 | 10.36 / 5.13 | 863 s | 888 s | 0 | 0.57 | – |
| mix-long S1101 | jc | 385 s | 9.10 | 5.80 / 4.18 | 583 s | 620 s | 0 | 0.28 | 105 |
| mix-long S1102 | B2F | 133 s | 3.05 | 2.57 / 1.46 | 142 s | 154 s | 141 s | 0 | – |
| mix-long S1102 | joint | 149 s | 3.34 | 3.63 / 1.61 | 159 s | 175 s | 0 | 0.15 | – |
| mix-long S1102 | jc | 151 s | 3.06 | 3.27 / 1.65 | 148 s | 163 s | 0 | 0.03 | 43 |
| mix-long S1103 | B2F | 124 s | 2.61 | 1.71 / 1.48 | 100 s | 100 s | 81 s | 0 | – |
| mix-long S1103 | joint | 145 s | 2.70 | 2.31 / 1.74 | 138 s | 155 s | 0 | 0.19 | – |
| mix-long S1103 | jc | 130 s | 2.21 | 1.68 / 1.58 | 88 s | 92 s | 0 | 0.03 | 41 |

**Pooled paired JCT ratios.** These are descriptive, from a seed-cluster bootstrap over
3 seeds (2 where marked). A ratio below 1 favors the first policy.

| Condition | jc / B2F | joint / B2F | jc / joint | B2F / fifo |
| --- | --- | --- | --- | --- |
| 0.5× | 0.992 [0.980, 0.999] | 0.997 [0.967, 1.028] | 0.996 [0.953, 1.033] | 0.722 [0.276, 0.990] |
| 2× | 1.059 [0.927, 1.208] | 1.280 [1.035, 1.615] | 0.845 [0.748, 1.007] | 0.252 [0.197, 0.361] |
| mix-short | 0.982 [0.943, 1.021] (2 seeds) | 1.022 [0.957, 1.105] | 1.001 [0.985, 1.016] (2 seeds) | 0.675 [0.248, 1.050] |
| mix-long | 1.133 [1.045, 1.220] | 1.270 [1.119, 1.525] | 0.903 [0.800, 1.012] | 0.188 [0.156, 0.246] |

For reference, joint / B2F in the eval confirmatory table (1×) is 1.11 [0.97, 1.33].

**Are long tasks unfairly delayed?** This is descriptive, as pre-registered.
- **In light cells nobody is delayed much.** Long-task slowdown is 1.0–1.4 under both B2F
  and jc.
- **In heavy cells sharing moves the delay onto long tasks.** In jc vs B2F:
  - long-task slowdown is 5.49 vs 3.95 (2× S1101) and 4.18 vs 3.23 (mix-long S1101);
  - long-task P95 contention delay is 646 vs 480 s and 620 vs 476 s.
- **Under B2F short tasks pay instead.** Their slowdown is 9.53 vs 6.56 and 8.33 vs 5.80.

This is the same FCFS vs processor-sharing contrast as section 3. Under capped
run-to-completion, long tasks are not singled out; under checkpointed sharing they carry
the delay.

**Not pre-registered: the trade-off weakens when long tasks dominate.** mix-long has only
3–5 short tasks per run.
- jc's JCT is 22%, 14% and 5% above B2F's. Pooled, 1.133 [1.045, 1.220] is the only
  condition in this supplement where jc is clearly slower than B2F.
- Its P95 slowdown is better only on S1103 (2.21 vs 2.61). It is equal on S1102 (3.06 vs
  3.05) and worse on S1101 (9.10 vs 6.01).
- A plausible reading: sharing's gain goes to short tasks, and with few of them there is
  little to gain, while preemption still costs (171 evictions and 105 saves on S1101).
- This is one trial per cell and was not predicted, so it is a lead, not a result.

**Reading.**
- **The value of scheduling follows queueing, whether the load comes from the arrival
  rate or from the job mix** (R1, R2, R5). On the lightest cells (0.5× and mix-short on
  S1102, 0.5× on S1103), fifo and B2F are within about 10% of each other (R 0.95–1.11),
  and the efficient policies tie. These are the "little benefit" cases the course asks for. S1101 stays
  heavy even at 0.5× (R 3.63).
- **R does not grow without bound.** Past saturation it falls (S1101 at 2×).
- **mix-long raises R more than its a priori load suggests** (6.20 on S1102, at an a
  priori wait of 6.1 s). The indicator models run-to-completion and does not count FIFO's
  recompute of long contexts.
- **The frontier trade-off carries over from heavy seeds to heavy rates** (R3). It is also
  just as uneven: jc gives up 5.9% JCT pooled (interval includes 1), and its P95 slowdown
  is 34%, 58% and 41% lower.
- **The negative result for `joint` grows with load** (R4). joint / B2F is 0.997 at 0.5×,
  1.11 at 1× and 1.280 at 2×. jc / joint at 2× is 0.845, the same value as on the frontier
  seeds: checkpoints are what make sharing affordable.

**Caveats.**
- One trial per cell, with about 10% trial noise for the sharing policies.
- The 1× cells come from an earlier session (11-19 / 11-20). Comparisons across conditions
  mix a condition effect with a session effect; comparisons between policies within a
  condition do not.
- mix-long has only 3–5 short tasks and mix-short only 3–5 long tasks, so short-task
  numbers in mix-long and long-task numbers in mix-short are noisy.
- These are new permutations and rates of the same 20 eval tasks.

## 7. What the data supports and what it does not

**Supported:**
- Avoiding recompute improves efficiency and fairness together (section 1).
- On new pre-registered seeds, KV checkpointing makes sharing worthwhile. jc
  beats joint on waste and JCT in 3 of 3 seeds (JCT ratio 0.845 [0.825, 0.856]). jc
  buys large proportional-fairness gains over B2F for a small, possibly zero,
  efficiency price (section 5).
- Under contention, admission-limited run-to-completion maximizes throughput and
  minimizes absolute tail delay, while sharing lowers proportional slowdown
  (sections 2 and 3).
- Long tool waits make both effects larger.
- The gain of any KV-keeping scheduler over FIFO follows queueing (section 6, all five
  pre-registered predictions hold):
  - it nearly vanishes at half rate and in a short-heavy mix on light seeds (fifo / B2F
    0.95–1.11);
  - it reaches 2.8–6.4× at double rate and in a long-heavy mix.
- The jc vs B2F trade-off generalizes from heavy seeds to heavy arrival rates, and the
  negative result for `joint` grows with load (section 6).

**Not supported:**
- That any one policy is "fairest". The answer changes with the metric and with the
  slowdown reference.
- That jc beats B2F overall. On 3 new heavy seeds, jc gives up about 5% JCT (not
  significant) for 35–50% lower P95 slowdown, and B2F keeps the absolute tail. That is
  a trade-off, not dominance. There is still one trial per cell, and jc is not the
  pre-registered final.
- Average-case claims from the contended seeds. Two of three eval seeds are light, and
  there every efficient policy ties.
- That checkpointed sharing helps in every heavy regime. In the long-heavy mix jc is 13%
  slower than B2F (pooled) with no consistent fairness gain. This was not predicted and
  rests on one trial per cell (section 6).

**Next experiments, if time allows:**
1. **J = 3 or 4 with checkpoints** (affinity, LRU, `checkpoint=True`). This
   interpolates between B2F and jc: more admission without recompute. It is the most
   direct way to trace the frontier, and under gap-long it could beat both ends.
2. **Size-aware admission (shortest expected job first) at J = 2.** It keeps zero
   waste and should cut short-job slowdown, at the cost of the longest jobs. It needs
   a new queue rule, so it belongs in a new campaign rather than the locked one.
3. **jc under gap-long**, and trial repeats for jc on the contended seeds.

## Reproducing

- **Per-seed tables (sections 1–2):** `runs/phase2/2_m4_campaign/{dev,eval,sensitivity}/summary/run_summary.csv`.
- **Contention delay and the δ shift (sections 3–4):** computed from the `jobs.csv`
  files in the same directories, as latency minus `isolated_s` and
  latency / (`isolated_s` + δ).
- **Supplement (section 5):**

```bash
bash scripts/phase2/3_frontier_run.sh
```

```bash
.venv/bin/python scripts/phase2/3_frontier_check.py
```

- **Arrival-rate / mix supplement (section 6):** the driver resumes where it stopped and
  runs `phase2.summarize` and the check at the end. Results are in
  `runs/phase2/4_m4_rate_mix/sensitivity/`, the summary in its `summary/`, and the session
  log in `runs/phase2/4_m4_rate_mix/rate_mix.log`.

```bash
nohup caffeinate -is bash scripts/phase2/4_rate_mix_run.sh --until 09:00 >/dev/null 2>&1 &
```

```bash
.venv/bin/python scripts/phase2/4_rate_mix_check.py
```
