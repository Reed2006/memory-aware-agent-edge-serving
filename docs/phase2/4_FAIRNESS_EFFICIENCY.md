# Phase 2: where fairness and efficiency trade off

Analysis of every completed Phase 2 run on the M4 Air (16 GB), written 2026-11-21:

| Source | Runs |
| --- | --- |
| dev | 57 |
| eval | 43 (T0 + T1) |
| sensitivity | 16 (gap-recorded and gap-long) |
| frontier supplement | [5_FRONTIER_PREREG.md](5_FRONTIER_PREREG.md) |

Sections 1–4 are post-hoc readings of data that also chose the configurations.
Section 5 is the pre-registered supplement on new seeds. Nothing here changes the
pre-registered Phase 2 result: `joint` does not beat the re-tuned strong baselines.

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

## 6. What the data supports and what it does not

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

**Not supported:**
- That any one policy is "fairest". The answer changes with the metric and with the
  slowdown reference.
- That jc beats B2F overall. On 3 new heavy seeds, jc gives up about 5% JCT (not
  significant) for 35–50% lower P95 slowdown, and B2F keeps the absolute tail. That is
  a trade-off, not dominance. There is still one trial per cell, and jc is not the
  pre-registered final.
- Average-case claims from the contended seeds. Two of three eval seeds are light, and
  there every efficient policy ties.

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
