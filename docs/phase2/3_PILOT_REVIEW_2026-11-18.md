# Review of the Phase 2 implementation and pilot results (2026-11-18)

> Translated from the Chinese original on 2026-11-22. This review covers the **pilot**,
> now in `runs/phase2/1_m4_pilot` (originally `runs/phase2-m4-air-16g`). Its three P1 issues
> were fixed in revision 2, whose campaign is `runs/phase2/2_m4_campaign`.

Scope:
- the valid probe / dev / eval raw logs, manifests, frozen config and campaign lock of the
  pilot;
- the source code at the time.

The latest session ended normally by deadline at 08:39 on 2026-11-18. This review added
only this document. It did not change the algorithm, the frozen config or raw results, and
started no model experiment.

## 1. Verdict

The research direction is sound, and data recording and replay-work integrity are good.
The most reliable finding so far: **capping the number of active tasks while keeping slot
affinity already captures most of the throughput gain. Cost-driven retention and
checkpoints can improve short-task fairness, but have not been shown to beat the strongest
simple baseline overall.**

What to do next:
- First fix three things: the inconsistency between checkpoint pricing and execution,
  the inconsistency in the frozen config, and the pairing of repeated experiments. Then
  run limited confirmation experiments.
- There is no need to redo all of Phase 2, and no reason to keep enlarging the policy grid
  indiscriminately.
- The existing measurements still describe the old implementation's behavior. After the
  implementation changes, related old results stay as exploratory results; they cannot
  replace confirmation of the new implementation.

## 2. How much is done, and is the data trustworthy?

**Coverage.**
- **dev:** 60/60 runs, 20 configs × 3 seeds × 1 trial, 600/600 tasks completed.
- **eval:** the manifest has 31 runs, of which 10/31 are done, all S1001/T0, with 200/200
  tasks completed. 1002/1003 and the RAM-cache group are not done.
- **Work per run.** Every valid eval run completes the same 20 tasks, 205 model steps and
  13,632 output tokens. Completion means the replay completed, not semantic task success.

**Integrity checks.**
- Recomputing dev from raw to summary agrees on completions, throughput, JCT, P50/P95,
  history hit and recompute waste.
- **Same seed, same work.** Within a seed, task order, steps, input/output lengths and
  planned gaps are identical. The largest deviation of actual arrival times is 5.85 ms on
  dev and 2.47 ms on eval.
- No duplicate steps, truncation, prefix verification failures or mock performance data
  were found.
- **Retired runs are kept apart.** The 21 retired complete runs and 1 deliberately
  cancelled run stay in `invalid/`, outside the valid summaries.
- **Inputs are clean.** The cost profile's source file hashes match, and the wait prior
  comes from the train/dev side, not eval.

**No tuning on eval.**
- Current source hashes match the campaign lock.
- The frozen config was approved before eval started.
- No evidence of tuning on this eval was found. The configuration problem below is a
  mismatch between the dev selection scores and the deployed parameters.

Main evidence:
- [dev summary](../../runs/phase2/1_m4_pilot/dev/summary/REPORT.md);
- [frozen config](../../runs/phase2/1_m4_pilot/manifests/frozen.json);
- [campaign sessions](../../runs/phase2/1_m4_pilot/campaign/sessions.jsonl).

Eval numbers were recomputed in memory with the same helpers as `phase2.summarize`, not
written back to the experiment folder.

## 3. How to read the results

### dev: means over three seeds

| Policy | tasks/min | Mean JCT / s | P95 / s | Short-task slowdown |
| --- | ---: | ---: | ---: | ---: |
| FIFO | 0.711 | 275.0 | 506.9 | 6.70 |
| best fixed: K2/J2 | 1.069 | 136.1 | 212.5 | 7.04 |
| affinity + K2/J2 | **1.087** | **129.3** | **208.5** | 6.49 |
| cache, aging=30 | 0.897 | 181.1 | 335.2 | 4.72 |
| joint, aging=80 | 1.003 | 146.2 | 235.2 | 5.91 |
| joint-checkpoint, aging=30 | 1.015 | 144.3 | 254.3 | **3.74** |

- Checkpoint throughput is about 1.43× FIFO, but about 6.6% below the strongest
  affinity + J2.
- Its main advantage over best-fixed is about 47% lower short-task slowdown, not the
  highest throughput.
- Throughput gaps of about 1%–2% between configs should be read with caution without
  repeats.

### eval: S1001/T0 only, preliminary

| Policy | tasks/min | Mean JCT / s | P95 / s | Short-task slowdown |
| --- | ---: | ---: | ---: | ---: |
| FIFO | 0.538 | 497.7 | 1238.7 | 5.36 |
| best fixed: K2/J2 | 0.933 | 114.7 | 213.5 | 3.61 |
| affinity + K2/J2 | 0.951 | **102.6** | **197.7** | 2.99 |
| cache, aging=80 | **0.958** | 106.7 | 234.4 | 3.07 |
| joint, aging=80 | 0.934 | 119.2 | 265.6 | 3.68 |
| joint-checkpoint, aging=80 | 0.944 | 114.0 | 257.0 | **2.47** |

- **joint vs best-fixed:** throughput only +0.11%, and P95 +24.4%. The joint queue +
  cache combination cannot yet be said to be synergistic.
- **checkpoint vs best-fixed:** throughput +1.12%, mean JCT −0.62%, short-task slowdown
  −31.5%, P95 +20.4%.
- **checkpoint vs affinity + J2:** throughput −0.72%, mean JCT +11.1%, P95 +30.0%.
- **cache-only** has the highest throughput for now, but only about 0.8% above
  affinity + J2, with worse mean and tail latency. One seed cannot show whether that
  repeats.
- **Checkpoint I/O.**
  - It executed 58 saves and 58 restores, writing and reading about 56.07 GB each, with
    64.92 s of total I/O call time.
  - No quota skips, I/O errors or uncertain states.
  - It does reduce recompute, but it is not yet shown that this I/O cost buys a necessary
    throughput gain in the main scenario.
- With one seed there is no cross-seed bootstrap; all percentages above are descriptive.

These patterns fit the mechanism: P2 has only two slots, so many interleaved tasks keep
overwriting private history, and J2 removes the main source of damage directly. A more
complex method still has research value if it serves more short tasks at close to J2
throughput, but it must show the cost in fairness, overall latency and tail latency at the
same time.

## 4. Priority fixes

### P1: checkpoint pricing and actual execution use different conditions

Location:
- [scheduler.py](../../phase2/scheduler.py): `_eviction_cost` (about lines 263–281),
  `_protect` and `_prepare_slot` (about lines 426–431);
- the net-benefit condition is `checkpoint_worthwhile` in
  [costs.py](../../phase2/costs.py).

**The mismatch.**
- `_eviction_cost` uses the cheap checkpoint cost to decide whether to give up the lease,
  as long as the budget has room and save+restore is below recompute.
- The actual save also requires the net-benefit condition with queue pressure, use
  probability and margin.
- When the latter refuses the save, the slot is still evicted and pays recompute, and the
  earlier lease decision is never re-evaluated against recompute.

Model-free reproduction with this machine's real profile: a 4,891-token shared prefix + a
400-token private suffix, 7 other ready requests, observed waits `[2, 2, 2]` s.

| Check | Result |
| --- | --- |
| `_eviction_cost` quote | checkpoint, 0.474447 s |
| Predicted actual recompute cost | 2.626774 s |
| `_protect` | no lease |
| `checkpoint_worthwhile` | False |
| Actual `save` calls | 0 |
| If the lease were chosen against recompute | keep 2 s, predicted net value 7.014190 s |

**How often it happened in eval.**
- The real eval checkpoint logs show at least 9 events where the queue state at the time
  confirms that the net-benefit gate refused.
- In total, 19 owner/versions were quoted as checkpoint and never saved. Some of these may
  have normal causes such as the task ending, so not all count as the bug.

**A traceable instance** in `eval/eval/main/joint-checkpoint/S1001-T0/raw/policy_decisions.jsonl`:
- Line 136 quotes owner `…:job:9`, version 1 as checkpoint at 0.4471 s and grants no lease.
- Line 138 evicts it, and the same owner/version is never saved.
- Then line 59 of `macro_requests.jsonl` has `cache_n=4892 < reusable_tokens=4986`: 94 old
  tokens were actually recomputed.

Separately, the 154 requests after a restore in valid checkpoint runs all reused at least
the snapshot's token count, so there is no evidence of restoring a wrong version.

**Minimal fix.**
- keep/save/recompute share one action quote that includes budget, state and the
  net-benefit condition.
- If the save will not actually happen, re-evaluate against the executable alternative
  before overwriting the slot.
- Add a regression test for "checkpoint is cheaper than recompute, but queue pressure makes
  it not worth saving"; the existing cheap-checkpoint test does not cover that condition.

The performance effect must be re-measured; the reproduction cannot predict how much it
improves.

### P1: the dev scores freeze selected do not match the final complete configs

Location: [freeze.py](../../phase2/freeze.py), about lines 150–161.

**The mismatch.**
- freeze picks aging=80 from the joint grid, then changes queue, cache and
  joint-checkpoint all to 80.
- But `_final` still chooses the final system using those policies' original aging=30
  dev scores.
- Checkpoint's dev parameter hash is `5a0a5192…`, while eval's is `a60b4703…`.

So one cannot say "the final complete config passed selection and the fairness gate on
dev".

**Minimal fix.**
- Preferably keep the shared-aging ablation design: complete queue/cache/checkpoint at
  aging=80 on dev, then select and freeze by full parameter hash.
- The other workable path is to freeze each config exactly as measured, stating that they
  do not form a strictly shared-parameter ablation.
- Add a test: every selected final complete config must have matching dev evidence, not
  just a matching policy name.

### P1: after the extra trials, the main ablations lose their paired comparisons

Location:
- [campaign.py](../../phase2/campaign.py): `KEY_LABELS` and `eval-trials-manifest`
  (about lines 313–319);
- [summarize.py](../../phase2/summarize.py), about lines 420–423.

**The problem.**
- The campaign adds trials (up to three) only for FIFO, best-fixed and the final system;
  the other methods keep T0.
- The summarizer requires both sides to have exactly the same trial set. So once the extra
  trials finish, cache/joint/TTL etc. are excluded from pairing with the core baselines by
  `trial_set_mismatch`.
- The strongest simple baseline, affinity + J2, is also absent from the core repeats and
  sensitivity set.

**Minimal fix.**
- Explicitly output an "all-policy T0 paired table" and a "core three-trial confirmation
  table", or compare on a stated common trial subset and publish the sample size.
- Put affinity-fixed-best into the core confirmation.
- Add a test for the "baseline three trials, ablation one trial" summary flow. Keep the
  mismatch diagnostics, and never silently mix different samples.

### P2: online prefill updates are sensitive to small token counts and fixed overhead

Location: `update` in [costs.py](../../phase2/costs.py), about lines 105–123.

**The problem.**
- Offline calibration excludes small retain samples, but online updates still feed every
  `prompt_ms / prompt_n` into the EWMA.
- The real logs have requests with 16 prompt tokens taking about 5 s. Fixed overhead or
  batching interference then reads as an extremely high per-token cost.
- In some dev runs, the final K2 slope reached about 7–10× its initial value.

**How big the effect is.** It should not be exaggerated.
- Rebuilding from real `model_done_s`, finished waits and decision events reproduces the
  lease/no-lease decisions checked in eval: cache 80/80, joint 90/90, checkpoint 80/80.
- A shadow computation that skips only updates with `prompt_n < 128` flips:
  - 6 of 134 decisions of dev joint-a80;
  - 3/90 for eval joint;
  - 0/80 for eval checkpoint.
- It is an estimation-robustness issue, not yet the main cause of the current checkpoint
  I/O or overall performance gap.

**Suggestion.** On dev, compare a frozen profile, updates only from large enough
prefills, or an explicit fixed-overhead model. The threshold should also be chosen on dev,
not tuned for a better result on this eval.

### P2: the TTL control must match aging and the protection time scale

**The mismatch.**
- TTL candidates are 1/3/10 s with aging=30, while the cost lease cap is 60 s and eval
  aging is 80.
- On S1001, TTL has 85 evictions and breaks 5 still-valid leases; cache has 8 evictions
  and breaks 0.
- So the TTL gap mixes the effects of protection length, aging and cost decisions.

**Suggestion.** Add limited dev controls of TTL10/aging80, and TTL15 or TTL30/aging80.
Otherwise limit the conclusion to "the whole configuration works", and do not attribute
the TTL gap to cost adaptivity alone. No large TTL grid is needed.

## 5. Realism and resource boundaries

- **Waits.** The main scenario's gap is a synthetic log-normal (median 2 s, sigma 1). Real
  Retail tool waits are mostly milliseconds. Sensitivity supports `gap-recorded`, but the
  campaign did not select it. Suggestion: use it in place of a low-priority mix
  condition, to test whether retention still has value at real wait scales.
- **Output identity.** Equal work lengths are verified, but generated content is not all
  byte-identical.
  - Against isolated runs: FIFO 16/20 tasks fully identical, affinity + J2 and cache 20/20,
    joint 18/20, checkpoint 19/20. The local restore probe is 16/16 identical.
  - Batching numerics may make generations diverge. That neither proves KV corruption nor
    lets us claim semantic quality is fully unchanged.
- **Power and temperature.**
  - All valid runs started and ended on AC, and no `pmset` thermal warning was reported.
  - There is no continuous temperature/frequency observation, so this does not prove the
    absence of thermal throttling.
  - The 11-18 warm-up covered only the rerun cost-policy dev runs. The old simple
    baselines came from an earlier session, so small dev gaps may include session-state
    differences.
- **Memory.**
  - Eval system swap used was about 0.49–0.50 GiB, so "zero swap throughout" cannot be
    claimed. System counters also cannot be read as the exact disk swap caused by the
    model.
  - RAM-cache peak RSS is about 6.2 GiB, the normal group about 3.5 GiB. The existing
    pairing correctly marks it as a resource mismatch; its resource cost should be
    reported separately.
- **Stale runbook.** The runbook at the time still said "no real-model experiment run yet",
  and some counts and durations lagged behind the manifests. Update the runbook along with
  later changes so readers do not take implementation-stage notes for the current state.

## 6. Recommended order for the next round

1. **Fix the implementation and evaluation pipeline first, with model-free regression
   tests.**
   - Finish the three P1s first. Do not resume the existing campaign directly, or it will
     keep accumulating results unfit for the final argument.
   - Make the cost update robust as a separate dev validation, to avoid mixing too many
     algorithm changes at once.
2. **Add limited dev.**
   - Confirm queue/cache/checkpoint at the shared aging=80, and add the limited
     matching-aging TTL controls.
   - Keep FIFO, best-fixed and affinity + J2 as references.
   - Do not enlarge the K/J grid, and add no new mechanisms such as RL or compression.
3. **Re-freeze traceable complete configs.**
   - Keep all current raw logs and version records.
   - If the method or parameters are re-chosen after the fixes, treat the already-seen
     S1001 as a pilot, and pre-register unseen new eval seeds for confirmation. Never
     tune toward S1001's ranking.
4. **Core confirmation first, then a few sensitivity conditions with a mechanism purpose.**
   - The core must include affinity + J2, and the all-policy T0 and core repeats get
     separate tables.
   - Prefer recorded gap, arrival pressure and one long-wait condition.
   - The remaining old plan is about 21 main runs + 18 extra runs + 48 sensitivity runs:
     about 40.6 h at 1.4 min per task, not counting restarts and dev confirmation.
     Shrinking the matrix is worth more than adding policies.
5. **Center the report on the Pareto trade-off.**
   - Show throughput, mean/P95 JCT, short/long-task slowdown, recompute and checkpoint I/O
     together.
   - Claim a general performance gain only if strong baselines and repeated confirmation
     support it.
   - If simple affinity + J2 is enough, that is also a reasonable conclusion consistent
     with the system's mechanism.

## 7. Verification in this review

Ran the existing tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -p no:cacheprovider
```

- **Inside the sandbox:** 195 passed / 12 failed. The failures came from macOS swap
  counter access and localhost ephemeral port permissions.
- **Outside the sandbox:** **207 passed in 5.85 s.**
- No real model was started and no cloud service was accessed.
- The pure in-memory reproduction of the checkpoint condition above was also run.
- Passing tests do not cover the decision-consistency and experiment-design gaps found in
  this review.
