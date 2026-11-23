# Phase 2 runbook

> Translated from the Chinese original on 2026-11-22.
>
> **Status as of 2026-11-22.** The M4 campaign (revision 2) is complete:
> - probes, dev 57 runs, eval 43 runs, sensitivity 16 runs;
> - results in `runs/phase2/2_m4_campaign` (originally `runs/phase2-m4-air-16g-r2`);
> - pilot logs in `runs/phase2/1_m4_pilot` (originally `runs/phase2-m4-air-16g`).
>
> The supplements have their own drivers in `scripts/phase2/`. What is left is listed in
> [7_TODO.md](7_TODO.md), and the map of every file is [0_README.md](0_README.md).
>
> **Paths below.** The commands use the default campaign folder
> `runs/phase2-$PLATFORM_ID-r2`, which is what `phase2.campaign init` creates. To put a new
> campaign straight into the numbered layout, pass `--root` before the action, for example
> `.venv/bin/python -m phase2.campaign --root runs/phase2/5_t4_campaign init t4 t4-g4dn-xlarge`.

The research design, gates and evaluation constraints are in [1_PLAN.md](1_PLAN.md);
section 13 there is the design supplement after the 2026-11-16 review.

How to run the commands:
- Every command runs from the project root with the existing `.venv`.
- `scripts/phase2.sh` never starts an inference server.
- `run` without `--execute` only prints commands.
- Logs from `httpx.MockTransport` carry `synthetic_performance=true` and are never used for
  performance claims.

## 0. Start here: staged plan and durations

Estimates come from M4 pilot measurements: a dev run takes about 13.4 minutes, an eval run
about 23.5 minutes. The "after reuse" column assumes `init --reuse-from` imports the pilot
(section 0.5). `status` counts what remains from the actual manifests. The rows assume
that the best-fixed chosen on dev differs from FIFO, and that affinity-fixed-best is kept.

| Tier | Content | Runs | To run after reuse | M4 estimate | Required? |
| --- | --- | ---: | ---: | ---: | --- |
| T0 | Mechanism probes, calibration, dev wait distribution; smoke | — | smoke only | ~10 min (about 3 h without reuse) | Required |
| T1a | Baselines and joint aging screen: 13 configs × 3 seeds | 39 | 9 (joint-a20/40/80) | ~2 h | Required |
| T1b | Q/C/CP and TTL10/15/30 at the chosen aging: 6 configs × 3 seeds | 18 | 18 | ~4 h | Required |
| T2 | Eval main comparison: 10 configs × 3 seeds + 1 isolated | 31 | 31 | ~12 h | Required |
| T3 | One extra trial of FIFO / best-fixed / affinity-fixed-best / final system | 12 | 12 | ~4.7 h | Should do |
| T4 | FIFO, affinity-fixed-best (or best-fixed), final system, TTL × 2 conditions × 2 seeds | 16 | 16 | ~6.7 h | As time allows |

The required part takes about 19 h, everything about 30 h. Sensitivity covers only
`gap-recorded,gap-long`:
- `gap-recorded` tests the recorded raw (millisecond-scale) external waits;
- `gap-long` tests 15 s waits, which the TTL tuned on dev does not cover.

Whether a T4 machine is available, and its budget, must be checked separately. Historical
speeds are not performance evidence for the new implementation.

Before formal experiments:
- **Connect AC power.**
  - Real runs and probes refuse to start on battery.
  - `--allow-battery` is only for labeled smoke tests, and such runs never enter paired
    comparisons.
- **Expect throttling.** The MacBook Air has no fan and throttles in long runs.
  - Each run's metadata records `pmset -g therm`.
  - Policy order rotates by seed and trial, so no policy always runs on a cool machine.
- **Keep the checkpoint budget at the 4 GiB that dev actually tested.**
  - freeze only validates the budget; it never overwrites the measured config.
  - Each start checks current disk and memory.
- **Do not claim zero swap.** The pilot eval system already had about 0.49 GiB of swap.
  System page counts also cannot be attributed directly to model disk swapping.
- **One sweep at a time, on a server dedicated to the experiment.**

## 0.5 Recommended: run in nightly segments (`scripts/phase2_night.sh`)

Every step of sections 2–6 is chained into one resumable campaign
([phase2/campaign.py](../../phase2/campaign.py)):
- start it each night;
- cut it off any time in the morning;
- the next night continues from where it stopped, with the environment guaranteed to be
  the same.

`scripts/phase2/2_campaign.sh` is a thin wrapper for the same launcher in the numbered
layout.

**First time only: lock the environment**

```bash
bash scripts/phase2_night.sh init m4 m4-air-16g --reuse-from runs/phase2-m4-air-16g
```

(The pilot now lives at `runs/phase2/1_m4_pilot`.)

**What `--reuse-from` imports.** It first validates the whole pilot and then imports; if
validation fails, nothing is written. It imports:
- probes, wait distribution, `probe.json` and `profile.json`;
- the completed, failure-free dev runs of fifo, fixed K/J, affinity, affinity-fixed and
  RAM-cache (30 on the M4 pilot).

The profiles are copied byte for byte, so reused runs keep the same cost hash. The online
settings missing from that profile take the defaults 128 / true / true, which are the
values current calibration writes.

**Why the imported runs are still valid.**
- These policies never price a lease, never checkpoint and never read the cost model.
- No code change after the pilot's first dev run touches their path (see the pilot's
  lock-history), so the pilot's runs are exactly what revision 2 would produce.

**Conditions and checks.**
- llama-server, model, Python, packages and workload must be identical to the pilot.
- Source differences are recorded in `campaign/imported.json` and the lock-history.
- Smoke tests are not imported: the new scheduling code must run once on the real backend.
- Every run's start time was checked: no imported run was the first after an idle machine
  (probes or other runs always came before).

**What `init` writes.**
- It writes `runs/phase2-m4-air-16g-r2/campaign/lock.json` (now
  `runs/phase2/2_m4_campaign/campaign/lock.json`) and switches the `.phase2_campaign`
  pointer. It never starts the model.
- It locks the build, model, sources, Python packages, workload, revision, seeds and
  sensitivity conditions. The folder must be empty.
- The old pilot can still show status and logs, but cannot resume or relock under the
  fixed code.
- For the T4 the command is `init t4 t4-g4dn-xlarge` (arrival rate 3.5/min, aging grid
  4/8/16 s).

**Each night**

```bash
bash scripts/phase2_night.sh start --until 07:30
```

This runs in the background and uses `caffeinate` to prevent system sleep (the display may
sleep). **Closing the lid puts the Mac to sleep**, so keep the lid open and the charger
connected. `--until 07:30` means no run or stage starts if it is expected to end after
07:30.

**In the morning (all three are safe)**

| Method | Effect |
| --- | --- |
| `bash scripts/phase2_night.sh stop` | Waits for the current run to finish (up to about 35 minutes on the M4), then shuts down the backend |
| `bash scripts/phase2_night.sh stop --now` | Aborts the current run now; the backend drains and shuts down; the run is redone next time |
| Unplug the charger or close the lid | The watchdog sees battery power or sleep, aborts the current run and shuts down the backend; the run is redone next time |

**Check any time**

```bash
bash scripts/phase2_night.sh status
```

```bash
bash scripts/phase2_night.sh log
```

`status` lists each stage as done, partial (e.g. `23/57 runs`) or pending, with the
rough time remaining.

**What every start does automatically**

1. **Checks the environment against the lock.**
   - Before any probe, calibration, smoke or measurement, `relock "reason"` can record a
     change, and the pinned build and model are re-verified.
   - Once such observations exist (including incomplete logs), an environment change
     needs a new campaign. This prevents reusing old calibration or mixing old and new
     implementations.
2. **Checks the machine.**
   - AC power, at least 6 GiB of disk and at least 3 GiB of available memory (a warning
     below 6 GiB);
   - that Low Power Mode is off.
3. **Starts a fresh backend.**
   - Shuts down a backend left from last time.
   - Deletes only snapshot files this project created (`phase2-*`, `probe-*`).
   - Starts a fresh backend. Its process identity differs, so "state uncertain" marks left
     by an interruption do not block this run.
4. **Advances in order:**
   1. probes;
   2. calibration, smoke and wait distribution;
   3. dev aging screen (normal and RAM groups), then the screen summary;
   4. choose aging from the measured joint rows;
   5. Q/C/CP/TTL confirmation at the same aging, then the dev summary;
   6. freeze, then **stop for review**;
   7. eval, then the core extra trials, then the split summaries;
   8. sensitivity.

   Each session warms up before its first timed run. A failed warm-up is retried once;
   on a second failure the session stops instead of timing with an unconfirmed thermal
   state.

**Guarantees on interruption and resume**

- A run is complete only once it has written a terminal record. An interrupted run's logs
  are moved to `raw/interrupted/`, it is rerun next time, and it never enters a summary.
- An interrupted probe group's logs move to `raw-interrupted-<time>/` and the whole group
  reruns (the longest group takes about 70 minutes).
- If a stage fails, the backend is restarted and the stage retried once. A second failure
  ends the session, with the reason written to `campaign/sessions.jsonl`.
- At summary time, a run with a clear wall-clock jump (the machine slept) is flagged and
  not paired. Battery runs are not paired either.

**Review after dev**

The campaign stops at `freeze-review`. Review `runs/phase2-<platform>/manifests/frozen.json`
and the dev summary together; once confirmed, run:

```bash
bash scripts/phase2_night.sh approve
```

Approval records the frozen file's hash. Any later change to the file needs a new approval.

**Total time.** Use `status` with the current manifests and remaining runs. T0–T3 is
roughly 42–62 h, sensitivity extra. The new implementation was not yet measured, so no
fixed number of nights can be promised.

## 1. Model-free checks

```bash
.venv/bin/python -m pytest tests -q
bash -n scripts/phase2.sh
bash scripts/phase2.sh
```

## 2. T0: backend, probes, calibration, smoke

**The main configuration.**
- Partitioned KV, `P=2`, `C=24576`, 12288 tokens per slot, f16 KV.
- The KV pool is about 3.38 GiB, and the weights need about another 2.3 GiB.
- `PLATFORM_ID` is a stable platform identifier such as `m4-air-16g` or `t4-g4dn-xlarge`.
  Probes, profile, dev and eval on one platform must use the same value.

```bash
export PLATFORM_ID=m4-air-16g
.venv/bin/python -c 'import psutil; m=psutil.virtual_memory(); s=psutil.swap_memory(); print({"available_gib":round(m.available/2**30,2),"swap_used_gib":round(s.used/2**30,2)})'
```

```bash
.venv/bin/python -m phase2.server --parallel 2 --ctx-size 24576 --cache-ram 0 --port 18080 --results-dir runs/phase2-backend
```

```bash
.venv/bin/python -m phase2.probes --suite all --results-dir runs/phase2-probes --platform-id "$PLATFORM_ID"
```

**What the probes check:**
- **affinity:** whether explicit `id_slot` really uses both slots;
- **incremental recompute:** the cost of recomputing a 0.5K–4K private suffix after it is
  overwritten (K=1/K=2);
- **interference:** the stall of decoding on the other slot during save/restore;
- **restore:** whether 64 greedy tokens match after same-slot and cross-slot restore.

First look at rows with `status != ok` in `runs/phase2-probes/raw/probes.jsonl`. If
restore shows a `mismatch`, **do not** enable the checkpoint branch.

First calibration (no wait samples yet):

```bash
.venv/bin/python -m phase2.probes calibrate --input runs/phase2-probes/raw/probes.jsonl --output runs/phase2-profiles/$PLATFORM_ID-probe.json
```

Prefill cost uses only the overwrite / erase conditions. The retain condition computes
only about 2 tokens, and its fixed overhead would inflate the per-token cost. After
generating, check:
- that the `1` and `2` buckets of `prefill` both exist;
- that `save` / `restore` are non-empty and plausible (M4 about 0.37 s / 0.35 s per 8K;
  T4 save about 9 s).

Smoke (2 tasks, to confirm that slots, logs and the checkpoint path all work; not used in
the report):

```bash
.venv/bin/python -m phase2.bench --workload runs/final/workload_dev.json --results-dir runs/phase2-smoke --policies fifo,joint,joint-checkpoint --cost-profile runs/phase2-profiles/$PLATFORM_ID-probe.json --platform-id "$PLATFORM_ID" --seeds 1 --num-jobs 2 --gap lognormal:2:1 --arrival scaled:1
```

Collect the dev wait distribution (one dev FIFO run, about 17 minutes). The cost policies'
leases need a wait distribution. Only dev observations form the prior; during eval the
policy keeps learning online from finished waits.

```bash
.venv/bin/python -m phase2.bench --workload runs/final/workload_dev.json --results-dir runs/phase2-waits --policies fifo --cost-profile runs/phase2-profiles/$PLATFORM_ID-probe.json --platform-id "$PLATFORM_ID" --seeds 100 --num-jobs 10 --gap lognormal:2:1 --arrival scaled:1
```

```bash
.venv/bin/python -m phase2.probes calibrate --input runs/phase2-probes/raw/probes.jsonl --waits-input runs/phase2-waits/raw/policy_decisions.jsonl --output runs/phase2-profiles/$PLATFORM_ID.json
```

Dev, eval and sensitivity then all use this same profile.

**What calibration writes:** `prefill_min_tokens=128`, `online_prefill=true` and
`online_io=true`.
- A prefill with fewer than 128 actually computed tokens does not update the slope.
  Decode and finished waits still learn.
- Every successful save/restore updates its slope by EWMA from the store's measured
  backend call time. Probes measure on an idle server; in the pilot the in-run cost was
  about 1.8× that.

**Ablations and seeds.**
- An ablation with frozen prefill must set `online_prefill` to false in a separate profile
  before any dev measurement, and must be confirmed and frozen on its own. It cannot be
  switched on during eval.
- Seed 100 is not one of the formal dev/eval seeds.

## 3. T1: dev parameter selection

The aging threshold decides how long a new task can be blocked by leases. It is the main
knob between throughput and short-task fairness and must be chosen on dev. The grid is
about 2/4/8 × the single-step model time:
- on the M4 (about 9.5 s per step), `20,40,80`;
- on the T4 (about 1.9 s), `4,8,16`.

```bash
bash scripts/phase2.sh create --phase dev --workload runs/final/workload_dev.json --cost-profile runs/phase2-profiles/$PLATFORM_ID.json --platform-id "$PLATFORM_ID" --results-dir runs/phase2-$PLATFORM_ID-r2/dev --manifest artifacts/phase2-$PLATFORM_ID-dev.json --seeds 101,102,103 --trials 1 --aging-grid 20,40,80 --minutes-per-task 1.7
```

The generic create command above generates candidates for every aging and suits a
hand-picked matrix. The campaign's two-step scheme avoids running everything, and it does
these steps automatically:
1. **Screen.** Thirteen configs: the five fixed `(K,J)` pairs, FIFO, affinity,
   affinity-fixed `J=2,3`, joint at three agings, and RAM-cache. `fixed-k1-j1` is the
   isolated reference.
2. **Choose aging.** After the screen summary, `phase2.freeze.select_shared(dev_root)`
   chooses aging read-only.
3. **Confirm.** The same seeds and profile generate `dev-confirm.json`, with six configs:
   queue/cache/checkpoint at that aging, and TTL 10/15/30.

For example, if the screen chooses aging 80, the confirmation matrix alone is:

```bash
bash scripts/phase2.sh create --phase dev --workload runs/final/workload_dev.json --cost-profile runs/phase2-profiles/$PLATFORM_ID.json --platform-id "$PLATFORM_ID" --results-dir runs/phase2-$PLATFORM_ID-r2/dev --manifest artifacts/phase2-$PLATFORM_ID-dev-confirm.json --seeds 101,102,103 --trials 1 --aging-grid 80 --only-labels queue-a80,cache-a80,joint-checkpoint-a80,ttl-10-a80,ttl-15-a80,ttl-30-a80
```

80 is only an example. The real value must come from this round's screen, never from an
old eval ranking.

Run the normal group first, then the RAM-cache group separately:

```bash
bash scripts/phase2.sh run artifacts/phase2-$PLATFORM_ID-dev.json --server-group P2-C24576-RAM0-idle0 --budget-s 86400 --execute
```

The RAM-cache group needs a separate backend with host cache. `slot erase` does not clear
llama.cpp's host prompt cache, so later runs of the same tasks would start warm. The
executor therefore requires `--restart-server ram-cache`, which restarts the backend
before each run:

```bash
.venv/bin/python -m phase2.server stop
.venv/bin/python -m phase2.server --parallel 2 --ctx-size 24576 --cache-ram 3072 --cache-idle-slots --port 18080 --results-dir runs/phase2-backend-ram
.venv/bin/python -m phase2.probes --suite ram-cache --results-dir runs/phase2-probes-ram --platform-id "$PLATFORM_ID"
bash scripts/phase2.sh run artifacts/phase2-$PLATFORM_ID-dev.json --server-group P2-C24576-RAM3072-idle1 --restart-server ram-cache --budget-s 86400 --execute
```

On the M4 the RAM-cache group uses an extra 3 GiB of host memory, so check available
memory first. If it swaps, record it and mark it not comparable in the report; never fill
the table with paging results.

Summarize dev. `--short-cutoff 782.5` is the median expected output tokens of the 10 dev
tasks:

```bash
bash scripts/phase2.sh summarize --results-dir runs/phase2-$PLATFORM_ID-r2/dev --short-cutoff 782.5 --baseline fifo
```

## 4. Freeze the eval configuration

`phase2.freeze` chooses best-fixed, B2F, aging, the TTL at the same aging, and the final
system.
- The queue rule keeps the value the chosen joint was measured with; it is never
  transplanted from another policy's score.
- Each output row carries the full config, parameter hash, dev run IDs, seed/trial and
  input evidence.
- It stops outright on a missing same-aging confirmation, a missing fairness reference,
  incomplete coverage or a budget mismatch.
- The aging rule picks the fastest candidate whose short-task slowdown is no worse than
  best-fixed's. If none qualifies, it picks the fairest, explicitly flagged as failing
  the fairness gate.

```bash
bash scripts/phase2.sh freeze --dev-results runs/phase2-$PLATFORM_ID-r2/dev --dev-workload runs/final/workload_dev.json --checkpoint-budget-gib 4 --output artifacts/phase2-$PLATFORM_ID-frozen.json
```

Review the notes, evidence and full configs.
- If the restore probe failed, fix the mechanism and re-measure first. Never hand-edit the
  frozen file to create a "confirmed" config.
- After freezing, never change parameters or pick seeds based on eval.

## 5. T2/T3: eval

`--trial-labels` makes key rows run the full `--trials`, and other rows only trial 0.
The eval manifest adds an `isolated` (K=1, J=1) run first, which is the denominator of
slowdown.

```bash
bash scripts/phase2.sh create --phase eval --workload runs/final/workload_eval.json --cost-profile runs/phase2-profiles/$PLATFORM_ID.json --frozen-config artifacts/phase2-$PLATFORM_ID-frozen.json --platform-id "$PLATFORM_ID" --results-dir runs/phase2-$PLATFORM_ID-r2/eval --manifest artifacts/phase2-$PLATFORM_ID-eval.json --seeds 1101,1102,1103 --trials 1 --minutes-per-task 1.7
```

**Arrival rate.** The T4's historical main rate is `scaled:3.5`: add `--arrival
scaled:3.5` to the dev, eval and sensitivity create commands. The M4 uses the default
`scaled:1`.

**Running.** It works like dev: the normal group first, then the RAM-cache group
(`--restart-server ram-cache`). With a limited budget, `--labels` can run some rows only,
but they must finish in the pre-arranged order; never decide what to add after seeing
results.

If best-fixed happens to be FIFO, the frozen file does not repeat it; summarize with only
`--baseline fifo`.

**Core extra trials.**
- On the M4 this is 1 extra trial, so trials 0 and 1; summarize with
  `--confirmatory-trials 0,1`.
- They use the frozen `final_label` plus whichever of `fifo,best-fixed,affinity-fixed-best`
  exist; joint is not hard-coded. The campaign handles missing labels and duplicates.

**About the seeds.** The old seed 1001 belongs to the pilot. The new seeds provide only
new load permutations; the task pool is still the same 20 eval tasks, so generalization
to new tasks cannot be claimed.

## 6. T4: sensitivity

```bash
bash scripts/phase2.sh create --phase sensitivity --workload runs/final/workload_eval.json --cost-profile runs/phase2-profiles/$PLATFORM_ID.json --frozen-config artifacts/phase2-$PLATFORM_ID-frozen.json --platform-id "$PLATFORM_ID" --results-dir runs/phase2-$PLATFORM_ID-r2/sensitivity --manifest artifacts/phase2-$PLATFORM_ID-sens.json --seeds 2101,2102 --only-labels fifo,affinity-fixed-best,joint-checkpoint,fixed-ttl --conditions gap-recorded,gap-long --no-isolated --minutes-per-task 1.7
```

**Available conditions:** `main`, `arrival-half`, `arrival-double`, `gap-zero`,
`gap-recorded`, `gap-long` (fixed 15 s), `mix-short`, `mix-long`.
- **Mix conditions** split the eval tasks by the frozen short cutoff, and the favored class
  appears 3× in the task pool (about 75/25).
  - The mix file is written into the results folder and its hash is fixed in the
    manifest.
  - Each task is still the original frozen trajectory, and slowdown reuses the source
    task's isolated reference.
- **Expected low-benefit conditions:** `gap-zero` and `arrival-half`.
- **`gap-long`** puts the dev wait prior out of line with the actual waits, so it shows
  whether the policy degrades when its predictions are wrong.

Replace joint-checkpoint in the example with this round's frozen `final_label`, and drop
the B2F label if freeze did not keep it; the campaign reads these values automatically.
`gap-recorded` is the recorded call interval, still not pure Retail tool latency.

The arrival-rate and mix conditions on the M4 run as a separate supplement:
`scripts/phase2/4_rate_mix_run.sh`, pre-registered in
[6_RATE_MIX_PREREG.md](6_RATE_MIX_PREREG.md).

## 7. Summaries, figures and definitions

`SHORT_CUTOFF` is the frozen file's `short_cutoff` (the dev median, 782.5 for the current
dev workload):

```bash
export SHORT_CUTOFF=782.5
```

```bash
bash scripts/phase2.sh summarize --results-dir runs/phase2-$PLATFORM_ID-r2/eval --short-cutoff "$SHORT_CUTOFF" --baseline fifo --baseline best-fixed --baseline affinity-fixed-best --baseline affinity --bootstrap-samples 2000 --bootstrap-seed 291
```

```bash
bash scripts/phase2.sh plot --results-dir runs/phase2-$PLATFORM_ID-r2/eval
```

When a sensitivity folder has no isolated run, add `--reference-dir
runs/phase2-$PLATFORM_ID-r2/eval`. The isolated reference of the same eval workload can be
reused, but never across platform, P/C, model or KV precision.

| File | Content |
| --- | --- |
| `summary/run_summary.csv` | Per run: completions/failures, throughput, JCT, fairness, hit rate, recompute fraction, peak RSS, minimum available memory, swap growth, GPU memory, lease/evict/aged/save/restore counts, ready wait, output match rate, battery flag |
| `summary/policy_summary.csv` | Descriptive summary over all observed trials; cannot be used to infer wins across different repeat counts |
| `summary/policy_summary_t0.csv`, `paired_t0.csv` | Main comparison using T0 only for every policy, with seed/trial N and exclusion reasons |
| `summary/policy_summary_confirmatory.csv`, `paired_confirmatory.csv` | Strictly complete core repeat comparison; by default both sides need the configured trials |
| `summary/paired.csv` | Keeps the original strict all-observed-trial matching and its diagnostics; never silently intersects |
| `summary/jobs.csv`, `summary/requests.csv` | Task-level and request-level derived metrics |
| `summary/REPORT.md` | Definitions and main tables |
| `figures/` | Default figures use T0; `*_confirmatory` figures use complete repeats, faceted by the actual seed/trial sets with N |

Key definitions:
- **Paired ratios.** `throughput_ratio > 1` and `jct_ratio < 1` favor the candidate policy.
  - Pairing requires the same seed, the same planned work, full completion, matching
    resources and matching trial sets.
  - Failed, interrupted, mock and battery runs are listed with an exclusion reason and
    produce no speedup.
- **Output match rate.** Decoding is greedy and output lengths are fixed, so a task's
  generated-token digest should match its isolated run.
  - Different batching can cause legitimate numerical differences, so this is a
    diagnostic across policies, not a hard gate.
  - If a policy (especially checkpoint restore) is clearly below FIFO, investigate state
    restore.
- **Memory.** llama.cpp preallocates the KV pool, so RSS hardly changes with load. Report
  peak RSS, minimum available memory and swap growth together; the RAM-cache row has an
  extra 3 GiB of host cache.
- **Policy overhead.**
  - `decision_cpu_s` is the CPU time to choose requests and slots.
  - save/restore counts the round-trip time and bytes of successful operations.
  - Waits caused by leases appear in the ready wait.
- **Replay.** All experiments are finite-stream replays. A completion is a completed
  replay, not semantic success.

## 8. Timeouts and checkpoint recovery boundaries

There are three different limits:
- `--request-timeout-s` is the deadline of one HTTP request, for completion and
  save/restore;
- `--timeout-s` is the deadline of each trajectory;
- `plan run --budget-s` only decides whether to start the next run. It never kills a
  running benchmark.

After an HTTP disconnect, I/O timeout or cancellation, the backend may still be writing or
inferring. The scheduler and store then:
- mark the state uncertain and stop reusing the slot;
- keep the same backend process from being used by the next sweep until it restarts.

**Manual kills.** If you kill a benchmark by hand, check or restart the server before
continuing, and do not delete snapshots that may still be written.

**Server errors.** Deterministic server errors are logged and fall back to recompute.
llama.cpp's slot save runs synchronously in the main loop, so the policy charges "the
whole server stalls during a save" and never claims asynchronous chunked offload.

## 9. Policies at a glance

`fifo`, `fixed` and `ram-cache` pass no `id_slot`, so llama-server places requests by
longest prefix (as in Phase 1). Every other policy manages slot ownership explicitly.

| Policy | Request order | Slot yield rule |
| --- | --- | --- |
| `affinity` / `affinity-fixed` | FIFO | own slot > empty slot > LRU victim |
| `ttl` | FIFO | keep for a fixed number of seconds after the tool wait starts |
| `queue` / `attained` | continuation first / least attained service first, both with aging | LRU |
| `cache` | FIFO | cost lease: expected saving of keeping vs making ready requests wait |
| `joint` | continuation first + aging | cost lease |
| `joint-checkpoint` | as joint | the lease is compared against the actually executable save/recompute action; a save must pass the same budget/state/net-benefit conditions, is re-evaluated if it becomes invalid at execution, and a return restores by net benefit |

Under FIFO order, if the head is blocked by a lease, the lease owner can use its reserved
slot directly when it returns (the head cannot use that slot anyway). Without this rule,
the TTL and cache-only baselines would wait for their own lease to expire and become
unfairly weak baselines.
