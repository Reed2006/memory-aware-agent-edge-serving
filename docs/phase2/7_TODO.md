# Phase 2 TODO and handoff

Written 2026-11-22. The final report is due **2026-12-10**. The file map is
[0_README.md](0_README.md); results so far are in
[4_FAIRNESS_EFFICIENCY.md](4_FAIRNESS_EFFICIENCY.md).

## Where things stand

| Item | State |
| --- | --- |
| M4 main campaign (probes, dev, freeze, eval, sensitivity) | Done: 116 runs, 0 failed jobs |
| M4 frontier supplement (3 new heavy seeds) | Done: 9 runs |
| M4 arrival-rate / mix supplement (48 runs) | Running since 2026-11-22 20:34; about 24–25 h in total |
| Second platform (T4) | Not started |
| Final report | Not started |

## A. The second server (collaborator)

The plan is the course AWS **g4dn.xlarge with one T4 GPU**, as in Phase 1. Phase 1 setup
scripts are in `runs/aws-gpu-pilot/scripts/`. If the GPU is different, everything below
still applies, but the `t4` preset in `phase2/campaign.py` was written for the T4. A
different GPU needs its own preset, and that is a code change (see A.5).

The T4 is the point of the second platform. On the M4, recompute is expensive and saving is
cheap. On the T4 it is the reverse:

| Cost | M4 | T4 |
| --- | ---: | ---: |
| Cold 8K recompute | ~44 s | ~5.1 s |
| 8K save / restore | 0.37 / 0.35 s | 8.95 / 0.47 s |

So the same frozen rules should make different decisions there.

### A.1 Set up

1. Clone the `phase2` branch:

   ```bash
   git clone -b phase2 https://github.com/Reed2006/memory-aware-agent-edge-serving.git
   ```

2. Create the Python environment: Python 3.13, then
   `python3.13 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt`.
3. Build llama.cpp **commit `620d2225c`** (build 11669) with CUDA, and put `llama-server` on
   `PATH`. The campaign refuses any other build (`llama-server --version` is checked).
4. Put the model at `models/Qwen3-4B-Q4_K_M.gguf`. Its sha256 must equal the one in
   `runs/final/model.sha256` (`7485fe6f…`).
5. Run the model-free tests: `.venv/bin/python -m pytest -q tests`. Expect 273 passed.
6. Check the course credit left before starting. Then estimate the bill as hours × the
   instance's current hourly price.

**Phase 2 has only ever run on macOS.** These parts are platform-specific; check them in
the first smoke stage:
- the power check (`pmset`, absent on Linux; the code tolerates its absence);
- `caffeinate` (absent; run inside `tmux` instead);
- the `nvidia-smi` metadata (only used on the T4).

### A.2 Run the T4 campaign

The campaign does probes, calibration, waits, dev, freeze, eval, extra trials and
sensitivity, resumably. The `t4` preset sets:
- arrival `scaled:3.5`;
- aging grid 4/8/16 s;
- a 4 GiB checkpoint budget.

```bash
.venv/bin/python -m phase2.campaign --root runs/phase2/5_t4_campaign init t4 t4-g4dn-xlarge
```

```bash
tmux new -s phase2
```

```bash
bash scripts/phase2_night.sh run --until 07:30
```

Other commands:
- `bash scripts/phase2_night.sh status` shows the stage table.
- The campaign **stops at `freeze-review`**. Review `runs/phase2/5_t4_campaign/manifests/frozen.json`
  and the dev summary, then run `bash scripts/phase2_night.sh approve` and `run` again.
- Do **not** use `--reuse-from` with the M4 pilot: it is a different platform.

Expected time: about 18–26 h (`2_RUNBOOK.md` section 0; the preset assumes 0.48 min per
task).

### A.3 Run the two supplements on the T4

These make the T4 results directly comparable with the M4 supplements.

**1. Frontier supplement.**
- Policies: B2F, joint, joint-checkpoint.
- Seeds: 3116, 3129 and 3120, the same as on the M4.
- Run `scripts/phase2/3_frontier_seeds.py` with the T4 profile and `--arrival scaled:3.5`.
  This records how heavy those seeds are on the T4.
- If they are light there, also add the top 3 of the T4 screen. Write that down in a
  pre-registration first.

**2. Rate/mix supplement.**
- Policies: fifo, B2F, joint and joint-checkpoint. Seeds: 1101–1103.
- Conditions: `arrival-half` (`scaled:1.75`), `arrival-double` (`scaled:7`), `mix-short` and
  `mix-long`.
- Create the manifests with `phase2.plan create --phase sensitivity --arrival scaled:3.5 ...`,
  using the T4 frozen config and profile.

**Before running either one:**
- The drivers `3_frontier_run.sh` and `4_rate_mix_run.sh` are M4-specific: the platform id,
  `R=runs/phase2/2_m4_campaign`, and the `pmset` battery check. Copy them to new
  `scripts/phase2/5_t4_*` scripts and adapt them. **Do not edit the M4 copies**:
  `4_rate_mix_run.sh` may still be running on the M4.
- If freeze on the T4 does not keep `affinity-fixed-best` (on the T4 the best fixed J was 4
  in Phase 1), use `best-fixed` in its place.
- Write a short pre-registration, like `5_` / `6_`, before the first run.

### A.4 Send results back

- **What to commit:** `runs/phase2/5_t4_campaign/` (and `6_t4_frontier/`, `7_t4_rate_mix/`), plus
  any new scripts and docs, to a branch such as `phase2-t4`.
- **Size:** no file over 100 MB. Snapshot files are deleted after each run; check that
  `backend/` holds no `phase2-*` snapshots.
- **Report:** failed or interrupted runs as they are. They stay in `raw/interrupted/` or
  `invalid/`; never delete them.

### A.5 If something must change in the code

The lock pins `phase1/*.py`, `phase2/*.py`, `scripts/phase2.sh`, `scripts/phase2_night.sh` and
`requirements-lock.txt`.
- Before any probe or measurement, `bash scripts/phase2_night.sh relock "reason"` records a
  change.
- After measurements exist, a code change needs a new campaign folder.
- If a fix touches scheduling logic, say so: M4 and T4 would then run different code, and
  the cross-platform comparison must state it.

## B. What this Mac still lacks

1. **Finish the rate/mix supplement.**
   - Tonight it runs to 09:00. Heavy conditions go first: arrival-double, then mix-long,
     then mix-short, then arrival-half.
   - It needs about 1–2 more nights. Restart it with:

     ```bash
     nohup caffeinate -is bash scripts/phase2/4_rate_mix_run.sh --until 09:00 >/dev/null 2>&1 &
     ```

     Finished runs are skipped.
   - When all 48 are done, run `scripts/phase2/4_rate_mix_check.py` and add the results to
     `4_FAIRNESS_EFFICIENCY.md`.
2. **Repeat the key frontier cells.** Run trial 1 of `joint-checkpoint` and B2F on seeds
   3116, 3129 and 3120: 6 runs, about 2 h. The efficiency price of jc (+4.8% JCT, CI
   [0.97, 1.10]) is inside the ~10% trial noise; a repeat gives it a variance estimate.
3. **Optional: J = 3 with checkpoints.** This interpolates between B2F and jc. It needs a
   new policy config, so it goes in a new campaign folder.
4. **Optional: a small live Retail correctness check** (plan section 9.3). It is not
   implemented. If it is skipped, the report states the limit: replay with fixed output
   lengths and token-digest agreement cannot show live semantic quality.
5. **Merge `phase2` into `main`** once the collaborator has checked out the branch.

## C. Expected conclusions

### C.1 What the M4 already shows (keep these claims)

1. **Avoiding recompute is a free lunch.** Policies that waste prefill (fifo, affinity,
   queue, J ≥ 3 without checkpoints) are worse on efficiency and fairness at once.
2. **The pre-registered final (`joint`) does not beat the re-tuned simple baselines.**
   Admission-capped affinity (B2F, J = 2) is enough for efficiency.
3. **Under contention there is a real but uneven trade-off.** Checkpointed sharing buys a
   large proportional-fairness gain (P95 slowdown −37% to −52%) for a small, possibly
   zero, efficiency price. Sharing without checkpoints is dominated.
4. **"Fair" depends on the definition.** B2F has the smaller absolute tail delay; sharing
   has the smaller slowdown. Jain's index is fragile to the slowdown reference, because a
   cold shared prefix costs about 30 s.

### C.2 What the rate/mix supplement should show (pre-registered R1–R5)

- **R1, low load:** at half rate the efficient policies tie, and the gain over fifo
  shrinks. This is the "little benefit" case the course asks for.
- **R2, overload:** at double rate the gain over fifo grows.
- **R3, the trade-off under heavy rates:** at double rate jc has a lower P95 slowdown than
  B2F, and no lower JCT.
- **R4, the negative result holds:** joint's JCT is above B2F's at double rate.
- **R5, a long-heavy mix acts like more load.**

### C.3 What the T4 should show (hypotheses written before any T4 data)

- **T1, keeping KV matters less.** Recompute is about 9× cheaper, so the gap between fifo
  and the KV-keeping policies is smaller than on the M4.
- **T2, the best fixed cap is higher.** In Phase 1 the best fixed J on the T4 was 4, and
  J = 2 lost to FIFO on one repeat. B2F with J = 2 may underuse the GPU, and freeze may
  pick J = 3 or 4.
- **T3, the cost model refuses checkpoints.** A whole-slot save costs about 9 s, while
  recomputing a private suffix takes under 1 s. So `joint-checkpoint` should save rarely and
  behave like `joint`. This tests that the cost model adapts, rather than that
  checkpoints help.
- **T4, cheap preemption makes sharing pay without checkpoints.** The M4 thesis is that
  sharing pays only when preemption is cheap. On the T4 recompute itself is cheap, so
  `joint` should be closer to the best simple baseline than on the M4.
- **T5, the fairness reading is unchanged.** Capped run-to-completion keeps the smaller
  absolute tail delay; sharing keeps the smaller slowdown.

If T3 and T4 hold, the cross-platform conclusion is: **which mechanism makes sharing
affordable depends on the platform's measured cost ratio, and a cost-aware scheduler picks
it without retuning.** If they fail, report that the simple capped baseline is the robust
choice on both platforms.

## D. How to form the cross-platform conclusion

**1. Hold everything fixed except the platform.**
- Same code commit, model file, quantization, KV precision, P = 2, ctx 24576.
- Same task pool and seeds.
- Same freeze rules.

**2. Let the platform-specific parts come from each platform's own measurements:**
- the cost profile (probes plus waits);
- the dev selection (aging, best fixed J, TTL);
- the arrival rate, scaled to platform speed (M4 `scaled:1`, T4 `scaled:3.5`) so that
  utilization is comparable. Check this with the a priori load indicator under each
  platform's own profile.

**3. Never compare absolute times across platforms.** Compare paired ratios within each
platform, each against that platform's own tuned baselines:
- policy / best-fixed;
- policy / B2F;
- policy / fifo.

These cover JCT and throughput, plus the fairness metrics: P95 and short/long slowdown,
P95 contention delay, and maximum admission wait.

**4. Explain the differences with mechanism measurements:**
- per-token recompute cost;
- save and restore cost;
- the ratio recompute / (save + restore);
- decision counts per run (leases, no_lease, evictions, saves, restores);
- wasted prefill fraction.

The claim is about whether decisions move in the direction the costs predict.

**5. Put it in one table and one figure:**
- **Table:** per platform, the cost ratio, the frozen choices (J, aging, TTL), and the
  paired ratios of joint and jc vs the best simple baseline, with 95% seed-cluster CIs.
- **Figure:** for each platform, the efficiency–fairness plane (mean JCT vs P95 slowdown),
  with every policy marked.

**6. Scope the claim.** Two platforms are two points, not a curve. Write "consistent with"
the mechanism, not a general law. Report session and noise limits: one trial per
supplement cell, and about 10% trial noise for the sharing policies.

## E. Final report checklist (due 12-10)

- [ ] Question, design, setup and limits from `1_PLAN.md`; results from
  `4_FAIRNESS_EFFICIENCY.md` plus the rate/mix and T4 results.
- [ ] Main table: eval confirmatory paired ratios, M4 and T4.
- [ ] Fairness section: slowdown and contention delay; Jain only with its δ = 30 check.
- [ ] Boundaries: arrival rate, mix and tool waits, including the low-benefit cases.
- [ ] Memory: peak RSS about 3.5 GiB (KV pool preallocated); RAM-cache 5–6.5 GiB, marked
  resource-incomparable; swap growth per run.
- [ ] Overhead: decision CPU below 0.01% of makespan; checkpoint I/O counted as server
  stall.
- [ ] Reproducibility: commands from `0_README.md`, the lock, manifests, and failed/invalid
  runs listed.
- [ ] The Phase 1 LaTeX framework is in `report/`.
