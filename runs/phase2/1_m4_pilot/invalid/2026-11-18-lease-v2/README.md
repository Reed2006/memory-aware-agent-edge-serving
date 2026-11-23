# Lease model v1 runs (set aside 2026-11-18)

These 15 dev runs (cache, joint-a20/a40/a80, joint-checkpoint; seeds 101-103) used the
first cost-lease value, `p(h) x loss - w x h`, with horizons Q25-Q90 capped at 10 s.
On dev a fixed 10 s TTL beat it (3-seed mean task completion 173 s vs cache 180 s;
seed 103: cache 37 non-overdue evictions, 47,510 recomputed tokens; ttl-10 1 and 42,200).

The value was revised (docs/PHASE2_PLAN.md section 13.6) to
`p(h) x (1 + w) x loss - w x E[min(remaining wait, h)]`, horizons Q25-Q99, cap 60 s
for cost-lease policies only. Other policies keep their configs and param hashes,
so their dev runs remain valid. These runs stay here as the v1 ablation; they are
never mixed into the dev summary that freezes the eval config.

Contents:
- dev/main/<label>/S<seed>-T0: the v1 runs (all completed, no failures; seed-101
  runs were the 2026-11-17 Q90 reruns, done right after a 2.5 h idle period, so
  they ran on a cool machine).
- manifests/dev.json, manifests/frozen.json: the v1 dev manifest and the frozen
  config it produced (never approved; no eval run used it).
- dev-summary-v1/: the dev summary and figures over all 60 v1 runs.
- campaign-done/dev-summary: the stage marker, removed so the summary is redone.
