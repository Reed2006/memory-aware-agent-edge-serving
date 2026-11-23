# Superseded dev runs: lease horizon candidates (2026-11-17)

These seed-101 dev runs used lease horizons from remaining-wait quantiles {0.25, 0.5, 0.75}.
On dev, a fixed 10 s TTL reached a 99% full-history hit rate while the cost lease reached
90%: the Q75 horizon (~3.0 s on the M4) could not cover the wait tail (P90 ~4.8 s).
The candidate set became {0.25, 0.5, 0.75, 0.9} (`phase2/costs.py: LEASE_QUANTILES`),
a dev-stage method change made before any parameter was frozen.

Every policy with `cache_policy=cost` is rerun: cache, joint-a20/a40/a80, joint-checkpoint.
(joint-checkpoint priced every eviction against a cheaper checkpoint and never leased, but
it falls back to lease pricing when the snapshot budget is full, so it is rerun too.)
Policies without cost leases are unaffected and keep their runs.

Results of these runs, for the record (seed 101, all 10/10 completed):
cache 0.763 tasks/min, hit 83%; joint-a20 0.779, 90%; joint-a40 0.790, 90%;
joint-a80 0.779, 90%; joint-checkpoint 0.797, 99%.
