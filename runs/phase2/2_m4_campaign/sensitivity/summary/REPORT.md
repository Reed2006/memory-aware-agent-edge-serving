# Phase 2 replay summary

Completion means replay steps completed, not semantic task success.
Fixed output lengths and token-digest agreement do not establish live semantic correctness. Output match is a diagnostic against matched isolated runs; a missing reference is reported as nan.
Trials are averaged within seed, then seeds receive equal weight. Latency percentiles describe completed jobs only.
Paired ratios require equal planned and completed work, matching resources/inputs, and matching trial sets.
Bootstrap resamples seed clusters (2000 draws; RNG seed 291); intervals need at least two seeds.
Throughput ratio > 1 favors the policy; JCT ratio < 1 favors the policy.
Slowdown uses only matched isolated prompt/decode timings plus each job's actual tool gaps.
Mock-transport runs are marked synthetic_performance and excluded from paired performance claims.
Aborted/incomplete runs retain their planned denominator and any completed records; they cannot establish a paired gain.
`paired.csv` retains strict equality of all observed trial sets as a diagnostic. `policy_summary.csv` retains all-trial descriptive aggregates and is not the mixed-trial ablation table.
Short-task cutoff (expected output tokens, frozen on dev): 782.5.

## T0 ablations

| Policy | Seeds / runs | Trial IDs | Completed / submitted | Tasks/min | Mean JCT (s) | P95 (s) | Short / long slowdown | Output match to isolated |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| affinity-fixed-best (654f202a) | 2 / 2 | [0] | 40 / 40 | 0.6331 | 561.6 | 938.3 | 10.7 / 2.23 | 0.875 |
| fifo (42eadd78) | 2 / 2 | [0] | 40 / 40 | 0.5395 | 693.1 | 1416 | 4.56 / 3.7 | 0.65 |
| joint (4ed3bc99) | 2 / 2 | [0] | 40 / 40 | 0.5582 | 658.7 | 1343 | 5.04 / 3.43 | 0.625 |
| fixed-ttl (f26b687b) | 2 / 2 | [0] | 40 / 40 | 0.4667 | 941 | 1790 | 5.95 / 5.08 | 0.65 |
| affinity-fixed-best (654f202a) | 2 / 2 | [0] | 40 / 40 | 1.18 | 131.5 | 205.6 | 8.5 / 2.76 | 0.975 |
| fifo (42eadd78) | 2 / 2 | [0] | 40 / 40 | 0.6467 | 499.8 | 1117 | 17.2 / 10.9 | 0.775 |
| joint (4ed3bc99) | 2 / 2 | [0] | 40 / 40 | 0.9396 | 236.7 | 498.1 | 12 / 4.84 | 0.9 |
| fixed-ttl (f26b687b) | 2 / 2 | [0] | 40 / 40 | 1 | 199.9 | 410.7 | 7.81 / 4.62 | 0.925 |

Exact seed IDs and per-seed trial IDs are in this scope's policy CSV. Rows with different seed sets are descriptive; use the paired table for comparisons.

Only trial 0 is used for every strategy and baseline in this scope. Additional baseline repeats do not remove T0-only ablations.

Paired results: `paired_t0.csv`. Each row uses its explicitly listed seed/trial pairs; excluded samples are not replaced by an intersection.

| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| affinity-fixed-best | fifo | 2 / 0 | 2 | 1.174 | 0.7847 |  |
| joint | fifo | 2 / 0 | 2 | 1.035 | 0.9261 |  |
| fixed-ttl | fifo | 2 / 0 | 2 | 0.8647 | 1.386 |  |
| affinity-fixed-best | fifo | 2 / 0 | 2 | 1.903 | 0.3432 |  |
| joint | fifo | 2 / 0 | 2 | 1.474 | 0.4676 |  |
| fixed-ttl | fifo | 2 / 0 | 2 | 1.583 | 0.4235 |  |
| affinity-fixed-best | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fifo | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| joint | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fixed-ttl | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| affinity-fixed-best | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fifo | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| joint | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fixed-ttl | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fifo | affinity-fixed-best | 2 / 0 | 2 | 0.8524 | 1.301 |  |
| joint | affinity-fixed-best | 2 / 0 | 2 | 0.8815 | 1.185 |  |
| fixed-ttl | affinity-fixed-best | 2 / 0 | 2 | 0.7376 | 1.827 |  |
| fifo | affinity-fixed-best | 2 / 0 | 2 | 0.5828 | 3.293 |  |
| joint | affinity-fixed-best | 2 / 0 | 2 | 0.8294 | 1.55 |  |
| fixed-ttl | affinity-fixed-best | 2 / 0 | 2 | 0.8724 | 1.357 |  |
| affinity-fixed-best | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fifo | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| joint | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fixed-ttl | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| affinity-fixed-best | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fifo | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| joint | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fixed-ttl | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |

## Confirmatory repeats

| Policy | Seeds / runs | Trial IDs | Completed / submitted | Tasks/min | Mean JCT (s) | P95 (s) | Short / long slowdown | Output match to isolated |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |

No runs have this complete trial scope.

Exact seed IDs and per-seed trial IDs are in this scope's policy CSV. Rows with different seed sets are descriptive; use the paired table for comparisons.

This scope requires exactly trials [0, 1] for both policies within each seed. Partial repeats are excluded rather than averaged with full repeats.

Paired results: `paired_confirmatory.csv`. Each row uses its explicitly listed seed/trial pairs; excluded samples are not replaced by an intersection.

| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| affinity-fixed-best | fifo | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| joint | fifo | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| fixed-ttl | fifo | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| affinity-fixed-best | fifo | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| joint | fifo | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| fixed-ttl | fifo | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| affinity-fixed-best | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fifo | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| joint | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fixed-ttl | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| affinity-fixed-best | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fifo | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| joint | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fixed-ttl | best-fixed | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fifo | affinity-fixed-best | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| joint | affinity-fixed-best | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| fixed-ttl | affinity-fixed-best | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| fifo | affinity-fixed-best | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| joint | affinity-fixed-best | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| fixed-ttl | affinity-fixed-best | 0 / 2 | 0 | nan | nan | confirmatory_trial_set_mismatch:2 |
| affinity-fixed-best | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fifo | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| joint | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fixed-ttl | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| affinity-fixed-best | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fifo | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| joint | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |
| fixed-ttl | affinity | 0 / 2 | 0 | nan | nan | missing_baseline:2 |

Baselines: fifo, best-fixed, affinity-fixed-best, affinity. Complete exclusion diagnostics remain in `paired.csv`.
Battery-powered runs are listed but never paired.
These are finite-stream replay measurements. They do not establish steady-state capacity.
