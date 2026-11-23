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
| affinity-fixed-best (654f202a) | 3 / 3 | [0] | 60 / 60 | 1.018 | 197.3 | 328.2 | 12.4 / 2.28 | 0.883 |
| joint (4ed3bc99) | 3 / 3 | [0] | 60 / 60 | 0.9368 | 246.5 | 478.5 | 11.4 / 3.09 | 0.717 |
| joint-checkpoint (a60b4703) | 3 / 3 | [0] | 60 / 60 | 1.015 | 208.7 | 420.6 | 7.77 / 2.81 | 0.817 |

Exact seed IDs and per-seed trial IDs are in this scope's policy CSV. Rows with different seed sets are descriptive; use the paired table for comparisons.

Only trial 0 is used for every strategy and baseline in this scope. Additional baseline repeats do not remove T0-only ablations.

Paired results: `paired_t0.csv`. Each row uses its explicitly listed seed/trial pairs; excluded samples are not replaced by an intersection.

| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| joint | affinity-fixed-best | 3 / 0 | 3 | 0.9279 | 1.24 |  |
| joint-checkpoint | affinity-fixed-best | 3 / 0 | 3 | 0.997 | 1.048 |  |
| affinity-fixed-best | joint | 3 / 0 | 3 | 1.081 | 0.8095 |  |
| joint-checkpoint | joint | 3 / 0 | 3 | 1.078 | 0.8454 |  |

## Confirmatory repeats

| Policy | Seeds / runs | Trial IDs | Completed / submitted | Tasks/min | Mean JCT (s) | P95 (s) | Short / long slowdown | Output match to isolated |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |

No runs have this complete trial scope.

Exact seed IDs and per-seed trial IDs are in this scope's policy CSV. Rows with different seed sets are descriptive; use the paired table for comparisons.

This scope requires exactly trials [0, 1, 2] for both policies within each seed. Partial repeats are excluded rather than averaged with full repeats.

Paired results: `paired_confirmatory.csv`. Each row uses its explicitly listed seed/trial pairs; excluded samples are not replaced by an intersection.

| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| joint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-best | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |

Baselines: affinity-fixed-best, joint. Complete exclusion diagnostics remain in `paired.csv`.
Battery-powered runs are listed but never paired.
These are finite-stream replay measurements. They do not establish steady-state capacity.
