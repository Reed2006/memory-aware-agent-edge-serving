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
| affinity-fixed-best (654f202a) | 3 / 3 | [0] | 60 / 60 | 0.7619 | 190.9 | 329.6 | 4.21 / 2.06 | 0.817 |
| fifo (42eadd78) | 3 / 3 | [0] | 60 / 60 | 0.425 | 968 | 1794 | 14 / 11 | 0.633 |
| joint (4ed3bc99) | 3 / 3 | [0] | 60 / 60 | 0.7199 | 258.4 | 488.5 | 5.43 / 2.83 | 0.75 |
| joint-checkpoint (a60b4703) | 3 / 3 | [0] | 60 / 60 | 0.7583 | 221.9 | 390.6 | 3.58 / 2.47 | 0.85 |
| affinity-fixed-best (654f202a) | 3 / 3 | [0] | 60 / 60 | 0.9318 | 57.26 | 123.2 | 1.45 / 1.1 | 0.883 |
| fifo (42eadd78) | 3 / 3 | [0] | 60 / 60 | 0.8272 | 128.8 | 316.4 | 3.3 / 3.05 | 0.883 |
| joint (4ed3bc99) | 3 / 3 | [0] | 60 / 60 | 0.9322 | 58.89 | 133 | 1.49 / 1.13 | 0.917 |
| joint-checkpoint (a60b4703) | 3 / 3 | [0] | 60 / 60 | 0.9321 | 56.63 | 134.8 | 1.31 / 1.17 | 0.9 |
| affinity-fixed-best (654f202a) | 3 / 3 | [0] | 60 / 60 | 0.4599 | 80.23 | 175.5 | 1.44 / 1.18 | 0.933 |
| fifo (42eadd78) | 3 / 3 | [0] | 60 / 60 | 0.4244 | 170.4 | 351 | 3.06 / 2.5 | 0.917 |
| joint (4ed3bc99) | 3 / 3 | [0] | 60 / 60 | 0.4602 | 80.31 | 168.8 | 1.4 / 1.19 | 0.9 |
| joint-checkpoint (a60b4703) | 3 / 3 | [0] | 60 / 60 | 0.46 | 79.5 | 173.5 | 1.23 / 1.19 | 0.883 |
| affinity-fixed-best (654f202a) | 3 / 3 | [0] | 60 / 60 | 1.18 | 196.4 | 346.4 | 7.02 / 2.63 | 0.9 |
| fifo (42eadd78) | 3 / 3 | [0] | 60 / 60 | 0.5811 | 757.7 | 1472 | 13.8 / 12.1 | 0.7 |
| joint (4ed3bc99) | 3 / 3 | [0] | 60 / 60 | 1.065 | 270.6 | 512.1 | 6.98 / 3.92 | 0.783 |
| joint-checkpoint (a60b4703) | 3 / 3 | [0] | 60 / 60 | 1.168 | 216 | 431.1 | 4.48 / 3.23 | 0.833 |

Exact seed IDs and per-seed trial IDs are in this scope's policy CSV. Rows with different seed sets are descriptive; use the paired table for comparisons.

Only trial 0 is used for every strategy and baseline in this scope. Additional baseline repeats do not remove T0-only ablations.

Paired results: `paired_t0.csv`. Each row uses its explicitly listed seed/trial pairs; excluded samples are not replaced by an intersection.

| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fifo | affinity-fixed-best | 3 / 0 | 3 | 0.5581 | 5.56 |  |
| joint | affinity-fixed-best | 3 / 0 | 3 | 0.9455 | 1.27 |  |
| joint-checkpoint | affinity-fixed-best | 3 / 0 | 3 | 0.9954 | 1.133 |  |
| fifo | affinity-fixed-best | 3 / 0 | 3 | 0.9063 | 2.121 |  |
| joint | affinity-fixed-best | 3 / 0 | 3 | 1.001 | 1.022 |  |
| joint-checkpoint | affinity-fixed-best | 2 / 1 | 2 | 1.001 | 0.9821 | battery_power:1 |
| fifo | affinity-fixed-best | 3 / 0 | 3 | 0.938 | 1.916 |  |
| joint | affinity-fixed-best | 3 / 0 | 3 | 1.001 | 0.9975 |  |
| joint-checkpoint | affinity-fixed-best | 3 / 0 | 3 | 1 | 0.9923 |  |
| fifo | affinity-fixed-best | 3 / 0 | 3 | 0.4933 | 4.294 |  |
| joint | affinity-fixed-best | 3 / 0 | 3 | 0.8992 | 1.28 |  |
| joint-checkpoint | affinity-fixed-best | 3 / 0 | 3 | 0.9885 | 1.059 |  |
| affinity-fixed-best | joint | 3 / 0 | 3 | 1.065 | 0.8022 |  |
| fifo | joint | 3 / 0 | 3 | 0.5903 | 4.568 |  |
| joint-checkpoint | joint | 3 / 0 | 3 | 1.06 | 0.9028 |  |
| affinity-fixed-best | joint | 3 / 0 | 3 | 0.9994 | 0.9816 |  |
| fifo | joint | 3 / 0 | 3 | 0.9057 | 2.005 |  |
| joint-checkpoint | joint | 2 / 1 | 2 | 0.9997 | 1.001 | battery_power:1 |
| affinity-fixed-best | joint | 3 / 0 | 3 | 0.9994 | 1.003 |  |
| fifo | joint | 3 / 0 | 3 | 0.9374 | 1.896 |  |
| joint-checkpoint | joint | 3 / 0 | 3 | 0.9995 | 0.9957 |  |
| affinity-fixed-best | joint | 3 / 0 | 3 | 1.133 | 0.8083 |  |
| fifo | joint | 3 / 0 | 3 | 0.5596 | 3.614 |  |
| joint-checkpoint | joint | 3 / 0 | 3 | 1.113 | 0.8446 |  |
| affinity-fixed-best | fifo | 3 / 0 | 3 | 1.805 | 0.1877 |  |
| joint | fifo | 3 / 0 | 3 | 1.694 | 0.2459 |  |
| joint-checkpoint | fifo | 3 / 0 | 3 | 1.795 | 0.2153 |  |
| affinity-fixed-best | fifo | 3 / 0 | 3 | 1.116 | 0.6753 |  |
| joint | fifo | 3 / 0 | 3 | 1.116 | 0.6702 |  |
| joint-checkpoint | fifo | 2 / 1 | 2 | 1.033 | 0.8669 | battery_power:1 |
| affinity-fixed-best | fifo | 3 / 0 | 3 | 1.074 | 0.722 |  |
| joint | fifo | 3 / 0 | 3 | 1.075 | 0.7137 |  |
| joint-checkpoint | fifo | 3 / 0 | 3 | 1.074 | 0.7191 |  |
| affinity-fixed-best | fifo | 3 / 0 | 3 | 2.033 | 0.2523 |  |
| joint | fifo | 3 / 0 | 3 | 1.831 | 0.3412 |  |
| joint-checkpoint | fifo | 3 / 0 | 3 | 2.012 | 0.2754 |  |

## Confirmatory repeats

| Policy | Seeds / runs | Trial IDs | Completed / submitted | Tasks/min | Mean JCT (s) | P95 (s) | Short / long slowdown | Output match to isolated |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |

No runs have this complete trial scope.

Exact seed IDs and per-seed trial IDs are in this scope's policy CSV. Rows with different seed sets are descriptive; use the paired table for comparisons.

This scope requires exactly trials [0, 1, 2] for both policies within each seed. Partial repeats are excluded rather than averaged with full repeats.

Paired results: `paired_confirmatory.csv`. Each row uses its explicitly listed seed/trial pairs; excluded samples are not replaced by an intersection.

| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fifo | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fifo | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | battery_power:1;confirmatory_trial_set_mismatch:2 |
| fifo | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fifo | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-best | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fifo | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-best | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fifo | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | joint | 0 / 3 | 0 | nan | nan | battery_power:1;confirmatory_trial_set_mismatch:2 |
| affinity-fixed-best | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fifo | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-best | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fifo | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | joint | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-best | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-best | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | fifo | 0 / 3 | 0 | nan | nan | battery_power:1;confirmatory_trial_set_mismatch:2 |
| affinity-fixed-best | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-best | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |

Baselines: affinity-fixed-best, joint, fifo. Complete exclusion diagnostics remain in `paired.csv`.
Battery-powered runs are listed but never paired.
These are finite-stream replay measurements. They do not establish steady-state capacity.
