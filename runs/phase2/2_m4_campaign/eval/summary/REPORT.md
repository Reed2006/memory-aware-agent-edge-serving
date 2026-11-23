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
| affinity (4521cb83) | 3 / 3 | [0] | 60 / 60 | 0.6244 | 335.6 | 745.8 | 5.3 / 5.11 | 0.767 |
| affinity-fixed-best (654f202a) | 3 / 3 | [0] | 60 / 60 | 0.8091 | 111.1 | 226 | 2.4 / 1.59 | 0.883 |
| cache (cc2ea1d0) | 3 / 3 | [0] | 60 / 60 | 0.7925 | 126.1 | 256.3 | 2.66 / 1.79 | 0.833 |
| fifo (42eadd78) | 3 / 3 | [0] | 60 / 60 | 0.6144 | 377.7 | 792.4 | 6.3 / 5.78 | 0.683 |
| best-fixed (f4348eb5) | 3 / 3 | [0] | 60 / 60 | 0.8044 | 119.6 | 235.4 | 2.75 / 1.72 | 0.833 |
| isolated (f89a9f09) | 1 / 1 | [0] | 20 / 20 | 0.7705 | 297.1 | 538.2 | 9.85 / 3.72 | 1 |
| joint (4ed3bc99) | 3 / 3 | [0] | 60 / 60 | 0.7876 | 132.4 | 277.5 | 2.74 / 1.89 | 0.833 |
| joint-checkpoint (a60b4703) | 3 / 3 | [0] | 60 / 60 | 0.8005 | 118.2 | 274.4 | 2.07 / 1.75 | 0.85 |
| queue (7ce3b490) | 3 / 3 | [0] | 60 / 60 | 0.6177 | 355.2 | 747.7 | 6.05 / 5.39 | 0.783 |
| fixed-ttl (f26b687b) | 3 / 3 | [0] | 60 / 60 | 0.784 | 132.9 | 290.6 | 2.87 / 1.79 | 0.933 |
| ram-cache (32f7767b) | 3 / 3 | [0] | 60 / 60 | 0.6284 | 331 | 725.7 | nan / nan | nan |

Exact seed IDs and per-seed trial IDs are in this scope's policy CSV. Rows with different seed sets are descriptive; use the paired table for comparisons.

Only trial 0 is used for every strategy and baseline in this scope. Additional baseline repeats do not remove T0-only ablations.

Paired results: `paired_t0.csv`. Each row uses its explicitly listed seed/trial pairs; excluded samples are not replaced by an intersection.

| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| affinity | fifo | 3 / 0 | 3 | 1.02 | 0.9142 |  |
| affinity-fixed-best | fifo | 3 / 0 | 3 | 1.375 | 0.4243 |  |
| cache | fifo | 3 / 0 | 3 | 1.341 | 0.4438 |  |
| best-fixed | fifo | 3 / 0 | 3 | 1.366 | 0.4444 |  |
| joint | fifo | 3 / 0 | 3 | 1.33 | 0.4549 |  |
| joint-checkpoint | fifo | 3 / 0 | 3 | 1.356 | 0.4334 |  |
| queue | fifo | 3 / 0 | 3 | 1.01 | 0.9759 |  |
| fixed-ttl | fifo | 3 / 0 | 3 | 1.325 | 0.4639 |  |
| ram-cache | fifo | 0 / 3 | 0 | nan | nan | resource_mismatch:3 |
| affinity | best-fixed | 3 / 0 | 3 | 0.7965 | 2.435 |  |
| affinity-fixed-best | best-fixed | 3 / 0 | 3 | 1.005 | 0.9456 |  |
| cache | best-fixed | 3 / 0 | 3 | 0.9877 | 1.025 |  |
| fifo | best-fixed | 3 / 0 | 3 | 0.7857 | 2.712 |  |
| joint | best-fixed | 3 / 0 | 3 | 0.9827 | 1.066 |  |
| joint-checkpoint | best-fixed | 3 / 0 | 3 | 0.9967 | 0.9824 |  |
| queue | best-fixed | 3 / 0 | 3 | 0.7878 | 2.602 |  |
| fixed-ttl | best-fixed | 3 / 0 | 3 | 0.9779 | 1.081 |  |
| ram-cache | best-fixed | 0 / 3 | 0 | nan | nan | resource_mismatch:3 |
| affinity | affinity-fixed-best | 3 / 0 | 3 | 0.7937 | 2.625 |  |
| cache | affinity-fixed-best | 3 / 0 | 3 | 0.9828 | 1.091 |  |
| fifo | affinity-fixed-best | 3 / 0 | 3 | 0.7831 | 2.926 |  |
| best-fixed | affinity-fixed-best | 3 / 0 | 3 | 0.995 | 1.061 |  |
| joint | affinity-fixed-best | 3 / 0 | 3 | 0.978 | 1.135 |  |
| joint-checkpoint | affinity-fixed-best | 3 / 0 | 3 | 0.9918 | 1.043 |  |
| queue | affinity-fixed-best | 3 / 0 | 3 | 0.785 | 2.797 |  |
| fixed-ttl | affinity-fixed-best | 3 / 0 | 3 | 0.9732 | 1.149 |  |
| ram-cache | affinity-fixed-best | 0 / 3 | 0 | nan | nan | resource_mismatch:3 |
| affinity-fixed-best | affinity | 3 / 0 | 3 | 1.338 | 0.4569 |  |
| cache | affinity | 3 / 0 | 3 | 1.305 | 0.4787 |  |
| fifo | affinity | 3 / 0 | 3 | 0.9807 | 1.096 |  |
| best-fixed | affinity | 3 / 0 | 3 | 1.329 | 0.4784 |  |
| joint | affinity | 3 / 0 | 3 | 1.296 | 0.4915 |  |
| joint-checkpoint | affinity | 3 / 0 | 3 | 1.32 | 0.4669 |  |
| queue | affinity | 3 / 0 | 3 | 0.9899 | 1.069 |  |
| fixed-ttl | affinity | 3 / 0 | 3 | 1.29 | 0.5013 |  |
| ram-cache | affinity | 0 / 3 | 0 | nan | nan | resource_mismatch:3 |

## Confirmatory repeats

| Policy | Seeds / runs | Trial IDs | Completed / submitted | Tasks/min | Mean JCT (s) | P95 (s) | Short / long slowdown | Output match to isolated |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| affinity-fixed-best (654f202a) | 3 / 6 | [0, 1] | 120 / 120 | 0.8099 | 111.7 | 227.9 | 2.41 / 1.6 | 0.875 |
| fifo (42eadd78) | 3 / 6 | [0, 1] | 120 / 120 | 0.6168 | 370.5 | 782.1 | 6.47 / 5.73 | 0.717 |
| best-fixed (f4348eb5) | 3 / 6 | [0, 1] | 120 / 120 | 0.8042 | 119.5 | 236.2 | 2.74 / 1.71 | 0.842 |
| joint (4ed3bc99) | 3 / 6 | [0, 1] | 120 / 120 | 0.791 | 129.3 | 268.1 | 2.67 / 1.85 | 0.85 |

Exact seed IDs and per-seed trial IDs are in this scope's policy CSV. Rows with different seed sets are descriptive; use the paired table for comparisons.

This scope requires exactly trials [0, 1] for both policies within each seed. Partial repeats are excluded rather than averaged with full repeats.

Paired results: `paired_confirmatory.csv`. Each row uses its explicitly listed seed/trial pairs; excluded samples are not replaced by an intersection.

| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| affinity | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-best | fifo | 3 / 0 | 6 | 1.364 | 0.4275 |  |
| cache | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| best-fixed | fifo | 3 / 0 | 6 | 1.353 | 0.4405 |  |
| joint | fifo | 3 / 0 | 6 | 1.325 | 0.4472 |  |
| joint-checkpoint | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| queue | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fixed-ttl | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| ram-cache | fifo | 0 / 3 | 0 | nan | nan | resource_mismatch:3 |
| affinity | best-fixed | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-best | best-fixed | 3 / 0 | 6 | 1.006 | 0.9535 |  |
| cache | best-fixed | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fifo | best-fixed | 3 / 0 | 6 | 0.7887 | 2.71 |  |
| joint | best-fixed | 3 / 0 | 6 | 0.9866 | 1.049 |  |
| joint-checkpoint | best-fixed | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| queue | best-fixed | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fixed-ttl | best-fixed | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| ram-cache | best-fixed | 0 / 3 | 0 | nan | nan | resource_mismatch:3 |
| affinity | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| cache | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fifo | affinity-fixed-best | 3 / 0 | 6 | 0.7855 | 2.912 |  |
| best-fixed | affinity-fixed-best | 3 / 0 | 6 | 0.9941 | 1.052 |  |
| joint | affinity-fixed-best | 3 / 0 | 6 | 0.981 | 1.109 |  |
| joint-checkpoint | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| queue | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fixed-ttl | affinity-fixed-best | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| ram-cache | affinity-fixed-best | 0 / 3 | 0 | nan | nan | resource_mismatch:3 |
| affinity-fixed-best | affinity | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| cache | affinity | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fifo | affinity | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| best-fixed | affinity | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint | affinity | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint | affinity | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| queue | affinity | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fixed-ttl | affinity | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| ram-cache | affinity | 0 / 3 | 0 | nan | nan | resource_mismatch:3 |

Baselines: fifo, best-fixed, affinity-fixed-best, affinity. Complete exclusion diagnostics remain in `paired.csv`.
Battery-powered runs are listed but never paired.
These are finite-stream replay measurements. They do not establish steady-state capacity.
