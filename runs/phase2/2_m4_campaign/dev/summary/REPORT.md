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
| affinity (4521cb83) | 3 / 3 | [0] | 30 / 30 | 0.701 | 275.7 | 512.6 | 6.34 / 4.85 | 0.933 |
| affinity-fixed-j2 (654f202a) | 3 / 3 | [0] | 30 / 30 | 1.087 | 129.3 | 208.5 | 6.49 / 1.79 | 0.9 |
| affinity-fixed-j3 (8376ef42) | 3 / 3 | [0] | 30 / 30 | 0.7319 | 278.1 | 454.7 | 12.1 / 4.31 | 0.933 |
| cache-a80 (cc2ea1d0) | 3 / 3 | [0] | 30 / 30 | 1 | 147.5 | 245 | 5.54 / 2.2 | 0.933 |
| fifo (42eadd78) | 3 / 3 | [0] | 30 / 30 | 0.7108 | 275 | 506.9 | 6.7 / 4.84 | 0.933 |
| fixed-k1-j1 (f89a9f09) | 3 / 3 | [0] | 30 / 30 | 0.8692 | 184.6 | 282 | 12.7 / 2.42 | 1 |
| fixed-k1-jinf (64f02c48) | 3 / 3 | [0] | 30 / 30 | 0.537 | 404.1 | 760.3 | 9.22 / 6.73 | 0.833 |
| fixed-k2-j2 (f4348eb5) | 3 / 3 | [0] | 30 / 30 | 1.069 | 136.1 | 212.5 | 7.04 / 1.9 | 0.967 |
| fixed-k2-j3 (8df38ae3) | 3 / 3 | [0] | 30 / 30 | 0.728 | 274 | 460.9 | 11.5 / 4.25 | 0.967 |
| fixed-k2-j4 (05f2467e) | 3 / 3 | [0] | 30 / 30 | 0.7169 | 280 | 478.1 | 11.5 / 4.61 | 0.9 |
| joint-a20 (105e453b) | 3 / 3 | [0] | 30 / 30 | 0.7685 | 238.1 | 427.4 | 6.6 / 4.02 | 0.9 |
| joint-a40 (d3268682) | 3 / 3 | [0] | 30 / 30 | 0.912 | 175.5 | 307.8 | 5.68 / 2.88 | 0.933 |
| joint-a80 (4ed3bc99) | 3 / 3 | [0] | 30 / 30 | 1.026 | 140.1 | 231.3 | 5.11 / 2.04 | 0.967 |
| joint-checkpoint-a80 (a60b4703) | 3 / 3 | [0] | 30 / 30 | 0.9967 | 152.2 | 267.5 | 4.63 / 2.43 | 0.933 |
| queue-a80 (7ce3b490) | 3 / 3 | [0] | 30 / 30 | 0.6524 | 326 | 582.6 | 8.9 / 5.7 | 0.967 |
| ttl-10-a80 (aa1f1b4b) | 3 / 3 | [0] | 30 / 30 | 1.013 | 139.6 | 243.4 | 5.62 / 2.04 | 0.9 |
| ttl-15-a80 (f26b687b) | 3 / 3 | [0] | 30 / 30 | 1.014 | 135.4 | 241.3 | 5.19 / 1.98 | 0.967 |
| ttl-30-a80 (d0a597f1) | 3 / 3 | [0] | 30 / 30 | 1.006 | 143.2 | 252.9 | 5.49 / 2.14 | 0.933 |
| ram-cache (32f7767b) | 3 / 3 | [0] | 30 / 30 | 0.6234 | 334.4 | 614.4 | nan / nan | nan |

Exact seed IDs and per-seed trial IDs are in this scope's policy CSV. Rows with different seed sets are descriptive; use the paired table for comparisons.

Only trial 0 is used for every strategy and baseline in this scope. Additional baseline repeats do not remove T0-only ablations.

Paired results: `paired_t0.csv`. Each row uses its explicitly listed seed/trial pairs; excluded samples are not replaced by an intersection.

| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| affinity | fifo | 3 / 0 | 3 | 0.9876 | 1.033 |  |
| affinity-fixed-j2 | fifo | 3 / 0 | 3 | 1.552 | 0.5575 |  |
| affinity-fixed-j3 | fifo | 3 / 0 | 3 | 1.033 | 1.037 |  |
| cache-a80 | fifo | 3 / 0 | 3 | 1.424 | 0.6082 |  |
| fixed-k1-j1 | fifo | 3 / 0 | 3 | 1.237 | 0.7694 |  |
| fixed-k1-jinf | fifo | 3 / 0 | 3 | 0.7591 | 1.662 |  |
| fixed-k2-j2 | fifo | 3 / 0 | 3 | 1.525 | 0.5824 |  |
| fixed-k2-j3 | fifo | 3 / 0 | 3 | 1.026 | 1.012 |  |
| fixed-k2-j4 | fifo | 3 / 0 | 3 | 1.009 | 0.9945 |  |
| joint-a20 | fifo | 3 / 0 | 3 | 1.084 | 0.8679 |  |
| joint-a40 | fifo | 3 / 0 | 3 | 1.294 | 0.685 |  |
| joint-a80 | fifo | 3 / 0 | 3 | 1.461 | 0.5767 |  |
| joint-checkpoint-a80 | fifo | 3 / 0 | 3 | 1.419 | 0.6205 |  |
| queue-a80 | fifo | 3 / 0 | 3 | 0.9165 | 1.186 |  |
| ttl-10-a80 | fifo | 3 / 0 | 3 | 1.442 | 0.5726 |  |
| ttl-15-a80 | fifo | 3 / 0 | 3 | 1.444 | 0.558 |  |
| ttl-30-a80 | fifo | 3 / 0 | 3 | 1.433 | 0.5943 |  |
| ram-cache | fifo | 0 / 3 | 0 | nan | nan | resource_mismatch:3 |

## Confirmatory repeats

| Policy | Seeds / runs | Trial IDs | Completed / submitted | Tasks/min | Mean JCT (s) | P95 (s) | Short / long slowdown | Output match to isolated |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |

No runs have this complete trial scope.

Exact seed IDs and per-seed trial IDs are in this scope's policy CSV. Rows with different seed sets are descriptive; use the paired table for comparisons.

This scope requires exactly trials [0, 1, 2] for both policies within each seed. Partial repeats are excluded rather than averaged with full repeats.

Paired results: `paired_confirmatory.csv`. Each row uses its explicitly listed seed/trial pairs; excluded samples are not replaced by an intersection.

| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| affinity | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-j2 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| affinity-fixed-j3 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| cache-a80 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fixed-k1-j1 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fixed-k1-jinf | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fixed-k2-j2 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fixed-k2-j3 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| fixed-k2-j4 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-a20 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-a40 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-a80 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| joint-checkpoint-a80 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| queue-a80 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| ttl-10-a80 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| ttl-15-a80 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| ttl-30-a80 | fifo | 0 / 3 | 0 | nan | nan | confirmatory_trial_set_mismatch:3 |
| ram-cache | fifo | 0 / 3 | 0 | nan | nan | resource_mismatch:3 |

Baselines: fifo. Complete exclusion diagnostics remain in `paired.csv`.
Battery-powered runs are listed but never paired.
These are finite-stream replay measurements. They do not establish steady-state capacity.
