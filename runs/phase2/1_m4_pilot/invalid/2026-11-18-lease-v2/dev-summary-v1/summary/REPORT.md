# Phase 2 replay summary

Completion means replay steps completed, not semantic task success.
Trials are averaged within seed, then seeds receive equal weight. Latency percentiles describe completed jobs only.
Paired ratios require equal planned and completed work, matching resources/inputs, and matching trial sets.
Bootstrap resamples seed clusters (2000 draws; RNG seed 291); intervals need at least two seeds.
Throughput ratio > 1 favors the policy; JCT ratio < 1 favors the policy.
Slowdown uses only matched isolated prompt/decode timings plus each job's actual tool gaps.
Mock-transport runs are marked synthetic_performance and excluded from paired performance claims.
Aborted/incomplete runs retain their planned denominator and any completed records; they cannot establish a paired gain.
Short-task cutoff (expected output tokens, frozen on dev): 782.5.

| Policy | Seeds | Completed / submitted | Tasks/min | Mean JCT (s) | P95 (s) | Short / long slowdown | Full-history hit | Peak RSS (GiB) | Leases / evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affinity (4521cb83) | 3 | 30 / 30 | 0.701 | 275.7 | 512.6 | 6.34 / 4.85 | 0.41 | 3.54 | 0 / 54.3 |
| affinity-fixed-j2 (654f202a) | 3 | 30 / 30 | 1.087 | 129.3 | 208.5 | 6.49 / 1.79 | 1.00 | 3.53 | 0 / 0 |
| affinity-fixed-j3 (8376ef42) | 3 | 30 / 30 | 0.7319 | 278.1 | 454.7 | 12.1 / 4.31 | 0.53 | 3.53 | 0 / 43 |
| attained (ff71c21e) | 3 | 30 / 30 | 0.6871 | 307.3 | 538.5 | 6.67 / 5.37 | 0.43 | 3.54 | 0 / 51.7 |
| cache (f1c58636) | 3 | 30 / 30 | 0.8938 | 179.5 | 316.9 | 4.23 / 2.95 | 0.69 | 3.52 | 14.7 / 28.7 |
| fifo (42eadd78) | 3 | 30 / 30 | 0.7108 | 275 | 506.9 | 6.7 / 4.84 | 0.41 | 3.54 | 0 / 0 |
| fixed-k1-j1 (f89a9f09) | 3 | 30 / 30 | 0.8692 | 184.6 | 282 | 12.7 / 2.42 | 1.00 | 3.52 | 0 / 0 |
| fixed-k1-jinf (64f02c48) | 3 | 30 / 30 | 0.537 | 404.1 | 760.3 | 9.22 / 6.73 | 0.13 | 3.55 | 0 / 0 |
| fixed-k2-j2 (f4348eb5) | 3 | 30 / 30 | 1.069 | 136.1 | 212.5 | 7.04 / 1.9 | 0.96 | 3.56 | 0 / 0 |
| fixed-k2-j3 (8df38ae3) | 3 | 30 / 30 | 0.728 | 274 | 460.9 | 11.5 / 4.25 | 0.54 | 3.54 | 0 / 0 |
| fixed-k2-j4 (05f2467e) | 3 | 30 / 30 | 0.7169 | 280 | 478.1 | 11.5 / 4.61 | 0.45 | 3.54 | 0 / 0 |
| joint-a20 (d347c511) | 3 | 30 / 30 | 0.7689 | 239.4 | 425 | 6.47 / 4.03 | 0.55 | 3.52 | 11 / 39.7 |
| joint-a40 (87d8171d) | 3 | 30 / 30 | 0.8252 | 209.8 | 365.6 | 6.17 / 3.34 | 0.62 | 3.5 | 20.3 / 33.3 |
| joint-a80 (e7a8d5c5) | 3 | 30 / 30 | 0.8495 | 206.4 | 352.6 | 6.16 / 3.27 | 0.67 | 3.49 | 24.7 / 29 |
| joint-checkpoint (e57ea888) | 3 | 30 / 30 | 1.017 | 139.5 | 249.7 | 3.79 / 2.22 | 0.93 | 3.57 | 0 / 42.3 |
| queue (80154f58) | 3 | 30 / 30 | 0.73 | 249.3 | 465.6 | 5.7 / 4.31 | 0.47 | 3.54 | 0 / 49 |
| ttl-1 (c882247c) | 3 | 30 / 30 | 0.6924 | 296.3 | 528.5 | 7.07 / 5.06 | 0.51 | 3.55 | 33 / 43.3 |
| ttl-10 (cb4c3428) | 3 | 30 / 30 | 0.9422 | 173.4 | 299.2 | 4.99 / 2.69 | 0.84 | 3.52 | 30.3 / 14.7 |
| ttl-3 (ae8669b8) | 3 | 30 / 30 | 0.7635 | 250.7 | 441.1 | 5.97 / 4.22 | 0.68 | 3.53 | 34.3 / 28.7 |
| ram-cache (32f7767b) | 3 | 30 / 30 | 0.6234 | 334.4 | 614.4 | nan / nan | 0.43 | 6.2 | 0 / 0 |

Paired comparisons against fifo are in `paired.csv`; excluded seeds and reasons remain visible.
Battery-powered runs are listed but never paired.
These are finite-stream replay measurements. They do not establish steady-state capacity.
