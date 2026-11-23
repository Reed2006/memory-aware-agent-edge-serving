"""Check the frontier supplement's predictions (docs/phase2/5_FRONTIER_PREREG.md).

Reads the summary that scripts/phase2/3_frontier_run.sh writes and prints, per seed,
efficiency and fairness for B2F, joint and joint-checkpoint, then P1-P5.

    .venv/bin/python scripts/phase2/3_frontier_check.py [summary-dir]
"""

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B2F, JOINT, JC = 'affinity-fixed-best', 'joint', 'joint-checkpoint'
LABELS = (B2F, JOINT, JC)


def p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(.95 * len(values)) - 1)] if values else math.nan


def jain(values):
    return sum(values) ** 2 / (len(values) * sum(v * v for v in values)) if values else math.nan


def job_metrics(jobs, delta=0.0):
    slow = [float(j['latency_s']) / (float(j['isolated_s']) + delta) for j in jobs]
    delay = [float(j['latency_s']) - float(j['isolated_s']) for j in jobs]
    short = [j['task_class'] == 'short' for j in jobs]
    return dict(p95_slowdown=p95(slow), jain=jain([1 / s for s in slow]),
                short_slowdown=statistics.mean([s for s, k in zip(slow, short) if k] or [math.nan]),
                delay_mean=statistics.mean(delay), delay_p95=p95(delay),
                short_delay_p95=p95([d for d, k in zip(delay, short) if k]))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    summary = Path(argv[0]) if argv else ROOT / 'runs/phase2/3_m4_frontier/eval/summary'
    runs = {(r['seed'], r['policy_label']): r for r in csv.DictReader(open(summary / 'run_summary.csv'))
            if r['trial'] == '0'}
    jobs = defaultdict(list)
    for j in csv.DictReader(open(summary / 'jobs.csv')):
        if j['trial'] == '0' and j['status'] == 'ok' and j['isolated_s'] not in ('', 'nan'):
            jobs[(j['seed'], j['policy_label'])].append(j)
    seeds = sorted({s for s, label in runs if label in LABELS})
    checks = defaultdict(list)
    for seed in seeds:
        if not all((seed, label) in runs for label in LABELS):
            print(f'seed {seed}: incomplete ({[l for l in LABELS if (seed, l) in runs]}); skipped')
            continue
        print(f'\nseed {seed}')
        print(f"{'policy':20s} {'tasks/min':>9s} {'JCT':>6s} {'P95 sd':>7s} {'short sd':>8s} {'Jain':>6s} "
              f"{'Jain@30':>7s} {'delay':>6s} {'P95 dl':>7s} {'shortP95':>8s} {'maxAdm':>6s} {'waste':>6s} {'evict':>5s} {'s/r':>7s}")
        m = {}
        for label in LABELS:
            r, js = runs[(seed, label)], jobs[(seed, label)]
            m[label] = dict(job_metrics(js), jct=float(r['jct_mean_s']), thr=float(r['jobs_per_min']),
                            waste=float(r['wasted_prefill_fraction']), jain30=job_metrics(js, 30.0)['jain'])
            x = m[label]
            print(f"{label:20s} {x['thr']:9.3f} {x['jct']:6.0f} {x['p95_slowdown']:7.2f} {x['short_slowdown']:8.2f} "
                  f"{x['jain']:6.3f} {x['jain30']:7.3f} {x['delay_mean']:6.0f} {x['delay_p95']:7.0f} "
                  f"{x['short_delay_p95']:8.0f} {float(r['max_admission_wait_s']):6.0f} {x['waste']:6.3f} "
                  f"{float(r['evictions']):5.0f} {r['saves']}/{r['restores']}")
        b, jo, jc = m[B2F], m[JOINT], m[JC]
        checks['P1 B2F JCT < jc JCT'].append(b['jct'] < jc['jct'])
        checks['P2 jc P95 sd < B2F and jc Jain > B2F'].append(jc['p95_slowdown'] < b['p95_slowdown'] and jc['jain'] > b['jain'])
        checks['P2 (Jain also at delta=30)'].append(jc['p95_slowdown'] < b['p95_slowdown'] and jc['jain30'] > b['jain30'])
        checks['P2-delay jc P95 delay < B2F'].append(jc['delay_p95'] < b['delay_p95'])
        checks['P3 jc waste < joint and jc JCT < joint'].append(jc['waste'] < jo['waste'] and jc['jct'] < jo['jct'])
        checks['P4 joint JCT > B2F'].append(jo['jct'] > b['jct'])
        checks['P5 B2F P95 delay < jc'].append(b['delay_p95'] < jc['delay_p95'])
    print('\nPredictions (holds if true in >= 2 of 3 seeds):')
    for name, results in checks.items():
        verdict = 'holds' if sum(results) >= 2 and len(results) >= 3 else (
            'not enough seeds' if len(results) < 3 else 'fails')
        print(f'  {name:42s} {sum(results)}/{len(results)}  {verdict}')
    paired = summary / 'paired_t0.csv'
    if paired.exists():
        print('\nPooled paired ratios (descriptive; seed-cluster bootstrap):')
        for r in csv.DictReader(open(paired)):
            if r['policy_label'] in LABELS and r['baseline'] in (B2F, JOINT) and r['eligible_seeds'] != '0':
                print(f"  {r['policy_label']:18s} vs {r['baseline']:20s} seeds={r['eligible_seeds']}  "
                      f"throughput {float(r['throughput_ratio']):.3f} [{float(r['throughput_ratio_ci_low']):.3f}, "
                      f"{float(r['throughput_ratio_ci_high']):.3f}]  JCT {float(r['jct_ratio']):.3f} "
                      f"[{float(r['jct_ratio_ci_low']):.3f}, {float(r['jct_ratio_ci_high']):.3f}]")


if __name__ == '__main__':
    main()
