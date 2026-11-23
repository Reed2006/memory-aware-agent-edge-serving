"""Check the rate/mix supplement's predictions (docs/phase2/6_RATE_MIX_PREREG.md).

Joins the supplement summary with the 1x eval cells (trial 0, same seeds and
policy hashes) and prints, per condition and seed, efficiency and fairness for
fifo, B2F, joint and joint-checkpoint, then R1-R5. Missing cells are reported,
so it can run on a partial summary.

    .venv/bin/python scripts/phase2/4_rate_mix_check.py [rate-mix-summary-dir] [eval-summary-dir]
"""

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIFO, B2F, JOINT, JC = 'fifo', 'affinity-fixed-best', 'joint', 'joint-checkpoint'
LABELS = (FIFO, B2F, JOINT, JC)
SEEDS = ('1101', '1102', '1103')
MAIN_WORKLOAD = 'f1eabdce'
MIX = {'e3342197': 'mix-short', '61eb926e': 'mix-long'}
ARRIVAL = {'scaled:0.5': '0.5x', 'scaled:1': '1x', 'scaled:2': '2x'}
CONDITIONS = ('0.5x', '1x', '2x', 'mix-short', 'mix-long')


def condition(row):
    workload = row['workload_hash'][:8]
    if workload in MIX:
        return MIX[workload] if row['arrival'] == 'scaled:1' else None
    if workload == MAIN_WORKLOAD and row['gap_mode'] == 'lognormal:2:1':
        return ARRIVAL.get(row['arrival'])
    return None


def p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(.95 * len(values)) - 1)] if values else math.nan


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else math.nan


def load(summary):
    runs, jobs = {}, defaultdict(list)
    if not (summary / 'run_summary.csv').exists():
        return runs, jobs
    for r in csv.DictReader(open(summary / 'run_summary.csv')):
        c = condition(r)
        if c and r['trial'] == '0' and r['seed'] in SEEDS and r['policy_label'] in LABELS:
            runs[(c, r['seed'], r['policy_label'])] = r
    for j in csv.DictReader(open(summary / 'jobs.csv')):
        c = condition(j)
        if c and j['trial'] == '0' and j['status'] == 'ok' and j['isolated_s'] not in ('', 'nan'):
            jobs[(c, j['seed'], j['policy_label'])].append(j)
    return runs, jobs


def metrics(run, jobs):
    slow = [float(j['latency_s']) / float(j['isolated_s']) for j in jobs]
    delay = [float(j['latency_s']) - float(j['isolated_s']) for j in jobs]
    short = [j['task_class'] == 'short' for j in jobs]
    return dict(
        thr=float(run['jobs_per_min']), jct=float(run['jct_mean_s']),
        p95_sd=p95(slow), short_sd=mean(s for s, k in zip(slow, short) if k),
        long_sd=mean(s for s, k in zip(slow, short) if not k),
        delay_p95=p95(delay), long_delay_p95=p95([d for d, k in zip(delay, short) if not k]),
        max_adm=float(run['max_admission_wait_s']), waste=float(run['wasted_prefill_fraction']),
        evict=float(run['evictions']), saves=run['saves'])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    supplement = Path(argv[0]) if argv else ROOT / 'runs/phase2/4_m4_rate_mix/sensitivity/summary'
    evaluation = Path(argv[1]) if len(argv) > 1 else ROOT / 'runs/phase2/2_m4_campaign/eval/summary'
    runs, jobs = load(evaluation)
    runs = {k: v for k, v in runs.items() if k[0] == '1x'}
    extra_runs, extra_jobs = load(supplement)
    runs.update(extra_runs)
    jobs.update(extra_jobs)
    m = {key: metrics(run, jobs[key]) for key, run in runs.items()}

    for c in CONDITIONS:
        print(f'\n== {c} ==')
        print(f"{'seed':5s} {'policy':20s} {'tasks/min':>9s} {'JCT':>6s} {'P95 sd':>7s} {'short sd':>8s} "
              f"{'long sd':>7s} {'P95 dl':>7s} {'longP95':>7s} {'maxAdm':>6s} {'waste':>6s} {'evict':>5s} {'saves':>5s}")
        for seed in SEEDS:
            for label in LABELS:
                x = m.get((c, seed, label))
                if x is None:
                    print(f'{seed:5s} {label:20s}   (not run yet)')
                    continue
                print(f"{seed:5s} {label:20s} {x['thr']:9.3f} {x['jct']:6.0f} {x['p95_sd']:7.2f} {x['short_sd']:8.2f} "
                      f"{x['long_sd']:7.2f} {x['delay_p95']:7.0f} {x['long_delay_p95']:7.0f} {x['max_adm']:6.0f} "
                      f"{x['waste']:6.3f} {x['evict']:5.0f} {x['saves']:>5s}")

    def have(c, seed, *labels):
        return all((c, seed, label) in m for label in labels)

    def ratio(c, seed):
        return m[(c, seed, FIFO)]['jct'] / m[(c, seed, B2F)]['jct']

    checks = defaultdict(list)
    for seed in SEEDS:
        if have('0.5x', seed, B2F, JOINT, JC) and have('0.5x', seed, FIFO) and have('1x', seed, FIFO, B2F):
            jcts = [m[('0.5x', seed, label)]['jct'] for label in (B2F, JOINT, JC)]
            checks['R1 0.5x: efficient within 10% and R(0.5x) < R(1x)'].append(
                max(jcts) / min(jcts) <= 1.10 and ratio('0.5x', seed) < ratio('1x', seed))
        if have('2x', seed, FIFO, B2F) and have('1x', seed, FIFO, B2F):
            checks['R2 R(2x) > R(1x)'].append(ratio('2x', seed) > ratio('1x', seed))
        if have('2x', seed, B2F, JC):
            b, jc = m[('2x', seed, B2F)], m[('2x', seed, JC)]
            checks['R3 2x: jc P95 sd < B2F and jc JCT >= B2F'].append(
                jc['p95_sd'] < b['p95_sd'] and jc['jct'] >= b['jct'])
        if have('2x', seed, B2F, JOINT):
            checks['R4 2x: joint JCT > B2F'].append(m[('2x', seed, JOINT)]['jct'] > m[('2x', seed, B2F)]['jct'])
        if have('mix-long', seed, FIFO, B2F) and have('mix-short', seed, FIFO, B2F):
            checks['R5 mix-long: B2F max adm and R larger than mix-short'].append(
                m[('mix-long', seed, B2F)]['max_adm'] > m[('mix-short', seed, B2F)]['max_adm']
                and ratio('mix-long', seed) > ratio('mix-short', seed))

    print('\nPredictions (hold if true in >= 2 of 3 seeds):')
    names = ['R1 0.5x: efficient within 10% and R(0.5x) < R(1x)', 'R2 R(2x) > R(1x)',
             'R3 2x: jc P95 sd < B2F and jc JCT >= B2F', 'R4 2x: joint JCT > B2F',
             'R5 mix-long: B2F max adm and R larger than mix-short']
    for name in names:
        results = checks.get(name, [])
        if len(results) < 3:
            verdict = f'{len(results)}/3 seeds available'
        else:
            verdict = 'holds' if sum(results) >= 2 else 'fails'
        print(f'  {name:55s} {sum(results)}/{len(results)}  {verdict}')

    print('\nDescriptive: B2F vs jc, long-task slowdown and P95 contention delay')
    for c in CONDITIONS:
        for seed in SEEDS:
            if have(c, seed, B2F, JC):
                b, jc = m[(c, seed, B2F)], m[(c, seed, JC)]
                print(f"  {c:9s} {seed}  long sd {b['long_sd']:.2f} vs {jc['long_sd']:.2f}   "
                      f"P95 delay {b['delay_p95']:.0f} vs {jc['delay_p95']:.0f}   "
                      f"long P95 delay {b['long_delay_p95']:.0f} vs {jc['long_delay_p95']:.0f}")


if __name__ == '__main__':
    main()
