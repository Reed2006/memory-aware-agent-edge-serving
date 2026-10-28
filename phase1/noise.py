"""Run-to-run noise: the same repeat index (identical tasks, arrivals and tool delays)
replayed several times with --trial. The spread across trials is measurement noise,
not workload variation, so it sets how large a paired difference within one repeat
must be before it means anything.

Writes summary/noise_trials.csv (one row per run) and summary/noise.csv (per
configuration: mean, SD, coefficient of variation, min and max over trials), plus the
per-trial throughput of each configuration relative to FIFO (J = ∞) with the same P and K.
"""

import argparse
from pathlib import Path

import pandas as pd

from phase1.summarize import read, request_metrics

METRICS = ['jobs_per_minute', 'makespan_s', 'p50_s', 'p95_s', 'full_history_hit_rate', 'wasted_prefill_fraction']
KEY = ['P', 'ctx_size', 'kv_unified', 'cache_ram_mib', 'k', 'j', 'gap_mode', 'arrival', 'repeat']


def trial_rows(root: Path) -> pd.DataFrame:
    """One row per logged run with its trial, throughput, latency and cache metrics."""
    runs = read(root / 'raw/macro_runs.jsonl')
    if runs.empty:
        return runs
    jobs = read(root / 'raw/macro_jobs.jsonl')
    requests = request_metrics(read(root / 'raw/macro_requests.jsonl'))
    runs['j'] = runs['j'].fillna(0).astype(int)
    runs['trial'] = runs['trial'].fillna(0).astype(int) if 'trial' in runs else 0
    done = jobs[jobs.status == 'ok'].groupby('label').latency_s
    req = requests.groupby('label')
    per_label = pd.DataFrame({
        'p50_s': done.median(), 'p95_s': done.quantile(.95),
        'full_history_hit_rate': req.full_history_hit.mean(),
        'wasted_prefill_fraction': req.wasted_prefill.sum() / req.prompt_n.sum(),
    })
    rows = runs.merge(per_label, left_on='label', right_index=True, how='left')
    pair = [c for c in KEY if c != 'j'] + ['trial']  # same everything except the admission limit
    baseline = rows[rows.j == 0].set_index(pair).jobs_per_minute
    index = pd.MultiIndex.from_frame(rows[pair])
    rows['vs_fifo'] = rows.jobs_per_minute.values / baseline.reindex(index).values
    return rows.sort_values(['j', 'trial'])


def spread(rows: pd.DataFrame) -> pd.DataFrame:
    """Per configuration: number of trials and mean, SD, CV, min, max of each metric."""
    out = []
    for config, group in rows.groupby(KEY, dropna=False):
        row = dict(zip(KEY, config), trials=len(group), failed=int(group.failed.sum()))
        for metric in METRICS + ['vs_fifo']:
            values = group[metric].dropna()
            mean, sd = values.mean(), values.std()
            row.update({f'{metric}_mean': mean, f'{metric}_sd': sd,
                        f'{metric}_cv': sd / mean if mean else float('nan'),
                        f'{metric}_min': values.min(), f'{metric}_max': values.max()})
        out.append(row)
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', required=True)
    root = Path(parser.parse_args().results_dir)
    rows = trial_rows(root)
    if rows.empty:
        raise SystemExit(f'No macro runs in {root}/raw')
    summary = spread(rows)
    (root / 'summary').mkdir(parents=True, exist_ok=True)
    rows.to_csv(root / 'summary/noise_trials.csv', index=False)
    summary.to_csv(root / 'summary/noise.csv', index=False)
    columns = ['label', 'trial', 'completed', 'tasks', 'jobs_per_minute', 'makespan_s', 'p50_s', 'p95_s',
               'full_history_hit_rate', 'vs_fifo']
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(rows[columns].round(3).to_string(index=False))
        print()
        print(summary[['j', 'trials', 'failed', 'jobs_per_minute_mean', 'jobs_per_minute_sd', 'jobs_per_minute_cv',
                       'p50_s_mean', 'p50_s_cv', 'vs_fifo_mean', 'vs_fifo_min', 'vs_fifo_max']].round(3).to_string(index=False))
    print(f'Wrote {root}/summary/noise.csv and noise_trials.csv')


if __name__ == '__main__':
    main()
