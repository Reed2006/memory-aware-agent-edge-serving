"""Per-configuration macro summary: completion, tail latency, wasted prefill and fairness.

Slowdown = job latency (arrival -> done, admission wait included) divided by
(isolated compute time of the same source task + this job's own tool delays).
Isolated compute = median(service - tool delays) over J=1, K=1 runs of that task,
pooled over server configurations. Without J=1, K=1 runs
slowdown columns are empty.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = ['P', 'ctx_size', 'kv_unified', 'cache_ram_mib', 'k', 'j', 'gap_mode', 'arrival', 'steps_limit']


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(s) for s in path.read_text().splitlines() if s])


def request_metrics(requests: pd.DataFrame) -> pd.DataFrame:
    requests = requests.copy()
    if 'timings' not in requests:  # every request failed before the server answered
        requests['timings'] = None
    for key in ['prompt_n', 'cache_n', 'prompt_ms']:
        requests[key] = requests['timings'].map(lambda d: d.get(key) if isinstance(d, dict) else np.nan)
    ok = requests.status == 'ok'
    requests['wasted_prefill'] = np.where(ok, (requests.prompt_n - requests.new_tokens).clip(lower=0), np.nan)
    continuation = requests.step > 0
    requests['full_history_hit'] = np.where(ok & continuation, requests.cache_n >= requests.reusable_tokens, np.nan)
    return requests


def jain(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.sum() ** 2 / (len(values) * (values ** 2).sum())) if len(values) else np.nan


def isolated_compute(jobs: pd.DataFrame) -> pd.Series:
    ref = jobs[(jobs.k == 1) & (jobs.j == 1) & (jobs.status == 'ok')].copy()
    ref['compute_s'] = ref.service_s - ref.gap_total_s
    return ref.groupby(['steps_limit', 'task_id']).compute_s.median()


def summarize(root: Path) -> pd.DataFrame:
    jobs = read(root / 'raw/macro_jobs.jsonl')
    runs = read(root / 'raw/macro_runs.jsonl')
    requests = read(root / 'raw/macro_requests.jsonl')
    if jobs.empty or 'P' not in jobs:
        raise SystemExit(f'No sweep-format macro records in {root}/raw')
    for frame in (jobs, runs, requests):
        frame['j'] = frame['j'].fillna(0).astype(int)
        frame['steps_limit'] = frame['steps_limit'].fillna(0).astype(int)
    requests = request_metrics(requests)
    reference = isolated_compute(jobs)
    keys = list(zip(jobs.steps_limit, jobs.task_id))
    jobs['isolated_s'] = [reference.get(key, np.nan) for key in keys] + jobs.gap_total_s
    jobs['slowdown'] = np.where(jobs.status == 'ok', jobs.latency_s / jobs.isolated_s, np.nan)
    long_cut = jobs.groupby('task_id').expected_output_tokens.first().median()
    rows = []
    for config, group in jobs.groupby(CONFIG, dropna=False):
        done = group[group.status == 'ok']
        run_group = runs.merge(group[['label']].drop_duplicates(), on='label')
        req = requests.merge(group[['label']].drop_duplicates(), on='label')
        long = done.expected_output_tokens > long_cut
        rows.append(dict(
            zip(CONFIG, config), runs=group.label.nunique(), submitted=len(group), completed=len(done),
            failed=len(group) - len(done),
            failure_types=';'.join(sorted(group[group.status != 'ok'].status.unique())),
            jobs_per_min_mean=run_group.jobs_per_minute.mean(), jobs_per_min_sd=run_group.jobs_per_minute.std(),
            p50_s=done.latency_s.median(), p95_s=done.latency_s.quantile(.95), p99_s=done.latency_s.quantile(.99),
            admission_wait_mean_s=group.admission_wait_s.mean(),
            fifo_wait_mean_s=req.queue_s.mean(), fifo_wait_p95_s=req.queue_s.quantile(.95),
            prefill_tokens=req.prompt_n.sum(), wasted_prefill_tokens=req.wasted_prefill.sum(),
            wasted_prefill_fraction=req.wasted_prefill.sum() / req.prompt_n.sum() if req.prompt_n.sum() else np.nan,
            full_history_hit_rate=req.full_history_hit.mean(),
            slowdown_mean=done.slowdown.mean(), slowdown_p95=done.slowdown.quantile(.95),
            slowdown_max=done.slowdown.max(), jain_1_over_slowdown=jain(1 / done.slowdown),
            slowdown_long_mean=done[long].slowdown.mean(), slowdown_short_mean=done[~long].slowdown.mean(),
        ))
    summary = pd.DataFrame(rows)
    out = root / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / 'sweep_summary.csv', index=False)
    jobs.to_csv(out / 'sweep_jobs.csv', index=False)
    requests.drop(columns=['timings']).to_csv(out / 'sweep_requests.csv', index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True)
    summary = summarize(Path(parser.parse_args().results_dir))
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(summary.round(3).T)


if __name__ == '__main__':
    main()
