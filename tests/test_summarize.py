import json

import numpy as np
import pytest

from phase1.summarize import summarize

CONFIG = dict(P=2, ctx_size=24576, kv_unified=False, cache_ram_mib=0, k=2, j=0, gap_mode='lognormal:2:1',
              arrival='scaled:3', steps_limit=0)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_trials_are_averaged_within_a_repeat(tmp_path):
    runs, jobs, requests = [], [], []
    # repeat -> tasks/min of trials 1, 2, 3 (trial 1 has no trial field, like existing runs)
    rates = {0: [0.6, 0.8, 0.7], 1: [0.4, 0.4, 0.4]}
    for repeat, values in rates.items():
        for trial, rate in enumerate(values, start=1):
            label = f'r{repeat}-t{trial}'
            run = dict(CONFIG, label=label, repeat=repeat, jobs_per_minute=rate)
            if trial > 1:
                run['trial'] = trial
            runs.append(run)
            jobs.append(dict(CONFIG, label=label, repeat=repeat, task_id='a', status='ok', latency_s=10.0,
                             service_s=9.0, gap_total_s=1.0, expected_output_tokens=5, admission_wait_s=0.0))
            requests.append(dict(CONFIG, label=label, status='ok', step=1, new_tokens=5, reusable_tokens=10,
                                 queue_s=0.0, timings=dict(prompt_n=5, cache_n=10, prompt_ms=1)))
    write(tmp_path / 'raw/macro_runs.jsonl', runs)
    write(tmp_path / 'raw/macro_jobs.jsonl', jobs)
    write(tmp_path / 'raw/macro_requests.jsonl', requests)

    row = summarize(tmp_path).iloc[0]
    assert (row.runs, row.repeats, row.trials) == (6, 2, 3)
    # Mean and SD across the two repeat means (0.7 and 0.4), not across six runs.
    assert row.jobs_per_min_mean == pytest.approx(0.55)
    assert row.jobs_per_min_sd == pytest.approx(np.std([0.7, 0.4], ddof=1))
    # Pooled within-repeat SD: variances 0.01 and 0.
    assert row.jobs_per_min_trial_sd == pytest.approx(np.sqrt(0.005))


def test_report_averages_trials_into_one_row_per_repeat():
    import pandas as pd
    from phase1.final import RUN_KEYS, average_trials
    base = dict(config='P2·24K K2 J∞', gap_mode='lognormal:2:1', arrival='scaled:3', P=2, k=2, j=0, ctx_size=24576,
                kv_unified=False, cache_ram_mib=0, repeat=0, submitted=8, completed=8, makespan_s=600.0, p50=100.0,
                p95=200.0, slowdown=2.0, wasted_frac=0.4, hit_rate=0.6)
    trials = pd.DataFrame([dict(base, label='a', trial=0, jobs_per_min=0.6), dict(base, label='a-T2', trial=2, jobs_per_min=0.8),
                           dict(base, label='b', repeat=1, trial=0, jobs_per_min=0.5)])
    out = average_trials(trials).set_index('repeat')
    assert out.loc[0, 'jobs_per_min'] == pytest.approx(0.7) and out.loc[0, 'jobs_per_min_tsd'] == pytest.approx(np.std([0.6, 0.8], ddof=1))
    assert (out.loc[0, 'trials'], out.loc[0, 'submitted'], out.loc[0, 'label']) == (2, 16, 'a')
    assert out.loc[1, 'trials'] == 1 and np.isnan(out.loc[1, 'jobs_per_min_tsd'])
    assert set(RUN_KEYS) <= set(out.reset_index().columns)
