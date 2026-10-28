import json

import pytest

from phase1.noise import spread, trial_rows

CONFIG = dict(P=2, ctx_size=24576, kv_unified=False, cache_ram_mib=0, k=2, gap_mode='lognormal:2:1',
              arrival='poisson:1', repeat=0, steps_limit=0)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_trials_give_spread_and_paired_ratio(tmp_path):
    runs, jobs, requests = [], [], []
    # trial -> (FIFO tasks/min, J2 tasks/min)
    for trial, (fifo, j2) in {1: (0.8, 0.84), 2: (0.9, 0.9), 3: (0.7, 0.77)}.items():
        for j, rate in [(0, fifo), (2, j2)]:
            label = f'macro-J{j}-T{trial}'
            runs.append(dict(CONFIG, label=label, j=j, trial=trial, tasks=2, completed=2, failed=0,
                             makespan_s=120 / rate, jobs_per_minute=rate))
            jobs += [dict(label=label, status='ok', latency_s=s) for s in (10.0, 30.0)]
            requests += [dict(label=label, status='ok', step=1, new_tokens=10, reusable_tokens=100,
                              timings=dict(prompt_n=10, cache_n=cache, prompt_ms=1)) for cache in (100, 0)]
    write(tmp_path / 'raw/macro_runs.jsonl', runs)
    write(tmp_path / 'raw/macro_jobs.jsonl', jobs)
    write(tmp_path / 'raw/macro_requests.jsonl', requests)

    rows = trial_rows(tmp_path)
    assert len(rows) == 6
    j2 = rows[rows.j == 2].set_index('trial')
    assert j2.vs_fifo.to_dict() == pytest.approx({1: 1.05, 2: 1.0, 3: 1.1})
    assert rows.p50_s.unique().tolist() == [20.0]
    assert rows.full_history_hit_rate.unique().tolist() == [0.5]

    summary = spread(rows).set_index('j')
    assert summary.loc[0, 'trials'] == 3
    assert summary.loc[0, 'jobs_per_minute_mean'] == pytest.approx(0.8)
    assert summary.loc[0, 'jobs_per_minute_sd'] == pytest.approx(0.1)
    assert summary.loc[0, 'jobs_per_minute_cv'] == pytest.approx(0.125)
    assert summary.loc[2, 'vs_fifo_min'] == pytest.approx(1.0)
    assert summary.loc[2, 'vs_fifo_max'] == pytest.approx(1.1)


def test_runs_without_trial_column_count_as_trial_zero(tmp_path):
    write(tmp_path / 'raw/macro_runs.jsonl', [dict(CONFIG, label='a', j=0, tasks=1, completed=1, failed=0,
                                                   makespan_s=60, jobs_per_minute=1.0)])
    write(tmp_path / 'raw/macro_jobs.jsonl', [dict(label='a', status='ok', latency_s=5.0)])
    write(tmp_path / 'raw/macro_requests.jsonl', [dict(label='a', status='ok', step=0, new_tokens=5,
                                                       reusable_tokens=0, timings=dict(prompt_n=5, cache_n=0))])
    rows = trial_rows(tmp_path)
    assert rows.trial.tolist() == [0] and rows.vs_fifo.tolist() == [1.0]
