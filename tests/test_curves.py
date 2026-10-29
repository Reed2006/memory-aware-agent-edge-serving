import pandas as pd
import pytest

from phase1.curves import active_at_dispatch, kv_demand, time_over


def frames():
    # Two tasks in one run: job 0 runs 0-10 s, job 1 is admitted at 4 s and ends at 12 s.
    jobs = pd.DataFrame([dict(label='a', job=0, arrival_s=0.0, admitted_s=0.0, done_s=10.0),
                         dict(label='a', job=1, arrival_s=2.0, admitted_s=4.0, done_s=12.0),
                         dict(label='b', job=0, arrival_s=0.0, admitted_s=0.0, done_s=99.0)])
    requests = pd.DataFrame([
        dict(label='a', job=0, step=0, enqueue_s=0.0, dispatch_s=0.0, done_s=2.0, input_tokens=100, expected_output=10, status='ok'),
        dict(label='a', job=0, step=1, enqueue_s=5.0, dispatch_s=5.0, done_s=7.0, input_tokens=130, expected_output=10, status='ok'),
        dict(label='a', job=1, step=0, enqueue_s=4.0, dispatch_s=6.0, done_s=8.0, input_tokens=200, expected_output=5, status='error'),
        dict(label='b', job=0, step=0, enqueue_s=1.0, dispatch_s=1.0, done_s=3.0, input_tokens=50, expected_output=5, status='ok'),
    ])
    return requests, jobs


def test_active_tasks_counted_within_the_same_run_only():
    requests, jobs = frames()
    assert active_at_dispatch(requests, jobs).tolist() == [1, 2, 2, 1]


def test_waiting_for_admission_is_not_active():
    requests, jobs = frames()
    early = requests.assign(dispatch_s=[0.0, 3.0, 6.0, 1.0])  # job 1 arrived at 2 s but is admitted at 4 s
    assert active_at_dispatch(early, jobs).tolist()[1] == 1


def test_kv_demand_tracks_history_held_by_active_tasks():
    requests, jobs = frames()
    run = requests[requests.label == 'a']
    demand = kv_demand(run, jobs[jobs.label == 'a'])
    at = lambda t: demand[demand.t_s <= t].iloc[-1]
    assert at(1).tokens == 100                    # job 0 prompt queued
    assert at(3).tokens == 110                    # prompt + generated, kept during the tool call
    assert at(4.5).tokens == 110 + 200 and at(4.5).active == 2
    assert at(7.5).tokens == 140 + 200            # job 0 step 1 done
    assert at(9).tokens == 140 + 200              # failed request adds no output
    assert at(11).tokens == 200 and at(11).active == 1
    assert demand.tokens.iloc[-1] == 0


def test_time_over_budget():
    demand = pd.DataFrame(dict(t_s=[0, 2, 6, 10], tokens=[100, 300, 150, 0], active=[1, 2, 1, 0]))
    assert time_over(demand, 200) == pytest.approx(0.4)
    assert time_over(demand, 1000) == 0


def test_contention_per_run(tmp_path):
    import json
    from phase1.curves import contention
    requests, jobs = frames()
    raw = tmp_path / 'raw'
    raw.mkdir()
    timings = lambda r: dict(prompt_n=r['input_tokens'], cache_n=0, prompt_ms=1)
    rows = [dict(r, reusable_tokens=0, new_tokens=r['input_tokens'], timings=timings(r)) for r in requests.to_dict('records')]
    (raw / 'macro_requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows if r['label'] == 'a'))
    (raw / 'macro_jobs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in jobs.to_dict('records') if r['label'] == 'a'))
    (raw / 'macro_runs.jsonl').write_text(json.dumps(dict(label='a', P=1, jobs_per_minute=1.0, makespan_s=12.0)) + '\n')
    row = contention(tmp_path).iloc[0]
    # Two tasks hold history from 4 s to 10 s of a 12 s run; one slot (P = 1).
    assert row.mean_active == pytest.approx((4 * 1 + 6 * 2 + 2 * 1) / 12)
    assert row.above_slots == pytest.approx(6 / 12)
