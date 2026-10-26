import asyncio
import json
import statistics

import pandas as pd
import pytest

from phase1.bench import arrival_times, gap_delay, job_sources
from phase1.summarize import jain, request_metrics


def test_gap_specs_are_deterministic_and_parameterised():
    assert gap_delay('recorded', 1.7, 'a') == 1.7
    assert gap_delay('zero', 1.7, 'a') == 0
    assert gap_delay('fixed:2.5', 9, 'a') == 2.5
    assert gap_delay('exp:3', 0, '0-1-2') == gap_delay('exp:3', 0, '0-1-2')
    assert gap_delay('exp:3', 0, '0-1-2') != gap_delay('exp:3', 0, '0-1-3')
    samples = [gap_delay('lognormal:2:0.5', 0, str(i)) for i in range(2000)]
    assert 1.8 < statistics.median(samples) < 2.2
    with pytest.raises(ValueError):
        gap_delay('uniform:1', 0, 'a')


def test_arrivals_batch_and_poisson():
    batch = arrival_times('batch', 5, 1)
    assert len(batch) == 5 and all(0 <= t <= 0.3 for t in batch)
    poisson = arrival_times('poisson:6', 4000, 1)  # 6 jobs/min -> mean gap 10 s
    assert poisson[0] == 0 and poisson == sorted(poisson)
    assert 9 < poisson[-1] / 3999 < 11
    assert arrival_times('poisson:6', 10, 1) == arrival_times('poisson:6', 10, 1)


def test_job_sources_are_balanced():
    tasks = ['a', 'b', 'c']
    jobs = job_sources(tasks, 7, 3)
    assert len(jobs) == 7 and all(2 <= jobs.count(t) <= 3 for t in tasks)
    assert sorted(jobs[:3]) == tasks


def test_wasted_prefill_and_history_hit():
    requests = pd.DataFrame([
        dict(status='ok', step=0, new_tokens=100, reusable_tokens=0, timings=dict(prompt_n=100, cache_n=0, prompt_ms=1)),
        dict(status='ok', step=1, new_tokens=20, reusable_tokens=150, timings=dict(prompt_n=20, cache_n=150, prompt_ms=1)),
        dict(status='ok', step=2, new_tokens=20, reusable_tokens=200, timings=dict(prompt_n=180, cache_n=40, prompt_ms=1)),
    ])
    out = request_metrics(requests)
    assert out.wasted_prefill.tolist() == [0, 0, 160]
    assert out.full_history_hit.tolist()[1:] == [1, 0]


def test_jain_index():
    assert jain([1, 1, 1]) == pytest.approx(1)
    assert jain([1, 0, 0]) == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_macro_end_to_end_with_admission_limit_and_overflow(tmp_path, monkeypatch):
    import argparse
    import httpx
    from phase1 import bench
    from phase1.core import ModelClient

    monkeypatch.chdir(tmp_path)  # no .server.json here: sampler runs without a PID
    limit = 40  # per-slot context; task "big" overflows on its second step

    async def handler(request):
        if request.url.path != '/completion':
            return httpx.Response(200, json={})  # slot erase
        body = json.loads(request.content)
        if len(body['prompt']) + body['n_predict'] > limit:
            return httpx.Response(400, json={'error': 'exceeds the available context size'})
        await asyncio.sleep(0.01)
        return httpx.Response(200, json={'tokens': [5] * body['n_predict'],
            'timings': {'prompt_n': len(body['prompt']), 'cache_n': 0, 'predicted_n': body['n_predict']}})

    small = dict(task_id='s', steps=[dict(delta=[1] * 5, output=3, gap_s=0),
                                     dict(delta=[2] * 2, output=3, gap_s=0)])
    big = dict(task_id='b', steps=[dict(delta=[1] * 20, output=3, gap_s=0),
                                   dict(delta=[2] * 30, output=3, gap_s=0)])
    workload = tmp_path / 'w.json'
    workload.write_text(json.dumps(dict(tasks=[small, big])))
    args = argparse.Namespace(workload=str(workload), steps_limit=0, num_jobs=4, k=[2], jobs_active=[1],
                              gap=['fixed:0.01'], arrival=['batch'], repeats=1, repeat_start=0)
    server = dict(P=2, ctx_size=80, ctx_per_slot=limit, kv_unified=False, cache_ram_mib=0)
    async with ModelClient('http://test', transport=httpx.MockTransport(handler)) as client:
        await bench.macro(client, tmp_path, args, server)
    jobs = [json.loads(s) for s in (tmp_path / 'raw/macro_jobs.jsonl').read_text().splitlines()]
    assert len(jobs) == 4
    assert {j['task_id'] for j in jobs if j['status'] == 'ok'} == {'s'}
    assert all(j['steps'] == 1 for j in jobs if j['task_id'] == 'b')
    spans = sorted((j['admitted_s'], j['done_s']) for j in jobs)
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(spans, spans[1:])), 'J=1 must serialise tasks'
    errors = [json.loads(s) for s in (tmp_path / 'raw/macro_requests.jsonl').read_text().splitlines()]
    assert any('exceeds' in r.get('error', '') for r in errors)
