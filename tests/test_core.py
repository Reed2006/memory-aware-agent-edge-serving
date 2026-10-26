import asyncio

import httpx
import pytest

from phase1.core import Fifo, Jsonl, ModelClient, append_step, common_prefix


@pytest.mark.asyncio
async def test_fifo_order_bound_and_failure_recovery(tmp_path):
    started, active, peak = [], 0, 0

    async def operation(payload):
        nonlocal active, peak
        started.append(payload['i'])
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.005)
            if payload['i'] == 2:
                raise ValueError('expected failure')
            return {'ok': payload['i']}
        finally:
            active -= 1

    async with Fifo(2, operation, Jsonl(tmp_path / 'requests.jsonl')) as fifo:
        results = await asyncio.gather(
            *(fifo.submit({'i': i}, {'job': str(i)}) for i in range(6)),
            return_exceptions=True,
        )
    assert started == list(range(6))
    assert peak == 2 and active == 0
    assert isinstance(results[2], ValueError) and results[-1]['ok'] == 5


@pytest.mark.asyncio
async def test_tool_wait_releases_capacity_and_preserves_dependency(tmp_path):
    events = []

    async def operation(p):
        events.append(p['name'])
        return {}

    async with Fifo(1, operation, Jsonl(tmp_path / 'requests.jsonl')) as fifo:
        async def agent_a():
            await fifo.submit({'name': 'a1'}, {})
            await asyncio.sleep(0.02)
            await fifo.submit({'name': 'a2'}, {})
        async def agent_b():
            await asyncio.sleep(0.005)
            await fifo.submit({'name': 'b1'}, {})
        await asyncio.gather(agent_a(), agent_b())
    assert events == ['a1', 'b1', 'a2']


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_capacity(tmp_path):
    async def operation(p):
        await asyncio.sleep(0.01)
        return p
    async with Fifo(1, operation, Jsonl(tmp_path / 'r.jsonl')) as fifo:
        first = asyncio.create_task(fifo.submit({'i': 1}, {}))
        await asyncio.sleep(0.002)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert (await fifo.submit({'i': 2}, {}))['i'] == 2


def test_replay_retains_actual_generation_and_has_exact_length():
    previous = [1, 2, 3]
    generated = [9, 8]
    prompt = append_step(previous, generated, [4, 5])
    assert prompt == [1, 2, 3, 9, 8, 4, 5]
    assert common_prefix(prompt, previous + generated) == 5
    assert common_prefix([1, 8], [1, 2]) == 1


@pytest.mark.asyncio
async def test_native_http_client_and_error_status():
    async def handler(request):
        if request.url.path == '/completion':
            return httpx.Response(200, json={'timings': {'predicted_n': 8}})
        return httpx.Response(503)
    async with ModelClient('http://test', transport=httpx.MockTransport(handler)) as client:
        assert (await client.complete({'n_predict': 8}))['timings']['predicted_n'] == 8
        with pytest.raises(httpx.HTTPStatusError):
            await client.post('/invalid', {})


def test_invalid_concurrency_and_jsonl_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        Fifo(0, None, None)
    import json
    path = tmp_path / 'events.jsonl'
    log = Jsonl(path)
    log.write({'status': 'failed', 'error_type': 'timeout'})
    assert json.loads(path.read_text())['status'] == 'failed'


@pytest.mark.asyncio
async def test_fifo_http_pipeline_logs_all_terminal_results(tmp_path):
    import json
    async def handler(request):
        body = json.loads(request.content)
        if body['n_predict'] == 999:
            return httpx.Response(503)
        return httpx.Response(200, json={'tokens': [7] * body['n_predict'],
                                        'timings': {'predicted_n': body['n_predict']}})
    path = tmp_path / 'pipeline.jsonl'
    async with ModelClient('http://test', transport=httpx.MockTransport(handler)) as client:
        async with Fifo(1, client.complete, Jsonl(path)) as fifo:
            rows = await asyncio.gather(
                fifo.submit({'prompt': [1, 2], 'n_predict': 8}, {'task_id': 'good'}),
                fifo.submit({'prompt': [3], 'n_predict': 999}, {'task_id': 'bad'}),
                return_exceptions=True)
    logged = [json.loads(s) for s in path.read_text().splitlines()]
    assert len(rows[0]['tokens']) == 8
    assert [r['status'] for r in logged] == ['ok', 'error']
    assert [r['task_id'] for r in logged] == ['good', 'bad']
