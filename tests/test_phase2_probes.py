import json

import httpx
import pytest

from phase1.core import ModelClient
from phase2 import probes


@pytest.mark.asyncio
async def test_affinity_and_incremental_probe_sequence_and_metrics(tmp_path):
    calls = []
    cache = {}

    def handle(request):
        body = json.loads(request.content)
        calls.append((request.url.path, str(request.url.query), body))
        if request.url.path != '/completion':
            cache.clear()
            return httpx.Response(200, json={})
        prompt = body['prompt']
        slot = body.get('id_slot', 0)
        old = cache.get(slot, [])
        n = 0
        for a, b in zip(old, prompt):
            if a != b:
                break
            n += 1
        out = [8] * body['n_predict']
        cache[slot] = prompt + out[:-1]
        return httpx.Response(200, json={'tokens': out, 'id_slot': slot,
            'timings': {'prompt_n': len(prompt) - n, 'cache_n': n,
                        'prompt_ms': len(prompt) - n, 'predicted_ms': 1}})

    async with ModelClient('http://test', transport=httpx.MockTransport(handle)) as client:
        rows = await probes.run_probes(client, ['affinity', 'incremental-recompute'], tmp_path,
            shared_prefix_tokens=4, suffix_lengths=[2], output_tokens=2, trials=1)
    assert {r['suite'] for r in rows} == {'affinity', 'incremental-recompute'}
    inc = [r for r in rows if r['suite'] == 'incremental-recompute']
    assert {r['condition'] for r in inc} == {'retain', 'overwrite', 'erase'}
    assert all(r['status'] == 'ok' for r in rows)
    assert any(c[2].get('id_slot') == 1 for c in calls)
    assert (tmp_path / 'raw/probes.jsonl').exists()


@pytest.mark.asyncio
async def test_restore_failure_does_not_unlink_inflight_state(tmp_path):
    calls = []
    path = tmp_path / 'state'
    path.mkdir()
    def handle(request):
        body = json.loads(request.content)
        calls.append(str(request.url))
        if request.url.path == '/completion':
            return httpx.Response(200, json={'tokens': [8] * body['n_predict'],
                'id_slot': body.get('id_slot', 0), 'timings': {}})
        if 'action=save' in str(request.url):
            (path / body['filename']).write_bytes(b'pending-state')
            return httpx.Response(200, json={'n_written': 13, 'n_saved': 7})
        if 'action=restore' in str(request.url):
            return httpx.Response(500, json={'error': 'PRIVATE'})
        return httpx.Response(200, json={})
    async with ModelClient('http://test', transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(probes.ProbeStateUncertain):
            await probes.run_probes(client, ['restore'], tmp_path,
                shared_prefix_tokens=4, suffix_lengths=[2], output_tokens=2, trials=1,
                slot_save_path=path)
    rows = [json.loads(line) for line in (tmp_path / 'raw/probes.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['status'] == 'error'
    assert list(path.iterdir()), 'failed restore leaves state for explicit recovery'
    assert 'PRIVATE' not in (tmp_path / 'raw/probes.jsonl').read_text()


@pytest.mark.asyncio
async def test_successful_restore_cleans_only_this_completed_snapshot(tmp_path):
    unrelated = tmp_path / 'unrelated.bin'
    unrelated.write_bytes(b'keep')
    def handle(request):
        body = json.loads(request.content)
        if request.url.path == '/completion':
            return httpx.Response(200, json={'tokens': [8] * body['n_predict'],
                'id_slot': body.get('id_slot', 0), 'timings': {'cache_n': 7}})
        if 'action=save' in str(request.url):
            (tmp_path / body['filename']).write_bytes(b'complete')
        return httpx.Response(200, json={'n_written': 8, 'n_saved': 7, 'n_restored': 7})
    async with ModelClient('http://test', transport=httpx.MockTransport(handle)) as client:
        rows = await probes.run_probes(client, ['restore'], tmp_path / 'out',
            shared_prefix_tokens=4, suffix_lengths=[2], output_tokens=2, trials=1,
            slot_save_path=tmp_path)
    assert all(row['identical_tokens'] for row in rows)
    assert list(tmp_path.glob('*.bin')) == [unrelated]


def test_calibration_uses_incremental_evaluated_tokens_and_wait_samples():
    profile = probes.calibrate_profile([
        {'suite': 'incremental-recompute', 'status': 'ok', 'concurrency': 1,
         'prompt_tokens': 100, 'reused_tokens': 80, 'prompt_ms': 40},
        {'suite': 'incremental-recompute', 'status': 'ok', 'concurrency': 2,
         'prompt_tokens': 100, 'reused_tokens': 60, 'prompt_ms': 120},
        {'suite': 'interference', 'status': 'ok', 'prompt_tokens': 100,
         'save_s': .2, 'restore_s': .1},
    ])
    assert profile['schema'] == 'phase2-cost-profile-v1'
    assert profile['prefill_min_tokens'] == 128
    assert profile['online_prefill'] is True
    assert profile['prefill']['1']['seconds_per_token'] == pytest.approx(.002)
    assert profile['prefill']['2']['seconds_per_token'] == pytest.approx(.003)
    assert profile['save']['seconds_per_token'] == pytest.approx(.002)
    assert profile['waits_s'] == []


@pytest.mark.asyncio
async def test_all_marks_ram_cache_precondition_without_faking_a_measurement(tmp_path):
    def handle(request):
        body = json.loads(request.content)
        if request.url.path == '/completion':
            return httpx.Response(200, json={'tokens': [8] * body['n_predict'],
                'id_slot': body.get('id_slot', 0), 'timings': {'cache_n': 7}})
        return httpx.Response(200, json={'n_written': 10, 'n_saved': 7, 'n_restored': 7})
    async with ModelClient('http://test', transport=httpx.MockTransport(handle)) as client:
        rows = await probes.run_probes(client, ['all'], tmp_path,
            shared_prefix_tokens=4, suffix_lengths=[2], output_tokens=2, trials=1,
            slot_save_path=tmp_path, stream=False)
    row = [r for r in rows if r['suite'] == 'ram-cache'][0]
    assert row['status'] == 'skipped_precondition' and 'roundtrip_s' not in row
    assert all(r['overlap_attempted'] is False for r in rows if r['suite'] == 'interference')


@pytest.mark.asyncio
async def test_explicit_ram_probe_requires_both_backend_flags(tmp_path):
    async with ModelClient('http://test', transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))) as client:
        with pytest.raises(ValueError, match='cache-ram'):
            await probes.run_probes(client, ['ram-cache'], tmp_path)


@pytest.mark.asyncio
async def test_stream_interference_waits_for_first_token_then_operates(tmp_path):
    import asyncio
    operations = []
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            operations.append('first-token')
            yield b'data: {"content":"a"}\n\n'
            await asyncio.sleep(.005)
            operations.append('second-token')
            yield b'data: {"content":"b"}\n\n'
            yield b'data: [DONE]\n\n'
    def handle(request):
        body = json.loads(request.content)
        if request.url.path == '/completion':
            if body.get('stream'):
                return httpx.Response(200, stream=Stream())
            return httpx.Response(200, json={'tokens': [8] * body['n_predict'],
                'id_slot': body.get('id_slot', 0), 'timings': {}})
        operations.append(str(request.url.query))
        return httpx.Response(200, json={'n_saved': 6, 'n_written': 20})
    async with ModelClient('http://test', transport=httpx.MockTransport(handle)) as client:
        rows = await probes.run_probes(client, ['interference'], tmp_path,
            shared_prefix_tokens=4, suffix_lengths=[2], output_tokens=2, trials=1,
            slot_save_path=tmp_path)
    assert len(rows) == 3 and all(r['status'] == 'ok' for r in rows)
    assert all(r['stream_events'] == 2 and r['overlap_attempted'] for r in rows)
    assert rows[1]['save_s'] >= 0 and rows[2]['restore_s'] >= 0
    for index, value in enumerate(operations):
        if value == 'first-token':
            remaining = operations[index + 1:]
            assert 'second-token' in remaining


@pytest.mark.asyncio
async def test_ram_cache_trigger_and_probe_validation(tmp_path):
    def handle(request):
        body = json.loads(request.content)
        if request.url.path == '/completion':
            return httpx.Response(200, json={'tokens': [8] * body['n_predict'], 'id_slot': body.get('id_slot', 0),
                'timings': {'cache_n': 4, 'prompt_n': 2, 'prompt_ms': 2}})
        return httpx.Response(200, json={})
    async with ModelClient('http://test', transport=httpx.MockTransport(handle)) as client:
        rows = await probes.run_probes(client, ['ram-cache'], tmp_path,
            shared_prefix_tokens=4, suffix_lengths=[2], output_tokens=2, trials=1,
            server_metadata={'cache_ram_mib': 10, 'cache_idle_slots': True})
        assert rows[0]['host_cache_enabled'] and rows[0]['reused_tokens'] == 4
        for options in [dict(suites=['bad']), dict(suites=['affinity'], shared_prefix_tokens=0),
                        dict(suites=['restore']), dict(suites=['affinity'], server_metadata={'parallel': 1})]:
            with pytest.raises(ValueError):
                await probes.run_probes(client, results_dir=tmp_path, **options)


def test_calibrate_cli_writes_compatible_cost_model(tmp_path, monkeypatch):
    from phase2.costs import CostModel
    source, target = tmp_path / 'probe.jsonl', tmp_path / 'costs.json'
    waits = tmp_path / 'dev-decisions.jsonl'
    source.write_text(json.dumps(dict(suite='incremental-recompute', status='ok', concurrency=1,
        prompt_tokens=100, reused_tokens=80, prompt_ms=40, generated_tokens=3,
        predicted_ms=4, platform_id='m4', model_hash='a' * 64, kv_precision='f16')) + '\n')
    waits.write_text(json.dumps(dict(action='observed_wait', status='ok', observed_wait_s=.5,
        workload_split='train', platform_id='m4', model_hash='a' * 64, kv_precision='f16')) + '\n')
    monkeypatch.setattr('sys.argv', ['probes', 'calibrate', '--input', str(source),
        '--waits-input', str(waits), '--output', str(target)])
    probes.main()
    costs = CostModel.from_file(target)
    assert costs.prefill_s(100, 80, 1) == pytest.approx(.04)
    profile = json.loads(target.read_text())
    assert profile['waits_s'] == [.5]
    assert profile['metadata']['platform_id'] == 'm4' and not profile['metadata']['example_only']


def test_calibration_rejects_mixed_hardware_and_synthetic_timings():
    row = dict(suite='incremental-recompute', status='ok', concurrency=1,
               prompt_tokens=10, reused_tokens=0, prompt_ms=1, platform_id='m4',
               model_hash='a' * 64, kv_precision='f16')
    with pytest.raises(ValueError, match='platform'):
        probes.calibrate_profile([row, dict(row, platform_id='t4')])
    with pytest.raises(ValueError, match='synthetic'):
        probes.calibrate_profile([dict(row, synthetic_performance=True)])
    with pytest.raises(ValueError, match='eval'):
        probes.calibrate_profile([row], [dict(action='observed_wait', status='ok',
            observed_wait_s=1, workload_split='test')])


@pytest.mark.asyncio
async def test_restore_greedy_equality_does_not_pass_without_kv_reuse(tmp_path):
    def handle(request):
        body = json.loads(request.content)
        if request.url.path == '/completion':
            return httpx.Response(200, json={'tokens': [8] * body['n_predict'],
                'id_slot': body.get('id_slot', 0), 'timings': {'cache_n': 0}})
        return httpx.Response(200, json={'n_saved': 7, 'n_restored': 7, 'n_written': 20})
    async with ModelClient('http://test', transport=httpx.MockTransport(handle)) as client:
        rows = await probes.run_probes(client, ['restore'], tmp_path,
            shared_prefix_tokens=4, suffix_lengths=[2], output_tokens=2, trials=1,
            slot_save_path=tmp_path)
    assert all(row['status'] != 'ok' and row['identical_tokens'] for row in rows)
    assert all(not row['restore_verified'] for row in rows)


@pytest.mark.asyncio
async def test_calibration_rejects_mock_emitted_rows_even_with_measured_identity(tmp_path):
    def handle(request):
        body = json.loads(request.content)
        if request.url.path == '/completion':
            return httpx.Response(200, json={'tokens': [8] * body['n_predict'],
                'id_slot': body.get('id_slot', 0),
                'timings': {'prompt_n': len(body['prompt']), 'cache_n': 0, 'prompt_ms': 1}})
        return httpx.Response(200, json={})
    async with ModelClient('http://test', transport=httpx.MockTransport(handle)) as client:
        rows = await probes.run_probes(client, ['incremental-recompute'], tmp_path,
            shared_prefix_tokens=4, suffix_lengths=[2], output_tokens=2, trials=1,
            server_metadata={'platform_id': 'm4', 'model_hash': 'a' * 64, 'kv_precision': 'f16'})
    with pytest.raises(ValueError, match='synthetic'):
        probes.calibrate_profile(rows)


def test_prefill_calibration_ignores_retained_continuations():
    rows = [{'suite': 'incremental-recompute', 'status': 'ok', 'concurrency': 1, 'condition': 'retain',
             'prompt_tokens': 5000, 'reused_tokens': 4998, 'prompt_ms': 40},
            {'suite': 'incremental-recompute', 'status': 'ok', 'concurrency': 1, 'condition': 'overwrite',
             'prompt_tokens': 5000, 'reused_tokens': 4000, 'prompt_ms': 5000}]
    profile = probes.calibrate_profile(rows)
    assert profile['prefill']['1']['seconds_per_token'] == pytest.approx(.005)
