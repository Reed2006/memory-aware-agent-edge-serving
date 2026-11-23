import asyncio
import json

import httpx
import pytest

from phase2 import bench


class FakeConfig:
    def __init__(self, name='fifo', **kwargs):
        self.name = name
        self.concurrency = 2
        self.slots = 2
        self.ctx_per_slot = 64
        self.jobs_active = 0
        self.checkpoint = False
        self.checkpoint_budget_bytes = 0
        self.__dict__.update(kwargs)


class FakeScheduler:
    seen = []

    def __init__(self, client, config, request_log, decision_log, **kwargs):
        self.client, self.log = client, request_log
        self.metrics = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def submit(self, task_id, body, metadata=None):
        import time
        FakeScheduler.seen.append((task_id, dict(body), dict(metadata)))
        now = time.perf_counter()
        self.metrics.setdefault(task_id, dict(arrival_s=now, admitted_s=now,
            first_dispatch_s=now, admission_wait_s=0, first_dispatch_wait_s=0,
            cumulative_service_s=0))
        response = await self.client.complete(body)
        self.log.write(dict(metadata, status='ok', enqueue_s=now, dispatch_s=now,
                            done_s=time.perf_counter(), timings=response['timings']))
        return response

    async def tool_start(self, task_id):
        pass

    async def tool_end(self, task_id):
        pass

    async def finish(self, task_id, status='ok'):
        pass

    def task_metrics(self, task_id):
        return self.metrics.get(task_id, {})


def workload(tmp_path):
    path = tmp_path / 'workload.json'
    path.write_text(json.dumps({'tasks': [dict(task_id='one', steps=[
        dict(delta=[1, 2], output=2, gap_s=0),
        dict(delta=[3], output=1, gap_s=0)])]}))
    return path


def server(**kwargs):
    return dict(parallel=2, ctx_size=128, ctx_per_slot=64, kv_unified=False,
                cache_ram_mib=0, model_hash='a' * 64, kv_precision='f16', **kwargs)


def handler(request):
    if request.url.path == '/props':
        return httpx.Response(200, json={'total_slots': 2,
            'default_generation_settings': {'n_ctx': 64}})
    if request.url.path == '/completion':
        body = json.loads(request.content)
        return httpx.Response(200, json={'tokens': [7] * body['n_predict'],
            'id_slot': 0, 'timings': {'prompt_n': len(body['prompt']),
                                    'cache_n': 0, 'prompt_ms': 1, 'predicted_ms': 1}})
    return httpx.Response(200, json={})


@pytest.mark.asyncio
async def test_replay_is_append_only_and_audit_has_no_future(tmp_path):
    FakeScheduler.seen = []
    row = await bench.run_replay(workload(tmp_path), tmp_path / 'out', FakeConfig(),
        transport=httpx.MockTransport(handler), scheduler_factory=FakeScheduler,
        server_metadata=server(), sample_resources=False, gap='zero', arrival='scaled:1e9')
    assert row['completed'] == 1 and row['failed'] == 0
    assert [x[1]['prompt'] for x in FakeScheduler.seen] == [[1, 2], [1, 2, 7, 7, 3]]
    assert all(not {'steps', 'gap_s', 'future', 'expected_steps'} & x[2].keys()
               for x in FakeScheduler.seen)
    jobs = bench.read_jsonl(tmp_path / 'out/raw/macro_jobs.jsonl')
    assert jobs[0]['steps'] == jobs[0]['expected_steps'] == 2
    assert jobs[0]['output_tokens'] == 3 and jobs[0]['prefix_verified']
    assert row['schema_version'] == 2 and len(row['param_hash']) == 64
    assert row['model_hash'] == 'a' * 64


@pytest.mark.asyncio
async def test_skip_done_and_quarantine_partial_are_identity_safe(tmp_path):
    path = workload(tmp_path)
    kwargs = dict(transport=httpx.MockTransport(handler), scheduler_factory=FakeScheduler,
                  server_metadata=server(), sample_resources=False, gap='zero', arrival='scaled:1e9')
    first = await bench.run_replay(path, tmp_path / 'out', FakeConfig(), **kwargs)
    second = await bench.run_replay(path, tmp_path / 'out', FakeConfig(), skip_done=True, **kwargs)
    assert second['skipped'] and first['label'] == second['label']
    other = await bench.run_replay(path, tmp_path / 'out', FakeConfig(name='affinity'), **kwargs)
    assert other['label'] != first['label']
    raw = tmp_path / 'partial/raw'
    raw.mkdir(parents=True)
    (raw / 'macro_jobs.jsonl').write_text(json.dumps({'label': first['label'], 'job': 99}) + '\n')
    bench.quarantine_partial(tmp_path / 'partial', first['label'])
    assert not bench.read_jsonl(raw / 'macro_jobs.jsonl')
    assert bench.read_jsonl(raw / 'interrupted/macro_jobs.jsonl')[0]['job'] == 99


@pytest.mark.asyncio
async def test_failures_keep_denominator_and_never_log_error_body(tmp_path):
    def fail(request):
        if request.url.path == '/completion':
            return httpx.Response(400, json={'error': 'DO_NOT_LOG_PRIVATE_BODY'})
        return handler(request)
    row = await bench.run_replay(workload(tmp_path), tmp_path / 'out', FakeConfig(),
        transport=httpx.MockTransport(fail), scheduler_factory=FakeScheduler,
        server_metadata=server(), sample_resources=False, gap='zero', arrival='scaled:1e9')
    assert row['tasks'] == row['failed'] == 1 and row['completed'] == 0
    text = (tmp_path / 'out/raw/macro_jobs.jsonl').read_text()
    assert 'HTTPStatusError' in text and 'DO_NOT_LOG_PRIVATE_BODY' not in text


@pytest.mark.asyncio
async def test_output_mismatch_and_task_timeout_are_failures(tmp_path):
    def short(request):
        if request.url.path == '/completion':
            return httpx.Response(200, json={'tokens': [], 'timings': {}})
        return handler(request)
    kw = dict(scheduler_factory=FakeScheduler, server_metadata=server(),
              sample_resources=False, gap='zero', arrival='scaled:1e9')
    row = await bench.run_replay(workload(tmp_path), tmp_path / 'short', FakeConfig(),
                                 transport=httpx.MockTransport(short), **kw)
    assert row['failed'] == 1
    async def slow(request):
        if request.url.path == '/completion':
            await asyncio.sleep(.05)
        return handler(request)
    row = await bench.run_replay(workload(tmp_path), tmp_path / 'timeout', FakeConfig(),
        transport=httpx.MockTransport(slow), timeout_s=.005, **kw)
    assert row['failed'] == 1
    assert bench.read_jsonl(tmp_path / 'timeout/raw/macro_jobs.jsonl')[0]['error_type'] == 'TimeoutError'


def test_server_and_workload_validation(tmp_path):
    with pytest.raises(ValueError, match='RAM'):
        bench.validate_server(FakeConfig(name='ram-cache'), server())
    with pytest.raises(ValueError, match='slots'):
        bench.validate_server(FakeConfig(slots=3), server())
    invalid = tmp_path / 'invalid.json'
    invalid.write_text(json.dumps({'tasks': [{'task_id': 'bad', 'steps': []}]}))
    with pytest.raises(ValueError):
        bench.load_workload(invalid)


def test_matrix_order_is_rotated_and_cli_preserves_trial_start():
    configs = [FakeConfig('fifo'), FakeConfig('affinity'), FakeConfig('joint')]
    order = bench.matrix_order(configs, [291], 3, trial_start=2)
    assert [trial for _, _, trial, _ in order] == [2] * 3 + [3] * 3 + [4] * 3
    assert len({tuple(c.name for c, s, t, o in order if t == trial) for trial in [2, 3, 4]}) == 3
    args = bench.parser().parse_args(['--workload', 'w.json', '--trial-start', '2'])
    assert args.trial_start == 2


@pytest.mark.asyncio
async def test_real_scheduler_mock_end_to_end_and_checkpoint_cleanup(tmp_path):
    from phase2.state import PolicyConfig
    cache = tmp_path / 'cache'
    cache.mkdir()
    def backend(request):
        response = handler(request)
        if request.url.path == '/completion':
            body = json.loads(request.content)
            data = json.loads(response.content)
            data['id_slot'] = body.get('id_slot', 0)
            return httpx.Response(200, json=data)
        return response
    for name in ['fifo', 'affinity', 'joint-checkpoint']:
        row = await bench.run_replay(workload(tmp_path), tmp_path / name,
            PolicyConfig.for_policy(name, ctx_per_slot=64),
            transport=httpx.MockTransport(backend), server_metadata=server(slot_save_path=str(cache)),
            sample_resources=False, gap='zero', arrival='scaled:1e9')
        assert row['completed'] == 1 and row['synthetic_performance']
        requests = bench.read_jsonl(tmp_path / name / 'raw/macro_requests.jsonl')
        assert all(r['label'] == row['label'] and r['seed'] == 291 for r in requests)


@pytest.mark.asyncio
async def test_transport_uncertainty_aborts_and_matrix_does_not_continue(tmp_path):
    from phase2.state import PolicyConfig
    from phase2.scheduler import BackendStateUncertain
    completions = []
    def disconnect(request):
        if request.url.path == '/completion':
            completions.append(1)
            raise httpx.ReadTimeout('PRIVATE_TIMEOUT_DETAIL', request=request)
        return handler(request)
    with pytest.raises(BackendStateUncertain):
        await bench.run_matrix(workload(tmp_path), tmp_path / 'out',
            [PolicyConfig.for_policy('fifo', ctx_per_slot=64),
             PolicyConfig.for_policy('affinity', ctx_per_slot=64)], [291],
            transport=httpx.MockTransport(disconnect), server_metadata=server(),
            sample_resources=False, gap='zero', arrival='scaled:1e9')
    assert len(completions) == 1
    meta = bench.read_jsonl(tmp_path / 'out/raw/runs_meta.jsonl')
    assert meta[-1]['event'] == 'aborted' and meta[-1]['backend_state_uncertain']
    assert 'PRIVATE_TIMEOUT_DETAIL' not in json.dumps(meta)
    assert not bench.read_jsonl(tmp_path / 'out/raw/macro_runs.jsonl')


def test_profile_and_server_flags_are_identity_inputs(tmp_path):
    assert bench.canonical_hash({'k': 1}) != bench.canonical_hash({'k': 2})
    with pytest.raises(ValueError, match='non-RAM'):
        bench.validate_server(FakeConfig(), dict(server(), cache_idle_slots=True))
    with pytest.raises(ValueError, match='slot-save'):
        bench.validate_server(FakeConfig(checkpoint=True), server())
    with pytest.raises(ValueError, match='precision'):
        bench.validate_server(FakeConfig(), dict(server(), kv_precision='q8_0'))


def test_cli_handles_full_config_overrides_and_trial_identity(tmp_path, monkeypatch):
    from phase2.state import PolicyConfig
    record = tmp_path / 'server.json'
    record.write_text(json.dumps(server()))
    captured = {}
    async def matrix(*args, **kwargs):
        captured['args'], captured['kwargs'] = args, kwargs
        return []
    monkeypatch.setattr(bench, 'run_matrix', matrix)
    overrides = {'fifo': PolicyConfig.for_policy('fifo', ctx_per_slot=64).to_dict()}
    overrides['fifo'].pop('name')
    monkeypatch.setattr('sys.argv', ['bench', '--workload', 'w.json', '--server-record', str(record),
        '--policy-overrides', json.dumps(overrides), '--trial-start', '2', '--trial-order-start', '4',
        '--platform-id', 'm4-local'])
    bench.main()
    assert captured['args'][2][0].ctx_per_slot == 64
    assert captured['kwargs']['trial_start'] == 2 and captured['kwargs']['trial_order_start'] == 4
    assert captured['kwargs']['platform_id'] == 'm4-local'


def test_backend_lease_rejects_concurrent_cooperating_clients(tmp_path):
    info = server(slot_save_path=str(tmp_path))
    with bench.backend_lease(info, 'http://test'):
        with pytest.raises(RuntimeError, match='exclusive'):
            with bench.backend_lease(info, 'http://test'):
                pass
    with bench.backend_lease(info, 'http://test'):
        pass


def test_uncertain_backend_taint_survives_new_client_until_restart(tmp_path):
    info = server(slot_save_path=str(tmp_path), pid=123, process_create_time=1)
    with pytest.raises(httpx.ReadTimeout):
        with bench.backend_lease(info, 'http://test'):
            raise httpx.ReadTimeout('uncertain')
    with pytest.raises(RuntimeError, match='restart'):
        with bench.backend_lease(info, 'http://test'):
            pass
    with bench.backend_lease(dict(info, process_create_time=2), 'http://test'):
        pass


def test_loopback_aliases_and_stale_record_cannot_bypass_backend_lock(tmp_path):
    info = server(slot_save_path=str(tmp_path), pid=123, process_create_time=1)
    with bench.backend_lease(info, 'http://localhost:18080'):
        with pytest.raises(RuntimeError, match='exclusive'):
            with bench.backend_lease(dict(info, pid=456), 'http://127.0.0.1:18080'):
                pass


def test_real_backend_identity_requires_matching_port_and_process(monkeypatch):
    info = server(pid=123, process_create_time=1, port=18080)
    with pytest.raises(ValueError, match='port'):
        bench._validate_backend_process(info, 'http://127.0.0.1:9999')
    class Process:
        def exe(self):
            return '/usr/bin/llama-server'
        def create_time(self):
            return 2
    monkeypatch.setattr(bench.psutil, 'Process', lambda pid: Process())
    with pytest.raises(ValueError, match='replaced'):
        bench._validate_backend_process(info, 'http://127.0.0.1:18080')
    info['process_create_time'] = 2
    bench._validate_backend_process(info, 'http://127.0.0.1:18080')


@pytest.mark.asyncio
async def test_job_records_label_output_digest_and_mix_reference(tmp_path):
    path = tmp_path / 'mix.json'
    path.write_text(json.dumps({'mix': 'short-heavy', 'source_workload_hash': 'f' * 64,
                                'tasks': json.loads(workload(tmp_path).read_text())['tasks']}))
    row = await bench.run_replay(path, tmp_path / 'out', FakeConfig(),
        transport=httpx.MockTransport(handler), scheduler_factory=FakeScheduler,
        server_metadata=server(), sample_resources=False, gap='zero', arrival='scaled:1e9',
        policy_label='best-fixed')
    assert row['policy_label'] == 'best-fixed' and row['reference_workload_hash'] == 'f' * 64
    assert row['workload_variant'] == 'short-heavy' and row['checkpoint_stats'] is None
    job = bench.read_jsonl(tmp_path / 'out/raw/macro_jobs.jsonl')[0]
    assert len(job['output_sha']) == 16


def test_real_runs_refuse_battery_power(monkeypatch):
    monkeypatch.setattr(bench, 'power_state', lambda: "Now drawing from 'Battery Power'")
    with pytest.raises(ValueError, match='AC power'):
        bench.require_ac_power()
    bench.require_ac_power(allow_battery=True)
    monkeypatch.setattr(bench, 'power_state', lambda: "Now drawing from 'AC Power'")
    bench.require_ac_power()
    monkeypatch.setattr(bench, 'power_state', lambda: None)  # Linux / EC2
    bench.require_ac_power()
