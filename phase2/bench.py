"""Frozen trajectory replay. Only this harness can inspect future steps or waits."""

import argparse
import asyncio
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil

from phase1.bench import (Sampler, arrival_times, gap_delay, job_sources, payload,
                          platform_info, power_state)
from phase1.core import Jsonl, ModelClient, append_step, common_prefix


SCHEMA_VERSION = 2


def thermal_state() -> str | None:
    """macOS thermal/performance warnings; a fanless laptop may throttle during long sweeps."""
    try:
        return subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True,
                              check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def gpu_memory() -> list[dict] | None:
    """NVIDIA memory in use; llama.cpp preallocates KV, so start/end snapshots suffice."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,temperature.gpu,'
                              'clocks.sm', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True, timeout=10).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    fields = ['uuid', 'memory_used_mib', 'memory_total_mib', 'temperature_c', 'sm_clock_mhz']
    return [dict(zip(fields, (value.strip() for value in line.split(','))))
            for line in out.splitlines() if line.strip()]


def machine_state() -> dict:
    return dict(power=power_state(), thermal=thermal_state(), gpu=gpu_memory(),
                recorded_unix=time.time())


def require_ac_power(allow_battery: bool = False) -> None:
    """Phase 1 discarded a battery-powered run; real Phase 2 timings follow the same rule."""
    state = power_state()
    if state and 'Battery Power' in state and not allow_battery:
        raise ValueError('real timings require AC power; connect the charger '
                         '(or pass --allow-battery for a labelled non-comparable smoke test)')


class BoundLog:
    """Give checkpoint and scheduler events the same experiment identity."""

    def __init__(self, log, context):
        self.log, self.context = log, context

    def write(self, value):
        self.log.write(self.context | value)


@contextmanager
def backend_lease(info: dict, base_url: str):
    """Exclude simultaneous Phase 2 clients; other HTTP clients must also stay idle."""
    url = urlsplit(base_url)
    host = url.hostname
    host = 'loopback' if host in {'localhost', '127.0.0.1', '::1'} else host
    endpoint = dict(host=host, port=url.port or (443 if url.scheme == 'https' else 80),
                    test_scope=info.get('_test_scope'))
    identity = canonical_hash({key: info.get(key) for key in
        ['pid', 'process_create_time', '_test_scope']})
    path = Path(tempfile.gettempdir()) / f'phase2-backend-{canonical_hash(endpoint)}.lock'
    state_path = path.with_suffix('.state.json')
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    acquired = False
    state = {'uncertain': False}
    def write_state(status):
        encoded = json.dumps(dict(state=status, backend_identity=identity,
                                  holder_pid=os.getpid(), updated_unix=time.time())).encode()
        # The lock inode must stay fixed; its separate state is atomically replaced.
        # A process killed mid-write leaves the previous active state intact.
        temporary_fd, temporary_path = tempfile.mkstemp(prefix=state_path.name + '.', dir=state_path.parent)
        try:
            with os.fdopen(temporary_fd, 'wb') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, state_path)
        finally:
            Path(temporary_path).unlink(missing_ok=True)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another Phase 2 client holds the exclusive backend lease') from error
        acquired = True
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        if previous.get('backend_identity') == identity and previous.get('state') in {'active', 'uncertain'}:
            acquired = False
            raise RuntimeError('backend state is uncertain; inspect and restart the server before reuse')
        write_state('active')
        try:
            yield state
        except BaseException as error:
            if _is_uncertain(error):
                state['uncertain'] = True
            raise
    finally:
        if acquired:
            write_state('uncertain' if state['uncertain'] else 'completed')
        os.close(descriptor)


def _is_uncertain(error):
    return (not isinstance(error, Exception) or isinstance(error, httpx.RequestError)
            or type(error).__name__ in {'BackendStateUncertain', 'CheckpointUncertain', 'ProbeStateUncertain'})


def _validate_backend_process(info, base_url):
    url = urlsplit(base_url)
    port = url.port or (443 if url.scheme == 'https' else 80)
    if port != info.get('port'):
        raise ValueError('base URL port does not match backend metadata')
    if info.get('pid') is None:
        raise ValueError('real runs require recorded backend process identity')
    process = psutil.Process(info['pid'])
    if 'llama-server' not in Path(process.exe()).name:
        raise ValueError('recorded backend PID is not llama-server')
    recorded = info.get('process_create_time', info.get('started_unix'))
    if recorded is None or abs(process.create_time() - recorded) > (0.01 if 'process_create_time' in info else 5):
        raise ValueError('recorded backend process was replaced')
    info['process_create_time'] = process.create_time()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_workload(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    tasks = data.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('workload requires a nonempty tasks list')
    for task in tasks:
        if not isinstance(task.get('task_id'), (str, int)) or not task.get('steps'):
            raise ValueError('each task requires an ID and nonempty steps')
        for step in task['steps']:
            delta, output, gap = step.get('delta'), step.get('output'), step.get('gap_s', 0)
            if (not isinstance(delta, list) or not all(type(t) is int and t >= 0 for t in delta)
                    or type(output) is not int or output < 1
                    or not isinstance(gap, (float, int)) or not math.isfinite(gap) or gap < 0):
                raise ValueError('invalid replay tokens, output length, or recorded wait')
    return data


def config_dict(config) -> dict:
    return dataclasses.asdict(config) if dataclasses.is_dataclass(config) else vars(config).copy()


def validate_server(config, server: dict) -> None:
    slots = server.get('parallel', server.get('P'))
    if slots != config.slots:
        raise ValueError('policy slots do not match the running server')
    if server.get('ctx_per_slot') != config.ctx_per_slot:
        raise ValueError('policy context budget does not match the running server')
    if server.get('kv_unified', False):
        raise ValueError('Phase 2 slot policies require partitioned KV')
    if server.get('kv_precision', 'f16') != 'f16':
        raise ValueError('the pinned experiment requires f16 KV precision')
    ram = server.get('cache_ram_mib', 0) or 0
    idle = server.get('cache_idle_slots', False)
    if config.name == 'ram-cache':
        if ram <= 0 or not idle:
            raise ValueError('RAM-cache baseline requires --cache-ram and --cache-idle-slots')
    elif ram or idle:
        raise ValueError('non-RAM policies require host cache and idle-slot cache disabled')
    if config.checkpoint and not server.get('slot_save_path'):
        raise ValueError('checkpoint policy requires the backend slot-save directory')


def quarantine_partial(results: Path, label: str) -> None:
    """Exclude unfinished attempts from aggregates while retaining their audit records."""
    for name in ['macro_requests.jsonl', 'macro_jobs.jsonl', 'resources.jsonl',
                 'policy_decisions.jsonl', 'runs_meta.jsonl']:
        path = Path(results) / 'raw' / name
        if not path.exists():
            continue
        rows = read_jsonl(path)
        partial = [row for row in rows if row.get('label') == label]
        if not partial:
            continue
        target = Jsonl(path.parent / 'interrupted' / name)
        for row in partial:
            target.write(row)
        temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                      for row in rows if row.get('label') != label))
        os.replace(temporary, path)


def matrix_order(configs, seeds, trials: int, trial_start: int = 0):
    if not configs or trials < 1 or trial_start < 0:
        raise ValueError('matrix requires policies and positive trials')
    cases = []
    for seed_index, seed in enumerate(seeds):
        for trial in range(trial_start, trial_start + trials):
            offset = (seed_index + trial) % len(configs)
            rotated = list(configs[offset:]) + list(configs[:offset])
            cases.extend((config, seed, trial, order) for order, config in enumerate(rotated))
    return cases


def _server_metadata(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f'backend metadata file does not exist: {path}')
    info = json.loads(path.read_text())
    if not info.get('model_hash'):
        command = info.get('command', [])
        model = info.get('model_path')
        if not model and '-m' in command:
            model = command[command.index('-m') + 1]
        if not model or not Path(model).is_file():
            raise ValueError('backend metadata requires a verifiable model hash')
        info['model_hash'] = file_hash(Path(model))
    info.setdefault('kv_precision', 'f16')
    info.setdefault('cache_idle_slots', '--cache-idle-slots' in info.get('command', []))
    return info


def _commit() -> str | None:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


async def run_replay(workload: Path, results_dir: Path, config, *,
                     base_url='http://127.0.0.1:18080', seed=291, trial=0,
                     gap='lognormal:2:1', arrival='scaled:1', num_jobs=None,
                     timeout_s=3600, request_timeout_s=300, skip_done=False,
                     server_metadata=None, server_record=Path('.server.json'),
                     profile_path=None, platform_id=None, trial_order=0,
                     transport=None, scheduler_factory=None, sample_resources=True,
                     warmup=None, policy_label=None, allow_battery=False) -> dict:
    """Run one paired input. HTTP/transport uncertainty aborts instead of reusing slots."""
    from phase2.costs import CostModel
    if scheduler_factory is None:
        from phase2.scheduler import Scheduler
        scheduler_factory = Scheduler

    workload, results = Path(workload), Path(results_dir)
    data = load_workload(workload)
    if timeout_s <= 0 or request_timeout_s <= 0:
        raise ValueError('timeouts must be positive')
    count = len(data['tasks']) if num_jobs is None else num_jobs
    if count < 1:
        raise ValueError('num_jobs must be positive')
    info = dict(server_metadata) if server_metadata is not None else _server_metadata(Path(server_record))
    if transport is None:
        require_ac_power(allow_battery)
        _validate_backend_process(info, base_url)
    else:
        info['_test_scope'] = str(results.resolve())
    validate_server(config, info)
    if not info.get('model_hash'):
        raise ValueError('backend model_hash is required')
    costs = CostModel.from_file(profile_path) if profile_path else CostModel()
    if profile_path:
        profile_metadata = json.loads(Path(profile_path).read_text()).get('metadata', {})
        if transport is None and profile_metadata.get('example_only'):
            raise ValueError('example-only cost profile cannot drive a real experiment')
        for key, expected in [('model_hash', info.get('model_hash')),
                              ('kv_precision', info.get('kv_precision', 'f16')),
                              ('platform_id', platform_id or info.get('platform_id'))]:
            if profile_metadata.get(key) is not None and profile_metadata[key] != expected:
                raise ValueError(f'cost profile {key} does not match this backend')
    cost_hash = file_hash(Path(profile_path)) if profile_path else canonical_hash(costs.to_dict())
    parameters = config_dict(config)
    param_hash, workload_hash = canonical_hash(parameters), file_hash(workload)
    platform_id = platform_id or info.get('platform_id')
    if not platform_id:
        if transport is None:
            raise ValueError('real runs require an explicit hardware-specific platform_id')
        platform_id = f'mock-{platform.system()}-{platform.machine()}'
    warmup = transport is None if warmup is None else warmup
    identity = dict(model_hash=info['model_hash'], platform_id=platform_id,
                    gap_mode=gap, arrival=arrival, num_jobs=count,
                    server={k: info.get(k) for k in ['parallel', 'ctx_size', 'ctx_per_slot',
                                                    'kv_unified', 'cache_ram_mib', 'cache_idle_slots', 'kv_precision']})
    label = (f'phase2-{config.name}-{param_hash[:12]}-W{workload_hash[:12]}'
             f'-C{cost_hash[:12]}-{canonical_hash(identity)[:12]}-S{seed}-T{trial}')
    run_log = Jsonl(results / 'raw/macro_runs.jsonl')
    existing = [row for row in read_jsonl(run_log.path) if row.get('label') == label]
    if existing:
        if skip_done:
            return dict(existing[-1], skipped=True)
        raise ValueError('this run identity is already complete; use skip_done or another trial')
    quarantine_partial(results, label)
    common = dict(schema_version=SCHEMA_VERSION, label=label, run_id=label,
        policy=config.name, policy_label=policy_label or config.name,
        policy_config=parameters, param_hash=param_hash,
        workload_hash=workload_hash, cost_hash=cost_hash, model_hash=info['model_hash'],
        # Derived mixes keep their source tasks' isolated references.
        reference_workload_hash=data.get('source_workload_hash') or workload_hash,
        workload_variant=data.get('mix', 'source'),
        platform_id=platform_id, kv_precision=info.get('kv_precision', 'f16'),
        seed=seed, trial=trial, trial_order=trial_order,
        synthetic_performance=transport is not None,
        tasks=count, num_jobs=count,
        workload_split=data.get('source_split', data.get('split')),
        P=config.slots, ctx_size=info.get('ctx_size'), ctx_per_slot=config.ctx_per_slot,
        kv_unified=False, cache_ram_mib=info.get('cache_ram_mib', 0),
        k=config.concurrency, j=config.jobs_active, gap_mode=gap, arrival=arrival)
    inputs = job_sources(data['tasks'], count, seed)
    arrivals = arrival_times(arrival, count, seed)
    # Validate delay syntax before changing any server state.
    gap_delay(gap, 0, f'{seed}-validation')
    snapshot = results / 'workloads' / f'{workload_hash}.json'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes(workload.read_bytes())
    metadata_log = Jsonl(results / 'raw/runs_meta.jsonl')
    real = transport is None
    metadata_log.write(dict(common, event='start', git_commit=_commit(),
                            server=info, profile_path=str(profile_path) if profile_path else None,
                            started_unix=time.time(), synthetic_performance=not real,
                            machine=machine_state() if real else None,
                            platform=platform_info() if real else None))
    request_log = Jsonl(results / 'raw/macro_requests.jsonl')
    decision_log = BoundLog(Jsonl(results / 'raw/policy_decisions.jsonl'), common)
    job_log = Jsonl(results / 'raw/macro_jobs.jsonl')
    rows = []
    checkpoint_stats = None
    sampler = None
    sampler_task = None
    lease = backend_lease(info, base_url)
    lease_acquired = False
    lease_state = None
    try:
        lease_state = lease.__enter__()
        lease_acquired = True
        async with ModelClient(base_url, transport=transport, timeout=request_timeout_s) as client:
            props = await client.props()
            if props.get('total_slots', config.slots) != config.slots:
                raise ValueError('live backend slots disagree with recorded metadata')
            actual_ctx = props.get('default_generation_settings', {}).get('n_ctx')
            if actual_ctx is not None and actual_ctx != config.ctx_per_slot:
                raise ValueError('live backend context disagrees with recorded metadata')
            if warmup:
                await client.complete(payload([1] * 5, 16))
            metadata_log.write(dict(common, event='warmup', performed=warmup))
            await client.erase(config.slots)
            checkpoints = None
            if config.checkpoint:
                from phase2.checkpoints import CheckpointStore
                checkpoints = CheckpointStore(client, Path(info['slot_save_path']),
                    config.checkpoint_budget_bytes, log=decision_log)
            try:
                if sample_resources:
                    sampler = Sampler(label, results)
                    sampler.pid = info.get('pid')
                    sampler_task = asyncio.create_task(sampler.run())
                start = time.perf_counter()
                async with scheduler_factory(client, config, request_log, decision_log,
                                              costs=costs, checkpoints=checkpoints) as scheduler:
                    async def job(index, source, offset):
                        await asyncio.sleep(max(0, start + offset - time.perf_counter()))
                        arrived = time.perf_counter()
                        runtime_id = f'{label}:job:{index}'
                        prompt, generated = [], []
                        total_output, done_steps, gap_total = 0, 0, 0.0
                        prefix_verified, error_type, status = True, None, 'ok'
                        digest = hashlib.sha256()

                        async def trajectory():
                            nonlocal prompt, generated, total_output, done_steps, gap_total, prefix_verified
                            for step_index, step in enumerate(source['steps']):
                                if step_index:
                                    delay = gap_delay(gap, step.get('gap_s', 0), f'{seed}-{index}-{step_index}')
                                    if not math.isfinite(delay) or delay < 0:
                                        raise ValueError('invalid generated wait')
                                    gap_total += delay
                                    await scheduler.tool_start(runtime_id)
                                    await asyncio.sleep(delay)
                                    await scheduler.tool_end(runtime_id)
                                previous = prompt + generated
                                prompt = append_step(prompt, generated, step['delta'])
                                prefix_verified &= common_prefix(previous, prompt) == len(previous)
                                reusable = max(0, len(previous) - 1)
                                audit = dict(common, job=index, task_id=str(source['task_id']),
                                    source_task_id=str(source['task_id']), runtime_task_id=runtime_id,
                                    step=step_index, expected_output=step['output'],
                                    reusable_tokens=reusable, new_tokens=len(prompt) - reusable,
                                    prefix_verified=prefix_verified)
                                response = await scheduler.submit(runtime_id, payload(prompt, step['output']), audit)
                                generated = response.get('tokens', [])
                                if (not isinstance(generated, list) or len(generated) != step['output']
                                        or not all(type(t) is int for t in generated)
                                        or response.get('truncated', False)):
                                    raise ValueError('output length mismatch, invalid tokens, or truncation')
                                total_output += len(generated)
                                digest.update(json.dumps(generated, separators=(',', ':')).encode())
                                done_steps += 1
                        try:
                            await asyncio.wait_for(trajectory(), timeout=timeout_s)
                        except Exception as error:
                            status, error_type = 'error', type(error).__name__
                            if error_type in {'BackendStateUncertain', 'CheckpointUncertain'}:
                                raise
                        finally:
                            await scheduler.finish(runtime_id, status=status)
                        done = time.perf_counter()
                        metrics = scheduler.task_metrics(runtime_id)
                        admitted = metrics.get('admitted_s') or arrived
                        row = dict(common, job=index, task_id=str(source['task_id']),
                            source_task_id=str(source['task_id']), runtime_task_id=runtime_id,
                            arrival_s=arrived, admitted_s=admitted,
                            first_dispatch_s=metrics.get('first_dispatch_s'), done_s=done,
                            admission_wait_s=metrics.get('admission_wait_s', admitted - arrived),
                            first_dispatch_wait_s=metrics.get('first_dispatch_wait_s'),
                            cumulative_service_s=metrics.get('cumulative_service_s', 0),
                            gap_total_s=gap_total, status=status, error_type=error_type,
                            steps=done_steps, expected_steps=len(source['steps']),
                            output_tokens=total_output,
                            # Greedy decoding: equal digests show identical generated work.
                            output_sha=digest.hexdigest()[:16],
                            expected_output_tokens=sum(step['output'] for step in source['steps']),
                            max_input_tokens=sum(len(step['delta']) + step['output'] for step in source['steps'])
                                             - source['steps'][-1]['output'],
                            prefix_verified=prefix_verified, latency_s=done - arrived,
                            service_s=done - admitted)
                        job_log.write(row)
                        return row
                    tasks = [asyncio.create_task(job(i, source, offset))
                        for i, (source, offset) in enumerate(zip(inputs, arrivals))]
                    gathered = asyncio.gather(*tasks)
                    try:
                        if sampler_task is not None:
                            finished, _ = await asyncio.wait([gathered, sampler_task],
                                                            return_when=asyncio.FIRST_COMPLETED)
                            if sampler_task in finished:
                                await sampler_task
                                raise RuntimeError('resource sampler stopped unexpectedly')
                        rows = await gathered
                    except BaseException:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                backend_drained_s = time.perf_counter()
            finally:
                if checkpoints is not None:
                    checkpoints.close()
                    checkpoint_stats = dict(checkpoints.stats, budget_bytes=checkpoints.budget_bytes)
    except BaseException as error:
        if lease_state is not None and _is_uncertain(error):
            lease_state['uncertain'] = True
        metadata_log.write(dict(common, event='aborted', error_type=type(error).__name__,
                                backend_state_uncertain=isinstance(error, httpx.RequestError) or
                                type(error).__name__ in {'BackendStateUncertain', 'CheckpointUncertain'},
                                done_unix=time.time()))
        raise
    finally:
        if sampler is not None:
            sampler.stop.set()
            result = await asyncio.gather(sampler_task, return_exceptions=True)
            if isinstance(result[0], BaseException):
                metadata_log.write(dict(common, event='resource_sampler_error',
                                        error_type=type(result[0]).__name__))
        if lease_acquired:
            lease.__exit__(None, None, None)
    elapsed = max(backend_drained_s, max(row['done_s'] for row in rows)) - min(row['arrival_s'] for row in rows)
    completed = sum(row['status'] == 'ok' for row in rows)
    result = dict(common, tasks=len(rows), completed=completed, failed=len(rows) - completed,
                  makespan_s=elapsed, jobs_per_minute=completed / elapsed * 60 if elapsed else 0,
                  backend_drained_s=backend_drained_s, checkpoint_stats=checkpoint_stats,
                  synthetic_performance=transport is not None, warmup_performed=warmup)
    run_log.write(result)
    metadata_log.write(dict(common, event='completed', done_unix=time.time(),
                            machine=machine_state() if real else None))
    return result


async def run_matrix(workload, results_dir, configs, seeds, trials=1, *,
                     trial_start=0, trial_order_start=0, **kwargs):
    results = []
    for config, seed, trial, order in matrix_order(configs, seeds, trials, trial_start):
        # An uncertain backend propagates out; a later run must never reset its slots.
        results.append(await run_replay(workload, results_dir, config, seed=seed,
            trial=trial, trial_order=trial_order_start + order, **kwargs))
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload', type=Path, required=True)
    p.add_argument('--results-dir', type=Path, default=Path('runs/phase2'))
    p.add_argument('--policies', default='fifo')
    p.add_argument('--policy-overrides', default='{}', help='JSON: per-policy overrides, or shared fields')
    p.add_argument('--seeds', default='291')
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--trial-start', type=int, default=0)
    p.add_argument('--trial-order-start', type=int, default=0)
    p.add_argument('--gap', default='lognormal:2:1')
    p.add_argument('--arrival', default='scaled:1')
    p.add_argument('--num-jobs', type=int)
    p.add_argument('--base-url', default='http://127.0.0.1:18080')
    p.add_argument('--server-record', type=Path, default=Path('.server.json'))
    p.add_argument('--cost-profile', type=Path)
    p.add_argument('--platform-id')
    p.add_argument('--kv-precision', choices=['f16'], default='f16')
    p.add_argument('--timeout-s', type=float, default=3600)
    p.add_argument('--request-timeout-s', type=float, default=300)
    p.add_argument('--skip-done', action='store_true')
    p.add_argument('--no-resource-sampling', action='store_true')
    p.add_argument('--policy-label', help='Reporting label from the manifest (one policy per call)')
    p.add_argument('--allow-battery', action='store_true',
                   help='Permit a battery-powered smoke test; never use for reported numbers')
    return p


def main():
    from phase2.state import PolicyConfig
    args = parser().parse_args()
    info = _server_metadata(args.server_record)
    overrides = json.loads(args.policy_overrides)
    configs = []
    if args.policy_label and ',' in args.policies:
        raise SystemExit('--policy-label names exactly one policy')
    for name in args.policies.split(','):
        specific = overrides.get(name, {}) if name in overrides else {
            key: value for key, value in overrides.items() if not isinstance(value, dict)}
        options = dict(slots=info.get('parallel', info.get('P')),
                       ctx_per_slot=info['ctx_per_slot']) | specific
        configs.append(PolicyConfig.for_policy(name, **options))
    rows = asyncio.run(run_matrix(args.workload, args.results_dir, configs,
        [int(seed) for seed in args.seeds.split(',')], args.trials,
        trial_start=args.trial_start, trial_order_start=args.trial_order_start,
        base_url=args.base_url, gap=args.gap, arrival=args.arrival,
        num_jobs=args.num_jobs, timeout_s=args.timeout_s,
        request_timeout_s=args.request_timeout_s, skip_done=args.skip_done,
        server_metadata=info, profile_path=args.cost_profile,
        platform_id=args.platform_id, sample_resources=not args.no_resource_sampling,
        policy_label=args.policy_label, allow_battery=args.allow_battery))
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
