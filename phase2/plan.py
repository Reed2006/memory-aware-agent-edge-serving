"""Create, inspect, or explicitly execute a frozen Phase 2 experiment manifest.

Creation and `run` without --execute never contact a server. The executor only
runs the benchmark against an already configured backend; it never starts one.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from phase2.state import PolicyConfig


SCHEMA = 'phase2-experiment-manifest-v1'
DEFAULT_SEEDS = {'dev': [101, 102, 103], 'eval': [1101, 1102, 1103],
                 'sensitivity': [2101, 2102]}


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Cannot read JSON input {path}: {type(error).__name__}') from error
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object in {path}')
    return data


def _file(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'Required reviewed input does not exist: {path}')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


ISOLATED = dict(label='isolated', policy='fixed', config={'concurrency': 1, 'jobs_active': 1})
MIX_WEIGHT = 3  # a "heavy" mix draws its favoured class three times as often


def _format(value: float) -> str:
    return f'{value:g}'.replace('.', 'p')


def _dev_policies(base: dict, aging_grid=None) -> list[dict]:
    # FIFO is fixed K=2/J=inf; TTL 0 is affinity with LRU victims. Exact duplicates
    # cost hours of laptop time and add no candidate, so they are not repeated.
    policies = [dict(label=f'fixed-k{k}-j{j or "inf"}', policy='fixed',
                     config=dict(base, concurrency=k, jobs_active=j))
                for k, j in [(1, 1), (1, 0), (2, 2), (2, 3), (2, 4)]]
    policies += [dict(label='fifo', policy='fifo', config=base.copy()),
                 dict(label='affinity', policy='affinity', config=base.copy())]
    # B2F: once placement is agent-aware, a fixed cap may become a stronger baseline.
    policies += [dict(label=f'affinity-fixed-j{j}', policy='affinity-fixed',
                      config=dict(base, jobs_active=j)) for j in (2, 3)]
    # The selected joint aging must also have measured queue/cache/checkpoint
    # ablations. A single-aging confirmation matrix can be generated with
    # --aging-grid A --only-labels ..., without rerunning the fixed baselines.
    ages = aging_grid or [PolicyConfig().aging_s]
    for aging in ages:
        suffix = f'-a{_format(aging)}' if aging_grid else ''
        for name in ['queue', 'attained', 'cache', 'joint', 'joint-checkpoint']:
            policies.append(dict(label=name + suffix, policy=name,
                                 config=dict(base, aging_s=float(aging))))
        # Include both sides of the 15-second sensitivity pause with a bounded
        # three-value grid; TTL is selected only among the same measured aging.
        policies += [dict(label=f'ttl-{_format(ttl)}' + suffix, policy='ttl',
                          config=dict(base, ttl_s=ttl, aging_s=float(aging)))
                     for ttl in (10., 15., 30.)]
    policies.append(dict(label='ram-cache', policy='ram-cache', config=base.copy(),
                         resources={'cache_ram_mib': 3072, 'cache_idle_slots': True}))
    return policies


def task_work(task: dict) -> int:
    """Offline task size used only to group tasks, never as a policy input."""
    return sum(step['output'] for step in task['steps'])


def make_mix(data: dict, source_hash: str, cutoff: float, kind: str,
             weight: int = MIX_WEIGHT) -> dict:
    if kind not in {'short', 'long'}:
        raise ValueError('mix kind must be short or long')
    short = [task for task in data['tasks'] if task_work(task) <= cutoff]
    long = [task for task in data['tasks'] if task_work(task) > cutoff]
    if not short or not long:
        raise ValueError('a mix needs tasks on both sides of the frozen short cutoff')
    favoured, other = (short, long) if kind == 'short' else (long, short)
    # Duplicating a task in the pool raises its draw probability in job_sources;
    # each occurrence replays the identical recorded trajectory.
    tasks = [task for task in favoured for _ in range(weight)] + list(other)
    return dict({key: value for key, value in data.items() if key != 'tasks'},
                tasks=tasks, mix=f'{kind}-heavy', mix_weight=weight, short_cutoff=cutoff,
                source_workload_hash=source_hash,
                mix_rule=f'{kind} tasks (expected output <= / > cutoff) appear {weight}x in the pool')


def _conditions(phase: str, gap: str, arrival: str) -> list[dict]:
    main = dict(condition='main', gap=gap, arrival=arrival)
    if phase != 'sensitivity':
        return [main]
    kind, separator, rate = arrival.partition(':')
    if not separator or kind not in {'scaled', 'poisson'}:
        raise ValueError('Arrival sensitivity requires scaled:RATE or poisson:RATE')
    rate = float(rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Arrival rate must be positive')
    return [main, dict(condition='arrival-half', gap=gap, arrival=f'{kind}:{rate / 2:g}'),
            dict(condition='arrival-double', gap=gap, arrival=f'{kind}:{rate * 2:g}'),
            dict(condition='gap-zero', gap='zero', arrival=arrival),
            dict(condition='gap-recorded', gap='recorded', arrival=arrival),
            dict(condition='gap-long', gap='fixed:15', arrival=arrival),
            dict(condition='mix-short', gap=gap, arrival=arrival, mix='short'),
            dict(condition='mix-long', gap=gap, arrival=arrival, mix='long')]


def _command(run: dict, manifest: dict, python: str) -> list[str]:
    overrides = {key: value for key, value in run['config'].items() if key != 'name'}
    workload = run.get('workload') or manifest['workload']
    return [python, '-m', 'phase2.bench', '--workload', workload['path'],
            '--policy-label', run['label'],
            '--results-dir', run['results_dir'], '--policies', run['policy'],
            '--seeds', str(run['seed']), '--trials', '1', '--trial-start', str(run['trial']),
            '--trial-order-start', str(run['trial_order']), '--num-jobs', str(manifest['num_jobs']),
            '--gap', run['gap'], '--arrival', run['arrival'], '--base-url', manifest['base_url'],
            '--server-record', manifest['server_record'], '--cost-profile', manifest['cost_profile']['path'],
            '--platform-id', manifest['platform_id'], '--kv-precision', manifest['kv_precision'],
            '--timeout-s', str(manifest['timeout_s']),
            '--request-timeout-s', str(manifest['request_timeout_s']), '--skip-done',
            '--policy-overrides', json.dumps({run['policy']: overrides}, sort_keys=True, separators=(',', ':'))]


def create_manifest(*, phase: str, workload: Path, cost_profile: Path, results_dir: Path,
                    platform_id: str, seeds=None, trials: int = 1, num_jobs: int | None = None,
                    frozen_config: Path | None = None, gap: str = 'lognormal:2:1',
                    arrival: str = 'scaled:1', parallel: int = 2, ctx_size: int = 24576,
                    kv_precision: str = 'f16', cache_ram_mib: int = 0, kv_unified: bool = False,
                    server_record: Path = Path('.server.json'), base_url: str = 'http://127.0.0.1:18080',
                    timeout_s: float = 3600., request_timeout_s: float = 300.,
                    python: str | None = None, aging_grid=None, conditions=None,
                    only_labels=None, trial_labels=None, isolated: bool = True) -> dict:
    if phase not in DEFAULT_SEEDS:
        raise ValueError('phase must be dev, eval, or sensitivity')
    if not platform_id or not re.fullmatch(r'[A-Za-z0-9_.-]+', platform_id):
        raise ValueError('platform_id must be a stable filesystem-safe machine identifier')
    if parallel < 1 or ctx_size < parallel or ctx_size % parallel or trials < 1:
        raise ValueError('positive parallel/trials and evenly partitioned ctx_size are required')
    if kv_unified:
        raise ValueError('Phase 2 manifests currently require partitioned KV for explicit slot capacity')
    if not math.isfinite(timeout_s) or timeout_s <= 0 or cache_ram_mib < 0 or \
            not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
        raise ValueError('timeouts must be positive and cache_ram_mib nonnegative')
    if not kv_precision:
        raise ValueError('kv_precision is required for matched isolated references')
    seeds = list(DEFAULT_SEEDS[phase] if seeds is None else seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ValueError('seeds must be a nonempty list of unique integers')
    num_jobs = num_jobs if num_jobs is not None else (10 if phase == 'dev' else 20)
    if num_jobs < 1:
        raise ValueError('num_jobs must be positive')
    workload_info, cost_info = _file(workload), _file(cost_profile)
    _load(Path(workload_info['path']))
    profile = _load(Path(cost_info['path']))
    if profile.get('schema') != 'phase2-cost-profile-v1':
        raise ValueError('cost_profile must use phase2-cost-profile-v1; calibrate before creating a real manifest')
    from phase2.costs import CostModel
    CostModel(profile)
    base = dict(slots=parallel, ctx_per_slot=ctx_size // parallel, concurrency=min(2, parallel))
    frozen_info, short_cutoff = None, None
    if aging_grid is not None:
        aging_grid = [float(value) for value in aging_grid]
        if not aging_grid or any(not math.isfinite(v) or v <= 0 for v in aging_grid) or \
                len(set(aging_grid)) != len(aging_grid):
            raise ValueError('aging grid needs unique positive finite seconds')
        if phase != 'dev':
            raise ValueError('aging is tuned on dev only; freeze the chosen value in the config')
    if phase == 'dev':
        if parallel < 2:
            raise ValueError('The fixed-candidate dev matrix requires at least two slots')
        policies = _dev_policies(base, aging_grid)
    else:
        if frozen_config is None:
            raise ValueError('eval/sensitivity require an explicit frozen dev config')
        frozen_info = _file(frozen_config)
        frozen = _load(Path(frozen_info['path']))
        if frozen.get('schema') != 'phase2-frozen-config-v1' or frozen.get('frozen') is not True \
                or frozen.get('selected_on') != 'dev':
            raise ValueError('frozen config must explicitly be frozen and selected_on=dev')
        measured_identity = (frozen.get('evidence') or {}).get('comparison_identity', {})
        if measured_identity.get('cost_hash') is not None and \
                measured_identity['cost_hash'] != cost_info['sha256']:
            raise ValueError('cost profile differs from the frozen dev evidence; confirm on dev before eval')
        for key, expected in [('platform_id', platform_id), ('P', parallel), ('ctx_size', ctx_size),
                              ('kv_precision', kv_precision), ('kv_unified', False)]:
            if key in measured_identity and measured_identity[key] != expected:
                raise ValueError(f'{key} differs from the frozen dev evidence')
        policies = frozen.get('policies')
        if not isinstance(policies, list) or not policies:
            raise ValueError('frozen dev config must contain a nonempty policies list')
        short_cutoff = frozen.get('short_cutoff')
        if short_cutoff is None or not isinstance(short_cutoff, (int, float)) or \
                not math.isfinite(short_cutoff) or short_cutoff < 0:
            raise ValueError('frozen dev config must contain a finite nonnegative short_cutoff')
        policies = list(policies)
        if isolated and not any(item.get('label') == ISOLATED['label'] for item in policies):
            # Slowdown needs each eval task's model time when it runs alone.
            policies.insert(0, dict(ISOLATED, reference_only=True))
    if only_labels is not None:
        wanted = set(only_labels) | ({ISOLATED['label']} if isolated and phase != 'dev' else set())
        unknown = set(only_labels) - {item.get('label', item.get('policy')) for item in policies}
        if unknown:
            raise ValueError(f'unknown policy labels: {sorted(unknown)}')
        policies = [item for item in policies if item.get('label', item.get('policy')) in wanted]
    trial_labels = None if trial_labels is None else set(trial_labels)
    manifest = dict(schema=SCHEMA, schema_version=2, phase=phase, platform_id=platform_id,
        workload=workload_info, cost_profile=cost_info, frozen_config=frozen_info,
        short_cutoff=short_cutoff, results_dir=str(Path(results_dir).resolve()), seeds=seeds,
        trials=trials, num_jobs=num_jobs, base_url=base_url, timeout_s=timeout_s,
        request_timeout_s=request_timeout_s,
        server_record=str(Path(server_record).resolve()), kv_precision=kv_precision,
        policy_order='left rotation by trial; identical seed/trial schedule per condition',
        execution_default='dry-run; backend must already be configured', runs=[])
    validated = []
    labels = set()
    for item in policies:
        label, name = item.get('label', item.get('policy')), item.get('policy')
        if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', label) or label in labels:
            raise ValueError('Policy labels must be unique filesystem-safe strings')
        labels.add(label)
        stored = item.get('config', {})
        if not isinstance(stored, dict):
            raise ValueError(f'Invalid config for {label}')
        evidence = item.get('evidence') or {}
        if item.get('param_hash') or evidence:
            if set(stored) != set(PolicyConfig().to_dict()):
                raise ValueError(f'{label}: measured evidence requires a complete frozen config')
        if 'name' in stored:
            if stored['name'] != name:
                raise ValueError(f'{label}: frozen policy name differs from its measured config')
            config = PolicyConfig.from_dict(stored)
        else:
            config = PolicyConfig.for_policy(name, **(base | stored))
        for claimed in (item.get('param_hash'), evidence.get('param_hash')):
            if claimed is not None and claimed != config.param_hash:
                raise ValueError(f'{label}: frozen config hash differs from measured dev evidence')
        if config.slots != parallel or config.ctx_per_slot != ctx_size // parallel:
            raise ValueError('Frozen policy slot capacity must match the manifest resource group')
        resources = dict(P=parallel, ctx_size=ctx_size, kv_unified=False, cache_ram_mib=cache_ram_mib,
                         cache_idle_slots=False)
        extra = item.get('resources', {})
        if not isinstance(extra, dict) or set(extra) - {'cache_ram_mib', 'cache_idle_slots'}:
            raise ValueError('Only cache_ram_mib and cache_idle_slots may vary in policy resources')
        resources.update(extra)
        if resources['cache_ram_mib'] < 0 or not isinstance(resources['cache_idle_slots'], bool):
            raise ValueError('Invalid RAM-cache resource requirements')
        group_key = f'P{parallel}-C{ctx_size}-RAM{resources["cache_ram_mib"]}-idle{int(resources["cache_idle_slots"])}'
        validated.append(dict(label=label, policy=name, config=config.to_dict(), param_hash=config.param_hash,
                              server_requirements=resources, server_group=group_key,
                              reference_only=bool(item.get('reference_only')), evidence=evidence))
    if trial_labels is not None and trial_labels - labels:
        raise ValueError(f'unknown trial labels: {sorted(trial_labels - labels)}')
    python = str(Path(python or sys.executable).resolve())
    available = _conditions(phase, gap, arrival)
    if conditions is not None:
        names = {item['condition'] for item in available}
        if set(conditions) - names:
            raise ValueError(f'unknown conditions {sorted(set(conditions) - names)}; choose from {sorted(names)}')
        available = [item for item in available if item['condition'] in set(conditions)]
    source = _load(Path(workload_info['path']))
    mixes = {}
    for condition in available:
        kind = condition.get('mix')
        if kind and kind not in mixes:
            mix = make_mix(source, workload_info['sha256'], short_cutoff, kind)
            path = Path(manifest['results_dir']) / 'workloads' / f'mix-{kind}-{workload_info["sha256"][:12]}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(mix, indent=1, sort_keys=True) + '\n'
            if not path.exists() or path.read_text() != encoded:
                path.write_text(encoded)
            mixes[kind] = _file(path)
    for condition_index, condition in enumerate(available):
        workload_ref = mixes[condition['mix']] if condition.get('mix') else workload_info
        for seed_index, workload_seed in enumerate(seeds):
            for trial in range(trials):
                members = [item for item in validated
                           if (trial == 0 or trial_labels is None or item['label'] in trial_labels)
                           and not (item['reference_only'] and (trial or seed_index or condition_index))]
                if not members:
                    continue
                # Rotate by seed and trial so no policy always runs first on a cool machine.
                offset = (seed_index + trial) % len(members)
                rotated = members[offset:] + members[:offset]
                if any(item['reference_only'] for item in rotated):
                    # The isolated reference is needed for slowdown; run it first.
                    rotated.sort(key=lambda item: not item['reference_only'])
                for order, item in enumerate(rotated):
                    run = dict(item, seed=workload_seed, trial=trial, trial_order=order,
                               workload=workload_ref, **{k: v for k, v in condition.items() if k != 'mix'})
                    location = Path(manifest['results_dir']) / phase / condition['condition'] / \
                        item['label'] / f'S{workload_seed}-T{trial}'
                    run['results_dir'] = str(location)
                    run['command'] = _command(run, manifest, python)
                    manifest['runs'].append(run)
    return manifest


def estimate_hours(manifest: dict, minutes_per_task: float) -> float:
    """Rough serial duration from a measured completed-task rate on this platform."""
    return sum(manifest['num_jobs'] * minutes_per_task for _ in manifest['runs']) / 60


def _verify_inputs(manifest: dict) -> None:
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unsupported manifest schema')
    items = [(field, manifest.get(field)) for field in ('workload', 'cost_profile', 'frozen_config')]
    items += [('workload', run['workload']) for run in manifest.get('runs', []) if run.get('workload')]
    checked = set()
    for field, item in items:
        if item is None or (item['path'], item['sha256']) in checked:
            continue
        if _file(Path(item['path']))['sha256'] != item['sha256']:
            raise ValueError(f'{field} changed since manifest creation; create a new reviewed manifest')
        checked.add((item['path'], item['sha256']))


def restart_backend(record_path: Path) -> None:
    """Stop and relaunch the recorded backend with identical flags (explicit opt-in only)."""
    from phase2 import server
    info = _load(Path(record_path))
    command = info.get('command', [])
    args = server.parser().parse_args([])
    args.model = Path(info['model_path'])
    args.cpu = bool(info.get('cpu'))
    args.threads = int(command[command.index('--threads') + 1]) if '--threads' in command else args.threads
    args.parallel, args.ctx_size = int(info['parallel']), int(info['ctx_size'])
    args.kv_unified = False
    args.cache_ram = int(info.get('cache_ram_mib') or 0)
    args.cache_idle_slots = bool(info.get('cache_idle_slots'))
    args.port = int(info['port'])
    args.results_dir = Path(info['slot_save_path']).parent
    args.server_record = Path(record_path)
    server.stop(record_path)
    server.start(args)


def _server_matches(path: Path, requirements: dict) -> None:
    data = _load(path)
    data['P'] = data.get('P', data.get('parallel'))
    for key, expected in requirements.items():
        if data.get(key) != expected:
            raise ValueError(f'Existing server {key}={data.get(key)!r}; manifest requires {expected!r}. '
                             'Configure the server manually, or select its matching --server-group.')


def execute_manifest(manifest: dict, *, execute: bool = False, budget_s: float = 3600.,
                     server_record: Path | None = None, server_group: str | None = None,
                     runner=None, restart_server: str = 'never', restarter=None,
                     labels=None, stop_file: Path | None = None, deadline_unix: float | None = None,
                     run_estimate_s: float = 0., clock=time.time) -> dict:
    if not math.isfinite(budget_s) or budget_s < 0:
        raise ValueError('budget_s must be finite and nonnegative')
    if restart_server not in {'never', 'ram-cache', 'always'}:
        raise ValueError('restart_server must be never, ram-cache or always')
    _verify_inputs(manifest)
    selected = [run for run in manifest['runs'] if server_group is None or run['server_group'] == server_group]
    if labels is not None:
        selected = [run for run in selected if run['label'] in set(labels)]
    if not selected:
        raise ValueError('No runs match the selected server group')
    host_cache = [run for run in selected if run['server_requirements'].get('cache_ram_mib', 0) > 0]
    if execute and len(host_cache) > 1 and restart_server == 'never':
        # slot erase does not clear llama.cpp's host prompt cache; a later run of the
        # same tasks would start warm and look faster than a cold first run.
        raise ValueError('host-cache runs share warm prompt-cache state across runs; '
                         'pass --restart-server ram-cache so each starts from a fresh backend')
    result = dict(dry_run=not execute, planned=len(selected), completed=0, failed=0, stopped=None,
                  skipped=0)
    if not execute:
        for run in selected:
            print(f'# server-group {run["server_group"]}; {run["label"]}; seed={run["seed"]}; trial={run["trial"]}')
            print(shlex.join(run['command']))
        return result
    if budget_s == 0:
        result['stopped'] = 'budget'
        return result
    runner = runner or _run_child
    deadline = time.monotonic() + budget_s
    server_path = Path(server_record or manifest['server_record'])
    estimate = max(0., float(run_estimate_s))
    for run in selected:
        if run_done(run, manifest):
            result['skipped'] += 1  # finished in an earlier session
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            result['stopped'] = 'budget'
            break
        if stop_file is not None and Path(stop_file).exists():
            result['stopped'] = 'stop_file'
            break
        if deadline_unix is not None and clock() + estimate > deadline_unix:
            # Do not start a run that cannot finish before the operator cuts the session.
            result['stopped'] = 'deadline'
            break
        _verify_inputs(manifest)
        _server_matches(server_path, run['server_requirements'])
        if restart_server == 'always' or (restart_server == 'ram-cache' and
                                          run['server_requirements'].get('cache_ram_mib', 0) > 0):
            (restarter or restart_backend)(server_path)
            _server_matches(server_path, run['server_requirements'])
        # Rebuild argv from validated data: saved display commands cannot inject executables.
        config = PolicyConfig.from_dict(run['config'])
        if config.param_hash != run['param_hash']:
            raise ValueError('Policy config changed since manifest creation')
        command = _command(run, manifest, sys.executable)
        if server_record is not None:
            command[command.index('--server-record') + 1] = str(server_path.resolve())
        # This is a launch budget. Killing this client could release its lock
        # while llama-server is still generating or writing a checkpoint.
        started = clock()
        completed = runner(command, shell=False)
        if completed.returncode:
            result['failed'] += 1
            result['stopped'] = 'benchmark_failure'
            break
        result['completed'] += 1
        estimate = max(estimate, clock() - started)  # learn the real run length
    return result


def terminal_record(run: dict, manifest: dict) -> dict | None:
    """The terminal record with this run's exact identity in its own directory, if any."""
    path = Path(run['results_dir']) / 'raw/macro_runs.jsonl'
    if not path.exists():
        return None
    workload = (run.get('workload') or manifest['workload'])['sha256']
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (row.get('param_hash') == run['param_hash'] and row.get('workload_hash') == workload
                and row.get('cost_hash') == manifest['cost_profile']['sha256']
                and row.get('seed') == run['seed'] and row.get('trial') == run['trial']
                and row.get('gap_mode') == run['gap'] and row.get('arrival') == run['arrival']):
            return row
    return None


def run_done(run: dict, manifest: dict) -> bool:
    """A terminal record with this run's exact identity exists in its own directory."""
    return terminal_record(run, manifest) is not None


def _run_child(command, **options):
    """Run one benchmark; on Ctrl-C let it drain backend work instead of killing it."""
    process = subprocess.Popen(command, **options)
    try:
        return subprocess.CompletedProcess(command, process.wait())
    except KeyboardInterrupt:
        # The benchmark shares this process group and received the same SIGINT.
        # It cancels its tasks, waits for in-flight model calls and writes an
        # 'aborted' record; the next session quarantines and reruns that run.
        try:
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    create = actions.add_parser('create', help='Write a reviewable manifest; print commands only')
    create.add_argument('--phase', choices=DEFAULT_SEEDS, required=True)
    for name in ('workload', 'cost-profile', 'results-dir', 'manifest'):
        create.add_argument(f'--{name}', type=Path, required=True)
    create.add_argument('--platform-id', required=True)
    create.add_argument('--frozen-config', type=Path)
    create.add_argument('--seeds', help='Comma-separated workload seeds; repeats are independent of trials')
    create.add_argument('--trials', type=int, default=1)
    create.add_argument('--num-jobs', type=int)
    create.add_argument('--gap', default='lognormal:2:1')
    create.add_argument('--arrival', default='scaled:1')
    create.add_argument('--parallel', type=int, default=2)
    create.add_argument('--ctx-size', type=int, default=24576)
    create.add_argument('--kv-precision', default='f16')
    create.add_argument('--cache-ram-mib', type=int, default=0)
    create.add_argument('--server-record', type=Path, default=Path('.server.json'))
    create.add_argument('--base-url', default='http://127.0.0.1:18080')
    create.add_argument('--timeout-s', type=float, default=3600.)
    create.add_argument('--request-timeout-s', type=float, default=300.)
    create.add_argument('--aging-grid', help='dev only: comma-separated aging seconds for joint')
    create.add_argument('--conditions', help='Comma-separated condition names (sensitivity subsets)')
    create.add_argument('--only-labels', help='Comma-separated policy labels to keep')
    create.add_argument('--trial-labels', help='Labels that get all --trials; others run trial 0 only')
    create.add_argument('--no-isolated', dest='isolated', action='store_false',
                        help='Omit the K=1/J=1 isolated reference run (slowdown then needs --reference-dir)')
    create.add_argument('--minutes-per-task', type=float,
                        help='Measured serial minutes per completed task, to print a duration estimate')
    show = actions.add_parser('show', help='Print commands and required resource groups')
    show.add_argument('manifest', type=Path)
    run = actions.add_parser('run', help='Dry-run unless --execute is explicitly provided')
    run.add_argument('manifest', type=Path)
    run.add_argument('--execute', action='store_true')
    run.add_argument('--budget-s', type=float, default=3600.,
                     help='Soft launch budget; an already started benchmark drains safely and may finish later')
    run.add_argument('--server-record', type=Path)
    run.add_argument('--server-group')
    run.add_argument('--labels', help='Comma-separated policy labels to run from this manifest')
    run.add_argument('--restart-server', choices=['never', 'ram-cache', 'always'], default='never',
                     help='Explicitly relaunch the recorded backend before matching runs')
    run.add_argument('--stop-file', type=Path, help='Do not launch another run once this file exists')
    run.add_argument('--deadline-unix', type=float,
                     help='Do not launch a run expected to end after this wall-clock time')
    run.add_argument('--run-estimate-s', type=float, default=0.,
                     help='Initial run-length estimate; replaced by the longest run seen')
    args = parser.parse_args(argv)
    split = (lambda value: [item for item in value.split(',') if item] if value else None)
    try:
        if args.action == 'create':
            options = vars(args).copy()
            del options['action']
            path = options.pop('manifest')
            minutes = options.pop('minutes_per_task')
            options['seeds'] = [int(s) for s in args.seeds.split(',')] if args.seeds else None
            options['aging_grid'] = split(args.aging_grid)
            for key in ('conditions', 'only_labels', 'trial_labels'):
                options[key] = split(options[key])
            manifest = create_manifest(**options)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
            print(f'Manifest written: {path.resolve()} ({len(manifest["runs"])} runs; none executed)')
            if minutes:
                print(f'Estimated serial time: {estimate_hours(manifest, minutes):.1f} h '
                      f'at {minutes:g} min per task (re-estimate after the first runs)')
            execute_manifest(manifest)
        elif args.action == 'show':
            execute_manifest(_load(args.manifest))
        else:
            result = execute_manifest(_load(args.manifest), execute=args.execute, budget_s=args.budget_s,
                                      server_record=args.server_record, server_group=args.server_group,
                                      restart_server=args.restart_server, labels=split(args.labels),
                                      stop_file=args.stop_file, deadline_unix=args.deadline_unix,
                                      run_estimate_s=args.run_estimate_s)
            print(json.dumps(result, sort_keys=True))
            if result['failed']:
                raise SystemExit(1)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
