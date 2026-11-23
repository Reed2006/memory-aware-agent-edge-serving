"""Model-free Phase 2 summaries with seed-clustered, equal-work comparisons.

Trials of one seed are repeated measurements, not independent workloads. Failed
or incomplete runs remain in completion/throughput counts, but are never used to
claim a paired speedup. Slowdown requires matching isolated model timings.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SUSPEND_S = 120.  # wall-clock time beyond the makespan that indicates a suspended machine
RESOURCES = ['cache_ram_mib', 'cache_idle_slots', 'checkpoint_budget_bytes']
IDENTITY = ['workload_hash', 'platform_id', 'model_hash', 'kv_precision', 'P', 'ctx_size',
            'kv_unified', 'gap_mode', 'arrival', 'steps_limit', 'cost_hash'] + RESOURCES
CONFIG = IDENTITY + ['policy', 'policy_label', 'param_hash', 'k', 'j', 'synthetic_performance']
# Derived workload mixes reuse their source tasks' isolated timings.
REFERENCE = ['reference_workload_hash', 'platform_id', 'model_hash', 'kv_precision', 'P', 'ctx_size',
             'kv_unified', 'cache_ram_mib', 'cache_idle_slots', 'steps_limit', 'task_id']
# Mechanism and memory columns explain *why* a policy wins or loses.
MECHANISM = ['peak_server_rss_gib', 'min_system_available_gib', 'swap_out_delta_mib',
    'gpu_memory_used_mib', 'mean_ready_wait_s', 'max_ready_wait_s', 'mean_admission_wait_s',
    'max_admission_wait_s', 'leases', 'lease_s', 'no_leases', 'evictions', 'aged_dispatches',
    'lease_owner_dispatches', 'deferred_dispatches', 'reconsidered_decisions', 'saves', 'restores',
    'unused_snapshots', 'output_match_rate']
RUN_COLUMNS = CONFIG + ['run_id', 'label', 'seed', 'trial', 'submitted', 'completed', 'failed',
    'makespan_s', 'jobs_per_min', 'completion_fraction', 'valid_work', 'work_signature',
    'p50_s', 'p95_s', 'p99_s', 'jct_mean_s', 'slowdown_mean', 'slowdown_p95',
    'slowdown_short_mean', 'slowdown_long_mean', 'jain_1_over_slowdown',
    'full_history_hit_rate', 'wasted_prefill_fraction', 'decision_cpu_s', 'io_s', 'io_bytes',
    'decision_count'] + MECHANISM + ['on_battery', 'clock_gap_s', 'failure_types', 'run_status']
SAMPLE_COLUMNS = ['sample_scope', 'seed_ids', 'trial_ids', 'trials_by_seed']
POLICY_COLUMNS = CONFIG + SAMPLE_COLUMNS + ['runs', 'seeds', 'submitted', 'completed', 'failed',
    'jobs_per_min_mean', 'jobs_per_min_sd', 'jobs_per_min_trial_sd', 'jct_mean_s', 'p50_s',
    'p95_s', 'p99_s', 'slowdown_mean', 'slowdown_p95', 'slowdown_short_mean',
    'slowdown_long_mean', 'jain_1_over_slowdown', 'full_history_hit_rate',
    'wasted_prefill_fraction', 'decision_cpu_s', 'io_s', 'io_bytes', 'decision_count'] + MECHANISM
PAIR_COLUMNS = IDENTITY + ['policy', 'policy_label', 'param_hash', 'baseline', 'baseline_param_hash',
    'sample_scope', 'required_trials', 'paired_seed_ids', 'matched_trials_by_seed',
    'candidate_trials_by_seed', 'baseline_trials_by_seed', 'paired_trial_pairs',
    'paired_seeds', 'eligible_seeds', 'excluded_seeds', 'excluded_reasons', 'throughput_ratio',
    'throughput_ratio_ci_low', 'throughput_ratio_ci_high', 'jct_ratio',
    'jct_ratio_ci_low', 'jct_ratio_ci_high', 'bootstrap_seed', 'bootstrap_samples']
JOB_COLUMNS = CONFIG + ['run_id', 'label', 'seed', 'trial', 'job', 'task_id', 'status',
    'latency_s', 'gap_total_s', 'expected_steps', 'expected_output_tokens',
    'isolated_compute_s', 'isolated_s', 'slowdown', 'task_class']
REQUEST_COLUMNS = CONFIG + ['run_id', 'label', 'seed', 'trial', 'job', 'task_id', 'step',
    'status', 'prompt_n', 'cache_n', 'prompt_ms', 'predicted_ms', 'wasted_prefill',
    'full_history_hit', 'model_s']


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f'{path}:{line}: invalid JSON') from error
        if not isinstance(row, dict) or row.get('schema_version') != 2:
            raise ValueError(f'{path}:{line}: expected schema_version=2')
        rows.append(row)
    return rows


def _load(root: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    collected = ([], [], [], [])
    seen = set()
    paths = {path.parent for name in ('macro_runs.jsonl', 'runs_meta.jsonl') for path in root.rglob(name)
             if path.parent.name == 'raw' and 'interrupted' not in path.parts}
    for directory in sorted(paths):
        path = directory / 'macro_runs.jsonl'
        runs = _read(path)
        history = {}
        for meta in _read(directory / 'runs_meta.jsonl'):
            run_id = str(meta.get('run_id', meta.get('label', '')))
            prior = history.get(run_id, {})
            lifecycle = prior.get('_lifecycle_status', 'incomplete')
            if meta.get('event') == 'aborted':
                lifecycle = 'aborted'
            elif meta.get('event') == 'start':
                lifecycle = 'incomplete'
            history[run_id] = prior | meta | {'_lifecycle_status': lifecycle}
        closed = {str(run.get('run_id', run.get('label', ''))) for run in runs}
        for run_id, meta in history.items():
            if run_id not in closed:
                duration = _number(meta.get('done_unix')) - _number(meta.get('started_unix'))
                runs.append(dict(meta, run_id=run_id,
                    run_status=meta['_lifecycle_status'],
                    makespan_s=duration if duration > 0 else np.nan))
        lookup = {}
        for run in runs:
            required = [key for key in IDENTITY if key not in RESOURCES + ['steps_limit']] \
                + ['policy', 'param_hash', 'seed', 'trial']
            missing = [key for key in required if key not in run or run[key] is None or run[key] == '']
            if missing:
                raise ValueError(f'{path}: incomplete run identity: {", ".join(missing)}')
            run_id = str(run.get('run_id', run.get('label', '')))
            if not run_id or run_id in seen:
                raise ValueError(f'{path}: missing or duplicate run_id {run_id!r}')
            seen.add(run_id)
            prior = history.get(run_id, {})
            server = prior.get('server') or {}
            config = run.get('policy_config') or {}
            run = dict(run, run_id=run_id, label=run.get('label', run_id))
            run.setdefault('steps_limit', 0)
            run.setdefault('synthetic_performance', False)
            run.setdefault('run_status', 'completed')
            run.setdefault('cache_ram_mib', server.get('cache_ram_mib', 0))
            run.setdefault('cache_idle_slots', server.get('cache_idle_slots', False))
            run.setdefault('checkpoint_budget_bytes', config.get('checkpoint_budget_bytes', 0))
            run.setdefault('policy_label', run['policy'])
            run.setdefault('reference_workload_hash', run['workload_hash'])
            machine = prior.get('machine') or {}
            run['on_battery'] = 'Battery Power' in str(machine.get('power') or '')
            # Wall time beyond the monotonic makespan means the machine was suspended
            # (lid closed) during the run; its timings are not comparable.
            wall = _number(prior.get('done_unix')) - _number(prior.get('started_unix'))
            run['clock_gap_s'] = wall - _number(run.get('makespan_s')) \
                if run['run_status'] == 'completed' else np.nan
            gpus = machine.get('gpu') or []
            run['gpu_memory_used_mib'] = max((_number(gpu.get('memory_used_mib')) for gpu in gpus),
                                             default=np.nan)
            if 'planned_tasks' in prior:
                run['planned_tasks'] = prior['planned_tasks']
            lookup[run_id] = run
            collected[0].append(run)
        samples: dict[str, list[dict]] = {}
        for row in _read_resources(directory / 'resources.jsonl'):
            samples.setdefault(str(row.get('label', '')), []).append(row)
        for run_id, run in lookup.items():
            rows = samples.get(run['label'], [])
            rss = [_number(r.get('server_rss')) for r in rows]
            available = [_number(r.get('system_available')) for r in rows]
            swap = [_number(r.get('swap_out')) for r in rows]
            run['peak_server_rss_gib'] = np.nanmax(rss) / 2**30 if np.isfinite(rss).any() else np.nan
            run['min_system_available_gib'] = np.nanmin(available) / 2**30 \
                if np.isfinite(available).any() else np.nan
            finite = [value for value in swap if np.isfinite(value)]
            run['swap_out_delta_mib'] = (finite[-1] - finite[0]) / 2**20 if finite else np.nan
        for index, filename in enumerate(['macro_jobs.jsonl', 'macro_requests.jsonl',
                                          'policy_decisions.jsonl'], 1):
            for row in _read(directory / filename):
                run_id = str(row.get('run_id', row.get('label', '')))
                if run_id not in lookup:
                    continue  # interrupted runs have no terminal run record
                run = lookup[run_id]
                for key in CONFIG + ['seed', 'trial']:
                    if key in row and key in run and row[key] != run[key]:
                        raise ValueError(f'{path.parent / filename}: inconsistent {key} for {run_id}')
                collected[index].append(run | row | {'run_id': run_id})
    return collected


def _read_resources(path: Path) -> list[dict]:
    """Sampler rows carry no schema version; read them leniently."""
    if not path.exists():
        return []
    rows = []
    for text in path.read_text().splitlines():
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue  # a killed sampler may leave a partial last line
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _number(value) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _mean(values) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def _quantile(values, q: float) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else np.nan


def _jain(values) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(values.sum() ** 2 / (len(values) * np.square(values).sum())) if len(values) else np.nan


def _key(row: dict, keys: list[str]) -> tuple:
    return tuple(row.get(key) if not isinstance(row.get(key), (dict, list)) else
                 json.dumps(row[key], sort_keys=True) for key in keys)


def _request(row: dict) -> dict:
    row = row.copy()
    timings = row.get('timings') or {}
    for key in ('prompt_n', 'cache_n', 'prompt_ms', 'predicted_ms'):
        row[key] = _number(timings.get(key))
    ok = row.get('status') == 'ok'
    row['model_s'] = (row['prompt_ms'] + row['predicted_ms']) / 1000 if ok else np.nan
    row['wasted_prefill'] = max(0., row['prompt_n'] - _number(row.get('new_tokens'))) if ok else np.nan
    row['full_history_hit'] = float(row['cache_n'] >= _number(row.get('reusable_tokens'))) \
        if ok and row.get('step', 0) > 0 else np.nan
    return row


def _references(jobs: list[dict], requests: list[dict]) -> tuple[dict, dict]:
    samples: dict[tuple, list[float]] = {}
    outputs: dict[tuple, set] = {}
    by_job: dict[tuple, list[dict]] = {}
    for row in requests:
        by_job.setdefault((row['run_id'], row.get('job', row.get('runtime_task_id'))), []).append(row)
    for job in jobs:
        if job.get('k') != 1 or job.get('j') != 1 or job.get('status') != 'ok' \
                or job.get('run_status') != 'completed':
            continue
        req = by_job.get((job['run_id'], job.get('job', job.get('runtime_task_id'))), [])
        if len(req) != job.get('expected_steps', job.get('steps')) or not req:
            continue
        times = [row['model_s'] for row in req]
        if all(np.isfinite(t) and t >= 0 for t in times) and sum(times) > 0:
            samples.setdefault(_key(job, REFERENCE), []).append(sum(times))
            if job.get('output_sha'):
                outputs.setdefault(_key(job, REFERENCE), set()).add(job['output_sha'])
    return {key: float(np.median(values)) for key, values in samples.items()}, outputs


def _job_metrics(jobs: list[dict], references: tuple[dict, dict],
                 short_cutoff: float | None) -> list[dict]:
    reference, outputs = references
    result = []
    for job in jobs:
        job = job.copy()
        key = _key(job, REFERENCE)
        compute = reference.get(key, np.nan)
        # Greedy decoding with fixed lengths: a matching digest shows the same
        # tokens as running alone. Batching can legitimately perturb numerics,
        # so this is a diagnostic rate, compared across policies, not a gate.
        job['output_match'] = float(job['output_sha'] in outputs[key]) \
            if job.get('status') == 'ok' and job.get('output_sha') and key in outputs else np.nan
        denominator = compute + _number(job.get('gap_total_s'))
        job.update(isolated_compute_s=compute, isolated_s=denominator,
                   slowdown=_number(job.get('latency_s')) / denominator
                   if job.get('status') == 'ok' and np.isfinite(denominator) and denominator > 0 else np.nan)
        work = _number(job.get('expected_output_tokens'))
        job['task_class'] = ('short' if work <= short_cutoff else 'long') \
            if short_cutoff is not None and np.isfinite(work) else None
        result.append(job)
    return result


def _work_signature(jobs: list[dict]) -> str:
    # Occurrences and planned work, rather than achieved throughput, identify equal work.
    fields = ['job', 'task_id', 'expected_steps', 'expected_output_tokens', 'gap_total_s']
    rows = sorted([dict(zip(fields, _key(job, fields))) for job in jobs],
                  key=lambda item: json.dumps(item, sort_keys=True))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _dispatches(actions: list[dict]) -> list[dict]:
    """Dispatch rows that reached the model; a deferred attempt is retried later.

    A deferral is logged after its own dispatch row (same request sequence), so
    it cancels the latest earlier dispatch of that sequence.
    """
    kept: list[dict | None] = []
    latest: dict = {}
    for row in actions:
        if row.get('action') == 'dispatch':
            if row.get('sequence') is not None:
                latest[row['sequence']] = len(kept)
            kept.append(row)
        elif row.get('action') == 'dispatch_deferred' and row.get('sequence') in latest:
            kept[latest.pop(row['sequence'])] = None
    return [row for row in kept if row is not None]


def _run_summary(runs, jobs, requests, decisions) -> pd.DataFrame:
    rows = []
    for run in runs:
        run_id = run['run_id']
        group = [job for job in jobs if job['run_id'] == run_id]
        req = [r for r in requests if r['run_id'] == run_id]
        actions = [d for d in decisions if d['run_id'] == run_id]
        done = [job for job in group if job.get('status') == 'ok']
        makespan = _number(run.get('makespan_s'))
        complete = len(done)
        expected = int(run.get('planned_tasks', run.get('tasks', run.get('submitted', len(group)))))
        valid = bool(group) and expected == len(group) and run['run_status'] == 'completed'
        for job in done:
            valid = valid and job.get('steps') == job.get('expected_steps') \
                and job.get('output_tokens') == job.get('expected_output_tokens') \
                and job.get('prefix_verified', True) is not False
        for job in group:
            valid = valid and sum(r.get('status') == 'ok' for r in req
                                  if r.get('job') == job.get('job')) == job.get('steps', 0)
        evaluated = sum(r['prompt_n'] for r in req if np.isfinite(r['prompt_n']))
        wasted = sum(r['wasted_prefill'] for r in req if np.isfinite(r['wasted_prefill']))
        dispatches = _dispatches(actions)
        waits = [_number(d.get('ready_wait_s')) for d in dispatches]
        if not any(np.isfinite(waits)):
            waits = [_number(r.get('queue_s')) for r in req]
        admission = [_number(j.get('admission_wait_s')) for j in group]
        count = Counter(d.get('action') for d in actions)
        stats = run.get('checkpoint_stats') or {}
        mechanism = {key: run.get(key, np.nan) for key in
                     ['peak_server_rss_gib', 'min_system_available_gib', 'swap_out_delta_mib',
                      'gpu_memory_used_mib']}
        mechanism.update(mean_ready_wait_s=_mean(waits), max_ready_wait_s=_quantile(waits, 1),
            mean_admission_wait_s=_mean(admission), max_admission_wait_s=_quantile(admission, 1),
            # A re-priced version's second no_lease is not another decision not to hold.
            leases=count['lease'], evictions=count['evict'],
            no_leases=sum(d.get('action') == 'no_lease' and not d.get('reconsidered') for d in actions),
            reconsidered_decisions=sum(bool(d.get('reconsidered')) for d in actions),
            deferred_dispatches=count['dispatch_deferred'],
            lease_s=sum(_number(d.get('duration_s', 0)) for d in actions if d.get('action') == 'lease'),
            aged_dispatches=sum(bool(d.get('aged')) for d in dispatches),
            lease_owner_dispatches=sum(d.get('reason') == 'lease_owner' for d in dispatches),
            saves=sum(d.get('action') == 'save' and d.get('status') == 'ok' for d in actions),
            restores=sum(d.get('action') == 'restore' and d.get('status') == 'ok' for d in actions),
            unused_snapshots=stats.get('unused_snapshots', np.nan),
            output_match_rate=_mean(j.get('output_match', np.nan) for j in done))
        rows.append({**{key: run.get(key) for key in CONFIG + ['run_id', 'label', 'seed', 'trial']},
            'submitted': max(expected, len(group)), 'completed': complete,
            'failed': max(expected, len(group)) - complete, 'makespan_s': makespan,
            'jobs_per_min': complete * 60 / makespan if makespan > 0 else np.nan,
            'completion_fraction': complete / expected if expected > 0 else np.nan,
            'valid_work': bool(valid), 'work_signature': _work_signature(group),
            'p50_s': _quantile((j.get('latency_s', np.nan) for j in done), .5),
            'p95_s': _quantile((j.get('latency_s', np.nan) for j in done), .95),
            'p99_s': _quantile((j.get('latency_s', np.nan) for j in done), .99),
            'jct_mean_s': _mean(j.get('latency_s', np.nan) for j in done),
            'slowdown_mean': _mean(j['slowdown'] for j in done),
            'slowdown_p95': _quantile((j['slowdown'] for j in done), .95),
            'slowdown_short_mean': _mean(j['slowdown'] for j in done if j['task_class'] == 'short'),
            'slowdown_long_mean': _mean(j['slowdown'] for j in done if j['task_class'] == 'long'),
            'jain_1_over_slowdown': _jain(1 / j['slowdown'] for j in done if j['slowdown'] > 0),
            'full_history_hit_rate': _mean(r['full_history_hit'] for r in req),
            'wasted_prefill_fraction': wasted / evaluated if evaluated else np.nan,
            'decision_cpu_s': sum(_number(d.get('decision_cpu_s', 0)) for d in actions),
            'io_s': sum(_number(d.get('io_s', d.get('roundtrip_s', 0))) for d in actions
                        if d.get('action') in {'save', 'restore'} and d.get('status') == 'ok'),
            'io_bytes': sum(_number(d.get('io_bytes', d.get('bytes', 0))) for d in actions
                            if d.get('action') in {'save', 'restore'} and d.get('status') == 'ok'),
            'decision_count': len(actions), **mechanism,
            'on_battery': bool(run.get('on_battery')), 'clock_gap_s': run.get('clock_gap_s', np.nan),
            'run_status': run['run_status'],
            'failure_types': ';'.join(sorted({str(j.get('error_type', j.get('status')))
                                             for j in group if j.get('status') != 'ok'}))})
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def _trials_by_seed(runs: pd.DataFrame) -> dict[str, list[int]]:
    return {str(int(seed)): sorted(int(trial) for trial in group.trial.unique())
            for seed, group in runs.groupby('seed', sort=True)}


def _policy_summary(runs: pd.DataFrame, sample_scope='all_strict') -> pd.DataFrame:
    rows = []
    metrics = ['jct_mean_s', 'p50_s', 'p95_s', 'p99_s', 'slowdown_mean', 'slowdown_p95',
        'slowdown_short_mean', 'slowdown_long_mean', 'jain_1_over_slowdown',
        'full_history_hit_rate', 'wasted_prefill_fraction', 'decision_cpu_s', 'io_s',
        'io_bytes', 'decision_count'] + MECHANISM
    for config, group in runs.groupby(CONFIG, dropna=False, sort=True):
        seeds = group.groupby('seed')
        rates = seeds.jobs_per_min.mean()
        row = dict(zip(CONFIG, config))
        row.update(sample_scope=sample_scope,
                   seed_ids=json.dumps(sorted(int(seed) for seed in group.seed.unique())),
                   trial_ids=json.dumps(sorted(int(trial) for trial in group.trial.unique())),
                   trials_by_seed=json.dumps(_trials_by_seed(group), sort_keys=True),
                   runs=len(group), seeds=len(rates), submitted=group.submitted.sum(),
                   completed=group.completed.sum(), failed=group.failed.sum(),
                   jobs_per_min_mean=rates.mean(), jobs_per_min_sd=rates.std(),
                   jobs_per_min_trial_sd=np.sqrt(seeds.jobs_per_min.var().mean()))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors='coerce')
            row[metric] = values.groupby(group.seed).mean().mean()
        rows.append(row)
    return pd.DataFrame(rows, columns=POLICY_COLUMNS)


def _bootstrap(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if len(values) < 2:
        return np.nan, np.nan  # one seed cannot quantify workload uncertainty
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _paired(runs: pd.DataFrame, baseline: str, samples: int, seed: int, *,
            sample_scope='all_strict', confirmatory_trials=(0, 1, 2)) -> pd.DataFrame:
    rows = []
    # Prefer the manifest label (e.g. best-fixed); fall back to the policy name.
    chosen = runs.policy_label == baseline
    if not chosen.any():
        chosen = runs.policy == baseline
    baselines = runs.loc[chosen]
    candidates = runs.loc[~chosen & (runs.policy_label != 'isolated')]
    for config, group in candidates.groupby(CONFIG, dropna=False, sort=True):
        metadata = dict(zip(CONFIG, config))
        compatible = baselines
        for key in [column for column in IDENTITY if column not in RESOURCES]:
            value = metadata[key]
            compatible = compatible.loc[compatible[key].isna()] if pd.isna(value) else compatible.loc[compatible[key] == value]
        matched = compatible
        for key in RESOURCES:
            matched = matched.loc[matched[key] == metadata[key]]
        # An explicitly named baseline with multiple parameterizations is ambiguous.
        variants = list(matched.param_hash.unique())
        if not variants:
            variants = [None]
        for baseline_hash in variants:
            reference = matched.loc[matched.param_hash == baseline_hash]
            ratios, jcts, reasons = [], [], Counter()
            paired_seeds, matched_trials = [], {}
            for workload_seed in sorted(set(group.seed) | set(reference.seed)):
                current = group.loc[group.seed == workload_seed]
                other = reference.loc[reference.seed == workload_seed]
                if sample_scope == 't0':
                    current = current.loc[current.trial == 0]
                    other = other.loc[other.trial == 0]
                if current.empty:
                    reasons['missing_candidate_t0' if sample_scope == 't0' else 'missing_candidate'] += 1
                    continue
                if current.synthetic_performance.any() or (not other.empty and other.synthetic_performance.any()):
                    reasons['synthetic_data'] += 1
                    continue
                if (current.run_status != 'completed').any() or (not other.empty and
                        (other.run_status != 'completed').any()):
                    reasons['aborted_or_incomplete'] += 1
                    continue
                if current.on_battery.any() or (not other.empty and other.on_battery.any()):
                    reasons['battery_power'] += 1
                    continue
                if (current.clock_gap_s > SUSPEND_S).any() or (not other.empty and
                                                              (other.clock_gap_s > SUSPEND_S).any()):
                    reasons['machine_suspended'] += 1
                    continue
                if other.empty:
                    if sample_scope == 't0' and (reference.seed == workload_seed).any():
                        reason = 'missing_baseline_t0'
                    else:
                        reason = 'resource_mismatch' if (compatible.seed == workload_seed).any() else 'missing_baseline'
                    reasons[reason] += 1
                    continue
                if sample_scope == 'confirmatory' and (set(current.trial) != set(confirmatory_trials)
                        or set(other.trial) != set(confirmatory_trials)):
                    reasons['confirmatory_trial_set_mismatch'] += 1
                    continue
                if set(current.trial) != set(other.trial) or current.trial.duplicated().any() \
                        or other.trial.duplicated().any():
                    reasons['trial_set_mismatch'] += 1
                    continue
                if (current.failed > 0).any() or (other.failed > 0).any():
                    reasons['failure'] += 1
                    continue
                if not current.valid_work.all() or not other.valid_work.all() or \
                        set(current.work_signature) != set(other.work_signature) or \
                        len(set(current.work_signature)) != 1:
                    reasons['work_mismatch'] += 1
                    continue
                rate, base_rate = current.jobs_per_min.mean(), other.jobs_per_min.mean()
                if not np.isfinite(rate) or not np.isfinite(base_rate) or base_rate <= 0:
                    reasons['invalid_duration'] += 1
                    continue
                ratios.append(float(rate / base_rate))
                paired_seeds.append(int(workload_seed))
                matched_trials[str(int(workload_seed))] = sorted(int(trial) for trial in current.trial)
                jct, base_jct = current.jct_mean_s.mean(), other.jct_mean_s.mean()
                jcts.append(float(jct / base_jct) if base_jct > 0 else np.nan)
            lo, hi = _bootstrap(ratios, samples, seed)
            jlo, jhi = _bootstrap(jcts, samples, seed)
            rows.append({**{key: metadata[key] for key in IDENTITY + ['policy', 'policy_label', 'param_hash']},
                'baseline': baseline, 'baseline_param_hash': baseline_hash,
                'sample_scope': sample_scope,
                'required_trials': json.dumps([0] if sample_scope == 't0' else list(confirmatory_trials))
                                   if sample_scope != 'all_strict' else 'all_observed_exact_match',
                'paired_seed_ids': json.dumps(paired_seeds),
                'matched_trials_by_seed': json.dumps(matched_trials, sort_keys=True),
                'candidate_trials_by_seed': json.dumps(_trials_by_seed(group), sort_keys=True),
                'baseline_trials_by_seed': json.dumps(_trials_by_seed(reference), sort_keys=True),
                'paired_trial_pairs': sum(len(trials) for trials in matched_trials.values()),
                'paired_seeds': len(ratios), 'eligible_seeds': len(ratios),
                'excluded_seeds': sum(reasons.values()),
                'excluded_reasons': ';'.join(f'{key}:{value}' for key, value in sorted(reasons.items())),
                'throughput_ratio': _mean(ratios), 'throughput_ratio_ci_low': lo,
                'throughput_ratio_ci_high': hi, 'jct_ratio': _mean(jcts),
                'jct_ratio_ci_low': jlo, 'jct_ratio_ci_high': jhi,
                'bootstrap_seed': seed, 'bootstrap_samples': samples})
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _complete_trial_sets(runs: pd.DataFrame, trials: tuple[int, ...]) -> pd.DataFrame:
    groups = [group for _, group in runs.groupby(CONFIG + ['seed'], dropna=False, sort=False)
              if set(group.trial) == set(trials) and not group.trial.duplicated().any()]
    return pd.concat(groups, ignore_index=True) if groups else runs.iloc[:0].copy()


def _report_policy_table(title: str, policies: pd.DataFrame) -> list[str]:
    lines = [f'## {title}', '',
        '| Policy | Seeds / runs | Trial IDs | Completed / submitted | Tasks/min | Mean JCT (s) | P95 (s) | '
        'Short / long slowdown | Output match to isolated |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in policies.to_dict('records'):
        lines.append(f'| {row["policy_label"]} ({str(row["param_hash"])[:8]}) | '
                     f'{row["seeds"]} / {row["runs"]} | {row["trial_ids"]} | '
                     f'{row["completed"]} / {row["submitted"]} | {row["jobs_per_min_mean"]:.4g} | '
                     f'{row["jct_mean_s"]:.4g} | {row["p95_s"]:.4g} | '
                     f'{row["slowdown_short_mean"]:.3g} / {row["slowdown_long_mean"]:.3g} | '
                     f'{row["output_match_rate"]:.3g} |')
    if policies.empty:
        lines += ['', 'No runs have this complete trial scope.']
    return lines + ['', 'Exact seed IDs and per-seed trial IDs are in this scope\'s policy CSV. '
                    'Rows with different seed sets are descriptive; use the paired table for comparisons.', '']


def _report_pair_table(filename: str, paired: pd.DataFrame) -> list[str]:
    lines = [f'Paired results: `{filename}`. Each row uses its explicitly listed seed/trial pairs; '
             'excluded samples are not replaced by an intersection.', '',
             '| Policy | Baseline | Eligible / excluded seeds | Trial pairs | Throughput ratio | JCT ratio | Excluded reasons |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for row in paired.to_dict('records'):
        lines.append(f'| {row["policy_label"]} | {row["baseline"]} | '
                     f'{row["eligible_seeds"]} / {row["excluded_seeds"]} | {row["paired_trial_pairs"]} | '
                     f'{row["throughput_ratio"]:.4g} | {row["jct_ratio"]:.4g} | {row["excluded_reasons"]} |')
    return lines + ['']


def summarize(root: Path, *, baseline='fifo', short_cutoff: float | None = None,
              bootstrap_samples: int = 2000, bootstrap_seed: int = 291,
              reference_dirs=(), confirmatory_trials=(0, 1, 2)) -> pd.DataFrame:
    baselines = [baseline] if isinstance(baseline, str) else list(baseline)
    if not baselines:
        raise ValueError('at least one baseline is required')
    confirmatory_trials = tuple(confirmatory_trials)
    if (len(confirmatory_trials) < 2 or len(set(confirmatory_trials)) != len(confirmatory_trials)
            or any(isinstance(trial, bool) or not isinstance(trial, int) or trial < 0
                   for trial in confirmatory_trials)):
        raise ValueError('confirmatory_trials must contain at least two distinct nonnegative trial IDs')
    confirmatory_trials = tuple(sorted(confirmatory_trials))
    root = Path(root)
    if bootstrap_samples < 1 or (short_cutoff is not None and
                               (not np.isfinite(short_cutoff) or short_cutoff < 0)):
        raise ValueError('bootstrap_samples must be positive; short_cutoff must be finite and nonnegative')
    runs, jobs, raw_requests, decisions = _load(root)
    requests = [_request(row) for row in raw_requests]
    ref_jobs, ref_requests = list(jobs), list(requests)
    for directory in reference_dirs:
        _, extra_jobs, extra_requests, _ = _load(Path(directory))
        ref_jobs.extend(extra_jobs)
        ref_requests.extend(_request(row) for row in extra_requests)
    jobs = _job_metrics(jobs, _references(ref_jobs, ref_requests), short_cutoff)
    run_table = _run_summary(runs, jobs, requests, decisions)
    policies = _policy_summary(run_table)
    t0 = _policy_summary(run_table.loc[run_table.trial == 0], 't0')
    confirmatory = _policy_summary(_complete_trial_sets(run_table, confirmatory_trials), 'confirmatory')
    pairs = {scope: pd.concat([_paired(run_table, name, bootstrap_samples, bootstrap_seed,
                             sample_scope=scope, confirmatory_trials=confirmatory_trials)
                             for name in baselines], ignore_index=True)
             for scope in ['all_strict', 't0', 'confirmatory']}
    output = root / 'summary'
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [('run_summary', run_table), ('policy_summary', policies),
                        ('policy_summary_t0', t0), ('policy_summary_confirmatory', confirmatory),
                        ('paired', pairs['all_strict']), ('paired_t0', pairs['t0']),
                        ('paired_confirmatory', pairs['confirmatory']),
                        ('jobs', pd.DataFrame(jobs) if jobs else pd.DataFrame(columns=JOB_COLUMNS)),
                        ('requests', pd.DataFrame(requests).drop(columns=['timings'], errors='ignore')
                         if requests else pd.DataFrame(columns=REQUEST_COLUMNS))]:
        table.to_csv(output / f'{name}.csv', index=False)
    lines = ['# Phase 2 replay summary', '',
        'Completion means replay steps completed, not semantic task success.',
        'Fixed output lengths and token-digest agreement do not establish live semantic correctness. '
        'Output match is a diagnostic against matched isolated runs; a missing reference is reported as nan.',
        'Trials are averaged within seed, then seeds receive equal weight. Latency percentiles describe completed jobs only.',
        'Paired ratios require equal planned and completed work, matching resources/inputs, and matching trial sets.',
        f'Bootstrap resamples seed clusters ({bootstrap_samples} draws; RNG seed {bootstrap_seed}); intervals need at least two seeds.',
        'Throughput ratio > 1 favors the policy; JCT ratio < 1 favors the policy.',
        'Slowdown uses only matched isolated prompt/decode timings plus each job\'s actual tool gaps.',
        'Mock-transport runs are marked synthetic_performance and excluded from paired performance claims.',
        'Aborted/incomplete runs retain their planned denominator and any completed records; they cannot establish a paired gain.',
        '`paired.csv` retains strict equality of all observed trial sets as a diagnostic. '
        '`policy_summary.csv` retains all-trial descriptive aggregates and is not the mixed-trial ablation table.',
        f'Short-task cutoff (expected output tokens, frozen on dev): {short_cutoff if short_cutoff is not None else "not provided; no short/long classification"}.', '']
    if policies.empty or not policies.completed.sum():
        lines += ['No completed jobs. No performance gain can be inferred.', '']
    lines += _report_policy_table('T0 ablations', t0)
    lines += ['Only trial 0 is used for every strategy and baseline in this scope. '
              'Additional baseline repeats do not remove T0-only ablations.', '']
    lines += _report_pair_table('paired_t0.csv', pairs['t0'])
    lines += _report_policy_table('Confirmatory repeats', confirmatory)
    lines += [f'This scope requires exactly trials {list(confirmatory_trials)} for both policies '
              'within each seed. Partial repeats are excluded rather than averaged with full repeats.', '']
    lines += _report_pair_table('paired_confirmatory.csv', pairs['confirmatory'])
    lines += [f'Baselines: {", ".join(baselines)}. Complete exclusion diagnostics remain in `paired.csv`.',
              'Battery-powered runs are listed but never paired.',
              'These are finite-stream replay measurements. They do not establish steady-state capacity.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    return policies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--baseline', action='append',
                        help='Label or policy to pair against; repeat for several (default: fifo)')
    parser.add_argument('--short-cutoff', type=float)
    parser.add_argument('--reference-dir', type=Path, action='append', default=[])
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=291)
    parser.add_argument('--confirmatory-trials', default='0,1,2',
                        help='Predeclared exact trial IDs for the repeated-confirmation view (default: 0,1,2)')
    parser.add_argument('--plot', action='store_true', help='also write figures/')
    args = parser.parse_args(argv)
    table = summarize(args.results_dir, baseline=args.baseline or ['fifo'],
                      short_cutoff=args.short_cutoff,
                      confirmatory_trials=tuple(int(trial) for trial in args.confirmatory_trials.split(',')),
                      reference_dirs=args.reference_dir, bootstrap_samples=args.bootstrap_samples,
                      bootstrap_seed=args.bootstrap_seed)
    print(table.to_string(index=False) if not table.empty else 'No completed runs found.')
    if args.plot:
        from phase2.plot import plot
        for path in plot(args.results_dir):
            print(path)


if __name__ == '__main__':
    main()
