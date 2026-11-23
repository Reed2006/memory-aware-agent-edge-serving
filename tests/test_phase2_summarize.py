import json

import numpy as np
import pandas as pd
import pytest

from phase2.summarize import summarize


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def logs(root, policy='fifo', seed=10, trial=0, rate=1., status='ok', **changes):
    common = dict(schema_version=2, label=f'{policy}-{seed}-{trial}', policy=policy,
                  param_hash=policy, workload_hash='work', cost_hash='cost', platform_id='machine',
                  model_hash='model', kv_precision='f16', seed=seed, trial=trial, P=2,
                  ctx_size=24576, kv_unified=False, cache_ram_mib=0, k=2, j=0,
                  gap_mode='fixed:1', arrival='batch', steps_limit=0)
    common.update(changes)
    common['run_id'] = common['label']
    run = dict(common, tasks=1, completed=int(status == 'ok'), failed=int(status != 'ok'),
               makespan_s=60 / rate, jobs_per_minute=rate)
    job = dict(common, job=0, task_id='task', runtime_task_id='job0', status=status,
               latency_s=12., service_s=10., gap_total_s=1., steps=1,
               expected_steps=1, output_tokens=5, expected_output_tokens=5, admission_wait_s=2.)
    req = dict(common, job=0, task_id='task', runtime_task_id='job0', status=status,
               step=0, new_tokens=5, reusable_tokens=0, queue_s=4., input_tokens=10,
               timings=dict(prompt_n=10, cache_n=0, prompt_ms=500., predicted_ms=500.))
    return run, job, req


def dataset(root, records):
    for name, index in [('macro_runs', 0), ('macro_jobs', 1), ('macro_requests', 2)]:
        write(root / f'raw/{name}.jsonl', [record[index] for record in records])


def test_seed_clusters_not_trials_are_independent_samples(tmp_path):
    rows = []
    for policy, rates in [('fifo', [[1., 3.], [1.]]), ('joint', [[2., 6.], [1.]])]:
        for seed, trials in enumerate(rates):
            for trial, rate in enumerate(trials):
                rows.append(logs(tmp_path, policy, seed, trial, rate))
    dataset(tmp_path, rows)
    table = summarize(tmp_path, short_cutoff=5., bootstrap_samples=100, bootstrap_seed=7)
    joint = table.loc[table.policy == 'joint'].iloc[0]
    assert joint.seeds == 2 and joint.runs == 3
    assert joint.jobs_per_min_mean == pytest.approx(2.5)  # mean(4, 1), not mean(2, 6, 1)
    assert joint.jobs_per_min_sd == pytest.approx(np.std([4., 1.], ddof=1))
    paired = pd.read_csv(tmp_path / 'summary/paired.csv')
    assert paired.iloc[0].paired_seeds == 2
    assert paired.iloc[0].throughput_ratio == pytest.approx(1.5)  # mean of seed ratios
    first = (tmp_path / 'summary/paired.csv').read_text()
    summarize(tmp_path, short_cutoff=5., bootstrap_samples=100, bootstrap_seed=7)
    assert (tmp_path / 'summary/paired.csv').read_text() == first


@pytest.mark.parametrize('change,reason', [
    ({'platform_id': 'other'}, 'missing_baseline'),
    ({'workload_hash': 'other'}, 'missing_baseline'),
    ({'ctx_size': 49152}, 'missing_baseline'),
])
def test_incompatible_runs_never_become_paired_speedups(tmp_path, change, reason):
    dataset(tmp_path, [logs(tmp_path), logs(tmp_path, 'joint', rate=2., **change)])
    summarize(tmp_path, bootstrap_samples=10)
    row = pd.read_csv(tmp_path / 'summary/paired.csv').iloc[0]
    assert row.eligible_seeds == 0 and reason in row.excluded_reasons
    assert np.isnan(row.throughput_ratio)


def test_failures_unequal_work_and_unpaired_trials_are_excluded(tmp_path):
    failed = logs(tmp_path, 'joint', seed=10, status='error', rate=2.)
    unequal = logs(tmp_path, 'joint', seed=11, rate=2.)
    unequal[1]['expected_steps'] = 2
    records = [logs(tmp_path, seed=10), failed, logs(tmp_path, seed=11), unequal,
               logs(tmp_path, seed=12), logs(tmp_path, 'joint', seed=12, trial=1)]
    dataset(tmp_path, records)
    summarize(tmp_path, bootstrap_samples=10)
    row = pd.read_csv(tmp_path / 'summary/paired.csv').iloc[0]
    assert row.eligible_seeds == 0
    assert all(word in row.excluded_reasons for word in ['failure', 'work_mismatch', 'trial_set_mismatch'])
    assert np.isnan(row.throughput_ratio)
    run = pd.read_csv(tmp_path / 'summary/run_summary.csv')
    assert run.loc[run.failed > 0, 'jobs_per_min'].iloc[0] == 0.


def test_slowdown_uses_pure_model_timings_with_matching_reference(tmp_path):
    isolated = logs(tmp_path, 'fixed', seed=99, k=1, j=1)
    candidate = logs(tmp_path, 'joint')
    other = logs(tmp_path, 'joint', seed=11, platform_id='other')
    dataset(tmp_path, [isolated, candidate, other])
    summarize(tmp_path, short_cutoff=5., bootstrap_samples=10)
    jobs = pd.read_csv(tmp_path / 'summary/jobs.csv')
    current = jobs.loc[jobs.label == candidate[0]['label']].iloc[0]
    assert current.isolated_compute_s == 1.  # excludes 4 seconds request queue and task service overhead
    assert current.slowdown == 6.  # 12 / (1 + current tool gap 1)
    assert current.task_class == 'short'
    assert jobs.loc[jobs.platform_id == 'other', 'slowdown'].isna().all()
    summarize(tmp_path, bootstrap_samples=10)
    assert pd.read_csv(tmp_path / 'summary/jobs.csv').task_class.isna().all()


def test_empty_all_failed_and_nested_directories_can_be_plotted(tmp_path):
    from phase2.plot import plot
    summarize(tmp_path, bootstrap_samples=10)
    assert 'No completed' in (tmp_path / 'summary/REPORT.md').read_text()
    plot(tmp_path)
    assert (tmp_path / 'figures/throughput.png').exists()
    dataset(tmp_path / 'nested', [logs(tmp_path, status='error')])
    summarize(tmp_path, bootstrap_samples=10)
    plot(tmp_path)
    assert pd.read_csv(tmp_path / 'summary/run_summary.csv').failed.iloc[0] == 1
    assert (tmp_path / 'figures/action_overhead.pdf').exists()


def test_decision_overhead_and_malformed_schema(tmp_path):
    record = logs(tmp_path)
    dataset(tmp_path, [record])
    write(tmp_path / 'raw/policy_decisions.jsonl', [dict(record[0], action='save', status='ok',
          io_s=.25, io_bytes=1024, decision_cpu_s=.005)])
    summarize(tmp_path, bootstrap_samples=10)
    row = pd.read_csv(tmp_path / 'summary/run_summary.csv').iloc[0]
    assert row.decision_cpu_s == .005 and row.io_s == .25 and row.io_bytes == 1024
    record[0]['schema_version'] = 1
    dataset(tmp_path, [record])
    with pytest.raises(ValueError, match='schema'):
        summarize(tmp_path)


def test_mock_results_never_produce_performance_claims_and_io_counts_actual_actions(tmp_path):
    rows = [logs(tmp_path, policy=policy, synthetic_performance=True) for policy in ['fifo', 'joint']]
    dataset(tmp_path, rows)
    write(tmp_path / 'raw/policy_decisions.jsonl', [
        dict(rows[1][0], action='save_decision', io_s=100., io_bytes=10000),
        dict(rows[1][0], action='lease', duration_s=99.),
        dict(rows[1][0], action='save', status='ok', roundtrip_s=.25, bytes=100),
        dict(rows[1][0], action='restore', status='ok', roundtrip_s=.1, bytes=100),
        dict(rows[1][0], action='save', status='http_error', roundtrip_s=20., bytes=400),
    ])
    summarize(tmp_path, bootstrap_samples=10)
    row = pd.read_csv(tmp_path / 'summary/paired.csv').iloc[0]
    assert row.eligible_seeds == 0 and 'synthetic' in row.excluded_reasons
    assert np.isnan(row.throughput_ratio)
    joint = pd.read_csv(tmp_path / 'summary/run_summary.csv').query('policy == "joint"').iloc[0]
    assert joint.io_s == pytest.approx(.35) and joint.io_bytes == 200
    assert 'synthetic' in (tmp_path / 'summary/REPORT.md').read_text()
    for filename in ['paired_t0.csv', 'paired_confirmatory.csv']:
        row = pd.read_csv(tmp_path / 'summary' / filename).iloc[0]
        assert row.eligible_seeds == 0 and 'synthetic' in row.excluded_reasons
        assert np.isnan(row.throughput_ratio)


def test_incomplete_identity_and_mismatched_records_are_rejected(tmp_path):
    row = logs(tmp_path)
    del row[0]['model_hash']
    dataset(tmp_path, [row])
    with pytest.raises(ValueError, match='model_hash'):
        summarize(tmp_path)
    row = logs(tmp_path)
    row[1]['seed'] = 99
    dataset(tmp_path, [row])
    with pytest.raises(ValueError, match='inconsistent seed'):
        summarize(tmp_path)


def test_aborted_runs_retain_denominator_and_completed_records(tmp_path):
    base = logs(tmp_path)
    aborted = logs(tmp_path, 'joint')
    dataset(tmp_path, [base])
    write(tmp_path / 'raw/runs_meta.jsonl', [dict(aborted[0], event='start', planned_tasks=3,
          started_unix=100.), dict(aborted[0], event='aborted', planned_tasks=3,
          done_unix=110., error_type='BackendStateUncertain')])
    write(tmp_path / 'raw/macro_jobs.jsonl', [base[1], aborted[1]])
    write(tmp_path / 'raw/macro_requests.jsonl', [base[2], aborted[2]])
    table = summarize(tmp_path, bootstrap_samples=10)
    joint = table.loc[table.policy == 'joint'].iloc[0]
    assert (joint.submitted, joint.completed, joint.failed) == (3, 1, 2)
    runs = pd.read_csv(tmp_path / 'summary/run_summary.csv')
    assert runs.loc[runs.policy == 'joint', 'run_status'].iloc[0] == 'aborted'
    paired = pd.read_csv(tmp_path / 'summary/paired.csv').iloc[0]
    assert np.isnan(paired.throughput_ratio) and 'aborted' in paired.excluded_reasons
    assert len(pd.read_csv(tmp_path / 'summary/requests.csv')) == 2


@pytest.mark.parametrize('resources', [dict(cache_ram_mib=3072, cache_idle_slots=True),
                                      dict(policy_config={'checkpoint_budget_bytes': 2**30})])
def test_extra_memory_or_snapshot_capacity_never_counts_as_equal_resources(tmp_path, resources):
    dataset(tmp_path, [logs(tmp_path), logs(tmp_path, 'joint', **resources)])
    summarize(tmp_path, bootstrap_samples=10)
    row = pd.read_csv(tmp_path / 'summary/paired.csv').iloc[0]
    assert row.eligible_seeds == 0 and np.isnan(row.throughput_ratio)
    assert 'resource' in row.excluded_reasons
    for filename in ['paired_t0.csv', 'paired_confirmatory.csv']:
        row = pd.read_csv(tmp_path / 'summary' / filename).iloc[0]
        assert row.eligible_seeds == 0 and 'resource' in row.excluded_reasons


def test_paired_reports_baseline_seeds_whose_candidate_never_started(tmp_path):
    dataset(tmp_path, [logs(tmp_path, seed=1), logs(tmp_path, seed=2),
                       logs(tmp_path, 'joint', seed=1)])
    summarize(tmp_path, bootstrap_samples=10)
    row = pd.read_csv(tmp_path / 'summary/paired.csv').iloc[0]
    assert row.eligible_seeds == 1 and row.excluded_seeds == 1
    assert 'missing_candidate' in row.excluded_reasons


@pytest.mark.asyncio
async def test_actual_mock_replay_logs_flow_through_summary(tmp_path):
    import httpx
    from phase2.bench import run_replay
    from phase2.state import PolicyConfig

    workload = tmp_path / 'workload.json'
    workload.write_text(json.dumps({'tasks': [{'task_id': 'a', 'steps': [
        {'delta': [1, 2], 'output': 2, 'gap_s': 0}, {'delta': [3], 'output': 1, 'gap_s': 0}]}]}))
    def handler(request):
        if request.url.path == '/props':
            return httpx.Response(200, json={'total_slots': 2,
                'default_generation_settings': {'n_ctx': 64}})
        if request.url.path == '/completion':
            body = json.loads(request.content)
            return httpx.Response(200, json={'tokens': [7] * body['n_predict'],
                'id_slot': body.get('id_slot', 0), 'timings': {'prompt_n': len(body['prompt']),
                'cache_n': 0, 'prompt_ms': 1., 'predicted_ms': 1.}})
        return httpx.Response(200, json={})
    for name, options in [('fixed', {'concurrency': 1, 'jobs_active': 1}), ('fifo', {}), ('joint', {})]:
        await run_replay(workload, tmp_path / 'data' / name,
            PolicyConfig.for_policy(name, slots=2, ctx_per_slot=64, **options),
            server_metadata=dict(parallel=2, ctx_size=128, ctx_per_slot=64, kv_unified=False,
                                 cache_ram_mib=0, model_hash='mock-model', kv_precision='f16'),
            transport=httpx.MockTransport(handler), sample_resources=False,
            platform_id='mock', gap='zero', arrival='batch')
    table = summarize(tmp_path / 'data', short_cutoff=3., bootstrap_samples=10)
    assert table.completed.sum() == 3
    assert pd.read_csv(tmp_path / 'data/summary/jobs.csv').isolated_compute_s.eq(.004).all()
    assert pd.read_csv(tmp_path / 'data/summary/paired.csv').throughput_ratio.isna().all()


def test_memory_mechanism_counts_and_output_agreement_are_summarized(tmp_path):
    isolated = logs(tmp_path, 'fixed', seed=99, k=1, j=1, policy_label='isolated')
    isolated[1]['output_sha'] = 'same'
    joint = logs(tmp_path, 'joint')
    joint[1]['output_sha'] = 'same'
    other = logs(tmp_path, 'joint', seed=11)
    other[1]['output_sha'] = 'different'
    dataset(tmp_path, [isolated, joint, other])
    write(tmp_path / 'raw/resources.jsonl', [
        dict(label=joint[0]['label'], server_rss=2 * 2**30, system_available=3 * 2**30, swap_out=0),
        dict(label=joint[0]['label'], server_rss=3 * 2**30, system_available=2 * 2**30, swap_out=2**20),
        'partial line'])
    (tmp_path / 'raw/resources.jsonl').write_text(
        (tmp_path / 'raw/resources.jsonl').read_text().replace('"partial line"', '{"label": '))
    write(tmp_path / 'raw/policy_decisions.jsonl', [
        dict(joint[0], action='lease', duration_s=2.),
        dict(joint[0], action='no_lease'),
        dict(joint[0], action='evict'),
        dict(joint[0], action='dispatch', aged=True, ready_wait_s=4., reason='rank'),
        dict(joint[0], action='dispatch', aged=False, ready_wait_s=2., reason='lease_owner')])
    summarize(tmp_path, bootstrap_samples=10)
    runs = pd.read_csv(tmp_path / 'summary/run_summary.csv').set_index('label')
    row = runs.loc[joint[0]['label']]
    assert row.peak_server_rss_gib == 3 and row.min_system_available_gib == 2
    assert row.swap_out_delta_mib == 1
    assert (row.leases, row.no_leases, row.evictions, row.lease_s) == (1, 1, 1, 2.)
    assert (row.aged_dispatches, row.lease_owner_dispatches) == (1, 1)
    assert row.mean_ready_wait_s == 3 and row.max_ready_wait_s == 4
    assert row.output_match_rate == 1 and runs.loc[other[0]['label']].output_match_rate == 0


def test_multiple_label_baselines_and_battery_runs_are_never_paired(tmp_path):
    rows = [logs(tmp_path, 'fifo'), logs(tmp_path, 'fixed', policy_label='best-fixed', param_hash='bf', rate=2.),
            logs(tmp_path, 'joint', rate=4.), logs(tmp_path, 'fixed', seed=99, k=1, j=1,
                                                   policy_label='isolated', param_hash='iso')]
    dataset(tmp_path, rows)
    summarize(tmp_path, baseline=['fifo', 'best-fixed'], bootstrap_samples=10)
    paired = pd.read_csv(tmp_path / 'summary/paired.csv')
    joint = paired.loc[paired.policy_label == 'joint'].set_index('baseline')
    assert joint.loc['fifo'].throughput_ratio == 4 and joint.loc['best-fixed'].throughput_ratio == 2
    assert 'isolated' not in set(paired.policy_label)
    write(tmp_path / 'raw/runs_meta.jsonl', [dict(rows[2][0], event='start',
          machine={'power': "Now drawing from 'Battery Power'"})])
    summarize(tmp_path, bootstrap_samples=10)
    paired = pd.read_csv(tmp_path / 'summary/paired.csv')
    row = paired.loc[paired.policy_label == 'joint'].iloc[0]
    assert np.isnan(row.throughput_ratio) and 'battery' in row.excluded_reasons


def test_mix_workloads_use_source_task_references(tmp_path):
    isolated = logs(tmp_path, 'fixed', seed=99, k=1, j=1)
    mixed = logs(tmp_path, 'joint', workload_hash='mix', reference_workload_hash='work')
    dataset(tmp_path, [isolated, mixed])
    summarize(tmp_path, bootstrap_samples=10)
    jobs = pd.read_csv(tmp_path / 'summary/jobs.csv')
    assert jobs.loc[jobs.workload_hash == 'mix', 'slowdown'].iloc[0] == 6.


def test_t0_ablations_and_confirmatory_trials_are_explicit_separate_samples(tmp_path):
    records = []
    for seed in [1, 2]:
        for trial, rate in enumerate([1., 2., 3.]):
            records += [logs(tmp_path, 'fifo', seed, trial, rate),
                        logs(tmp_path, 'joint', seed, trial, 2 * rate)]
        records.append(logs(tmp_path, 'cache', seed, 0, 1.5))
    dataset(tmp_path, records)
    summarize(tmp_path, bootstrap_samples=10)
    strict = pd.read_csv(tmp_path / 'summary/paired.csv').set_index('policy')
    assert np.isnan(strict.loc['cache', 'throughput_ratio'])
    assert 'trial_set_mismatch' in strict.loc['cache', 'excluded_reasons']
    t0 = pd.read_csv(tmp_path / 'summary/paired_t0.csv').set_index('policy')
    assert t0.loc['cache', 'throughput_ratio'] == 1.5
    assert t0.loc['joint', 'throughput_ratio'] == 2
    assert t0.loc['cache', 'paired_trial_pairs'] == 2
    assert json.loads(t0.loc['cache', 'paired_seed_ids']) == [1, 2]
    assert json.loads(t0.loc['cache', 'matched_trials_by_seed']) == {'1': [0], '2': [0]}
    repeated = pd.read_csv(tmp_path / 'summary/paired_confirmatory.csv').set_index('policy')
    assert repeated.loc['joint', 'throughput_ratio'] == 2
    assert repeated.loc['joint', 'paired_trial_pairs'] == 6
    assert repeated.loc['cache', 'eligible_seeds'] == 0
    assert 'confirmatory_trial_set_mismatch' in repeated.loc['cache', 'excluded_reasons']
    table = pd.read_csv(tmp_path / 'summary/policy_summary_t0.csv').set_index('policy')
    assert table.loc['fifo', 'jobs_per_min_mean'] == 1 and table.loc['cache', 'seeds'] == 2
    table = pd.read_csv(tmp_path / 'summary/policy_summary_confirmatory.csv').set_index('policy')
    assert set(table.index) == {'fifo', 'joint'} and table.loc['fifo', 'runs'] == 6
    report = (tmp_path / 'summary/REPORT.md').read_text()
    assert 'T0 ablations' in report and 'Confirmatory repeats' in report
    assert 'paired_t0.csv' in report and 'paired_confirmatory.csv' in report


def test_confirmatory_never_uses_partial_trial_intersection(tmp_path):
    records = [logs(tmp_path, policy, 1, trial, 1. if policy == 'fifo' else 2.)
               for policy, trials in [('fifo', [0, 1, 2]), ('joint', [0, 1])]
               for trial in trials]
    dataset(tmp_path, records)
    summarize(tmp_path, bootstrap_samples=10)
    row = pd.read_csv(tmp_path / 'summary/paired_confirmatory.csv').iloc[0]
    assert row.eligible_seeds == 0 and np.isnan(row.throughput_ratio)
    assert 'confirmatory_trial_set_mismatch' in row.excluded_reasons
    assert json.loads(row.candidate_trials_by_seed) == {'1': [0, 1]}
    assert json.loads(row.baseline_trials_by_seed) == {'1': [0, 1, 2]}


def test_t0_pairs_include_strong_affinity_baseline_and_keep_resource_gates(tmp_path):
    records = []
    for trial in [0, 1, 2]:
        records.append(logs(tmp_path, 'affinity-fixed', 1, trial, 2.,
                            policy_label='affinity-fixed-best'))
    records += [logs(tmp_path, 'joint', 1, 0, 3.),
                logs(tmp_path, 'ram-cache', 1, 0, 4., cache_ram_mib=3072, cache_idle_slots=True)]
    dataset(tmp_path, records)
    summarize(tmp_path, baseline=['affinity-fixed-best'], bootstrap_samples=10)
    rows = pd.read_csv(tmp_path / 'summary/paired_t0.csv').set_index('policy')
    assert rows.loc['joint', 'throughput_ratio'] == 1.5
    assert 'resource_mismatch' in rows.loc['ram-cache', 'excluded_reasons']
    assert np.isnan(rows.loc['ram-cache', 'throughput_ratio'])


def test_plot_uses_t0_and_complete_repeats_in_distinct_panels(tmp_path):
    from phase2.plot import plot
    records = [logs(tmp_path, policy, 1, trial, 1. if policy == 'fifo' else 2.)
               for policy, trials in [('fifo', [0, 1, 2]), ('joint', [0])]
               for trial in trials]
    dataset(tmp_path, records)
    summarize(tmp_path, bootstrap_samples=10)
    paths = plot(tmp_path)
    assert tmp_path / 'figures/throughput.png' in paths
    assert tmp_path / 'figures/throughput_confirmatory.png' in paths
    t0 = pd.read_csv(tmp_path / 'summary/policy_summary_t0.csv')
    assert set(t0.trial_ids) == {'[0]'} and set(t0.runs) == {1}


def test_plot_never_combines_different_seed_samples(tmp_path):
    from phase2.plot import _panels
    dataset(tmp_path, [logs(tmp_path, 'fifo', 1), logs(tmp_path, 'fifo', 2),
                       logs(tmp_path, 'joint', 1)])
    summarize(tmp_path, bootstrap_samples=10)
    table = pd.read_csv(tmp_path / 'summary/policy_summary_t0.csv')
    panels = _panels(table)
    assert len(panels) == 2
    assert all(len(group.seed_ids.unique()) == len(group.trials_by_seed.unique()) == 1
               for _, group in panels)


def test_t0_missing_baseline_is_not_mislabeled_as_a_resource_mismatch(tmp_path):
    dataset(tmp_path, [logs(tmp_path, 'fifo', 1, 1), logs(tmp_path, 'joint', 1, 0)])
    summarize(tmp_path, bootstrap_samples=10)
    row = pd.read_csv(tmp_path / 'summary/paired_t0.csv').iloc[0]
    assert row.eligible_seeds == 0 and 'missing_baseline_t0' in row.excluded_reasons
    assert 'resource_mismatch' not in row.excluded_reasons


def test_confirmatory_trial_ids_are_predeclared_and_cli_configurable(tmp_path):
    from phase2.summarize import main
    records = [logs(tmp_path, policy, 1, trial, 1. if policy == 'fifo' else 2.)
               for policy in ['fifo', 'joint'] for trial in [0, 1]]
    dataset(tmp_path, records)
    main(['--results-dir', str(tmp_path), '--confirmatory-trials', '0,1', '--bootstrap-samples', '10'])
    row = pd.read_csv(tmp_path / 'summary/paired_confirmatory.csv').iloc[0]
    assert row.paired_trial_pairs == 2 and row.throughput_ratio == 2
    assert json.loads(row.required_trials) == [0, 1]
    for invalid in [(0,), (0, 0), (-1, 0), (0, '1')]:
        with pytest.raises(ValueError, match='confirmatory_trials'):
            summarize(tmp_path, confirmatory_trials=invalid)


def test_deferred_attempts_and_repricing_are_not_counted_twice(tmp_path):
    joint = logs(tmp_path, 'joint-checkpoint')
    dataset(tmp_path, [joint])
    write(tmp_path / 'raw/policy_decisions.jsonl', [
        dict(joint[0], action='no_lease'),
        dict(joint[0], action='dispatch', sequence=4, aged=False, ready_wait_s=1., reason='rank'),
        dict(joint[0], action='no_lease', reconsidered=True),  # same version, priced again
        dict(joint[0], action='lease', duration_s=3., reconsidered=True),
        dict(joint[0], action='dispatch_deferred', sequence=4, reason='checkpoint_alternative_reconsidered'),
        dict(joint[0], action='dispatch', sequence=4, aged=True, ready_wait_s=5., reason='rank'),
        dict(joint[0], action='dispatch', aged=False, ready_wait_s=3., reason='rank')])  # pilot row
    summarize(tmp_path, bootstrap_samples=10)
    row = pd.read_csv(tmp_path / 'summary/run_summary.csv').iloc[0]
    assert (row.no_leases, row.leases, row.lease_s) == (1, 1, 3.)  # the held lease still counts
    assert (row.reconsidered_decisions, row.deferred_dispatches) == (2, 1)
    assert row.aged_dispatches == 1 and row.mean_ready_wait_s == 4 and row.max_ready_wait_s == 5
