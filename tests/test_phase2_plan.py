import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from phase2.plan import create_manifest, execute_manifest, main


def inputs(tmp_path):
    workload = tmp_path / 'workload.json'
    workload.write_text(json.dumps({'tasks': [
        {'task_id': 'a', 'steps': [{'delta': [1, 2], 'output': 1, 'gap_s': 0}]},
        {'task_id': 'b', 'steps': [{'delta': [3, 4], 'output': 50, 'gap_s': 0}]}]}))
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'schema': 'phase2-cost-profile-v1',
                                 'prefill': {'1': {'seconds_per_token': .001, 'intercept_s': 0}}}))
    return dict(workload=workload, cost_profile=profile, results_dir=tmp_path / 'runs',
                platform_id='mock', seeds=[1, 2], trials=2, num_jobs=10)


def frozen(tmp_path, **changes):
    data = dict(schema='phase2-frozen-config-v1', frozen=True, selected_on='dev',
                short_cutoff=20., policies=[dict(label='fixed-best', policy='fixed',
                config={'concurrency': 2, 'jobs_active': 4}),
                dict(label='joint', policy='joint', config={})])
    data.update(changes)
    path = tmp_path / 'frozen.json'
    path.write_text(json.dumps(data))
    return path


def test_dev_matrix_has_six_fixed_candidates_ablations_and_rotating_order(tmp_path):
    manifest = create_manifest(phase='dev', **inputs(tmp_path))
    fixed = {(r['config']['concurrency'], r['config']['jobs_active']) for r in manifest['runs']
             if r['policy'] == 'fixed'}
    # (2, inf) is the FIFO row itself; it is not run twice.
    assert fixed == {(1, 1), (1, 0), (2, 2), (2, 3), (2, 4)}
    assert {'fifo', 'affinity', 'affinity-fixed', 'ttl', 'queue', 'cache', 'joint',
            'joint-checkpoint', 'ram-cache'} <= {run['policy'] for run in manifest['runs']}
    ttls = {r['config']['ttl_s'] for r in manifest['runs'] if r['policy'] == 'ttl'}
    assert ttls == {10., 15., 30.}  # bounded grid includes the 15-second tool-wait boundary
    assert len({r['label'] for r in manifest['runs']}) == len({r['param_hash'] for r in manifest['runs']})
    first = [r['label'] for r in manifest['runs'] if r['seed'] == 1 and r['trial'] == 0]
    second = [r['label'] for r in manifest['runs'] if r['seed'] == 1 and r['trial'] == 1]
    assert second == first[1:] + first[:1]
    assert len({r['results_dir'] for r in manifest['runs']}) == len(manifest['runs'])
    for run in manifest['runs']:
        assert run['command'][1:3] == ['-m', 'phase2.bench']
        assert '--trial-start' in run['command']
        assert '--trial-order-start' in run['command']
        assert run['server_requirements']['cache_ram_mib'] == (3072 if run['policy'] == 'ram-cache' else 0)


def test_eval_requires_actual_frozen_dev_config_and_pins_hashes(tmp_path):
    kwargs = inputs(tmp_path)
    with pytest.raises(ValueError, match='frozen'):
        create_manifest(phase='eval', **kwargs)
    with pytest.raises(ValueError, match='dev'):
        create_manifest(phase='eval', frozen_config=frozen(tmp_path, selected_on='eval'), **kwargs)
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path), **kwargs)
    assert manifest['frozen_config']['sha256'] and manifest['cost_profile']['sha256']
    assert manifest['short_cutoff'] == 20.
    assert {r['policy'] for r in manifest['runs']} == {'fixed', 'joint'}


def test_standalone_defaults_use_fresh_confirmation_seeds(tmp_path):
    kwargs = inputs(tmp_path)
    del kwargs['seeds']
    config = frozen(tmp_path)
    evaluation = create_manifest(phase='eval', frozen_config=config, **kwargs)
    sensitivity = create_manifest(phase='sensitivity', frozen_config=config, **kwargs)
    assert evaluation['seeds'] == [1101, 1102, 1103]
    assert sensitivity['seeds'] == [2101, 2102]


def test_dry_run_never_executes_and_hash_mutation_blocks_execution(tmp_path):
    kwargs = inputs(tmp_path)
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path), **kwargs)
    called = []
    result = execute_manifest(manifest, runner=lambda *args, **kw: called.append(args))
    assert not called and result['dry_run']
    kwargs['cost_profile'].write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        execute_manifest(manifest, execute=True, runner=lambda *a, **kw: called.append(a))
    assert not called


def test_explicit_run_checks_resource_group_and_does_not_use_shell(tmp_path):
    kwargs = inputs(tmp_path)
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path), **kwargs)
    server = tmp_path / 'server.json'
    server.write_text(json.dumps(dict(parallel=2, ctx_size=24576, kv_unified=False,
                                     cache_ram_mib=0, cache_idle_slots=False)))
    calls = []
    def runner(command, **options):
        calls.append((command, options))
        return CompletedProcess(command, 0)
    result = execute_manifest(manifest, execute=True, server_record=server, runner=runner)
    assert result['completed'] == len(manifest['runs'])
    assert all(options.get('shell', False) is False for _, options in calls)
    assert all('timeout' not in options for _, options in calls)  # budget never kills a backend client
    server.write_text(json.dumps(dict(parallel=4, ctx_size=24576, kv_unified=False,
                                     cache_ram_mib=0, cache_idle_slots=False)))
    with pytest.raises(ValueError, match='server'):
        execute_manifest(manifest, execute=True, server_record=server, runner=runner)


def test_budget_stop_failure_and_skip_done(tmp_path):
    kwargs = inputs(tmp_path)
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path), **kwargs)
    assert execute_manifest(manifest, execute=True, budget_s=0)['stopped'] == 'budget'
    server = tmp_path / 'server.json'
    server.write_text(json.dumps(dict(parallel=2, ctx_size=24576, kv_unified=False,
                                     cache_ram_mib=0, cache_idle_slots=False)))
    result = execute_manifest(manifest, execute=True, server_record=server,
                              runner=lambda command, **kw: CompletedProcess(command, 2))
    assert result['failed'] == 1 and result['completed'] == 0
    assert '--skip-done' in manifest['runs'][0]['command']


def test_cli_creates_only_manifest_and_sensitivity_has_no_cartesian_explosion(tmp_path, capsys):
    kwargs = inputs(tmp_path)
    path = tmp_path / 'plan.json'
    main(['create', '--phase', 'dev', '--workload', str(kwargs['workload']), '--cost-profile',
          str(kwargs['cost_profile']), '--results-dir', str(kwargs['results_dir']), '--platform-id',
          'mock', '--manifest', str(path), '--seeds', '1', '--trials', '1'])
    assert path.exists() and not kwargs['results_dir'].exists()
    assert 'phase2.bench' in capsys.readouterr().out
    main(['show', str(path)])
    main(['run', str(path)])
    assert not kwargs['results_dir'].exists()
    sensitivity = create_manifest(phase='sensitivity', frozen_config=frozen(tmp_path), **kwargs)
    assert {'main', 'arrival-half', 'arrival-double', 'gap-zero', 'gap-recorded', 'gap-long',
            'mix-short', 'mix-long'} == {run['condition'] for run in sensitivity['runs']}
    assert all(Path(run['results_dir']).is_absolute() for run in sensitivity['runs'])


def test_bench_parser_accepts_every_generated_command(tmp_path):
    from phase2.bench import parser
    manifest = create_manifest(phase='dev', **inputs(tmp_path))
    for run in manifest['runs']:
        args = parser().parse_args(run['command'][3:])
        assert args.trial_start == run['trial'] and args.trial_order_start == run['trial_order']
        assert args.request_timeout_s == manifest['request_timeout_s']


def test_invalid_profile_timeouts_and_frozen_policy_resources_are_rejected(tmp_path):
    kwargs = inputs(tmp_path)
    with pytest.raises(ValueError, match='timeout'):
        create_manifest(phase='dev', request_timeout_s=0., **kwargs)
    with pytest.raises(ValueError, match='slot'):
        create_manifest(phase='eval', frozen_config=frozen(tmp_path, policies=[
            dict(label='joint', policy='joint', config={'slots': 4})]), **kwargs)
    kwargs['cost_profile'].write_text(json.dumps({'schema': 'phase2-cost-profile-v1',
                                                'prefill': {'1': {'seconds_per_token': -1}}}))
    with pytest.raises(ValueError):
        create_manifest(phase='dev', **kwargs)


def test_soft_budget_stops_launching_without_killing_the_current_run(tmp_path, monkeypatch):
    import phase2.plan as plan
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path), **inputs(tmp_path))
    server = tmp_path / 'server.json'
    server.write_text(json.dumps(dict(parallel=2, ctx_size=24576, kv_unified=False,
                                     cache_ram_mib=0, cache_idle_slots=False)))
    ticks = iter([0., 1., 3.])
    monkeypatch.setattr(plan.time, 'monotonic', lambda: next(ticks))
    calls = []
    def runner(command, **options):
        calls.append(options)
        return CompletedProcess(command, 0)
    result = execute_manifest(manifest, execute=True, budget_s=2., server_record=server, runner=runner)
    assert len(calls) == result['completed'] == 1 and result['stopped'] == 'budget'
    assert 'timeout' not in calls[0]


def test_policy_order_rotates_by_seed_as_well_as_trial(tmp_path):
    manifest = create_manifest(phase='dev', **dict(inputs(tmp_path), trials=1))
    first = [r['label'] for r in manifest['runs'] if r['seed'] == 1]
    second = [r['label'] for r in manifest['runs'] if r['seed'] == 2]
    assert second == first[1:] + first[:1]


def test_aging_grid_is_dev_only_and_creates_distinct_joint_candidates(tmp_path):
    kwargs = inputs(tmp_path)
    manifest = create_manifest(phase='dev', aging_grid=[20, 40.5], **kwargs)
    joint = {r['label']: r['config']['aging_s'] for r in manifest['runs'] if r['policy'] == 'joint'}
    assert joint == {'joint-a20': 20., 'joint-a40p5': 40.5}
    for name in ['queue', 'cache', 'joint-checkpoint']:
        variants = {r['label']: r['config']['aging_s'] for r in manifest['runs'] if r['policy'] == name}
        assert variants == {f'{name}-a20': 20., f'{name}-a40p5': 40.5}
    assert {(r['config']['aging_s'], r['config']['ttl_s']) for r in manifest['runs']
            if r['policy'] == 'ttl'} == {(a, ttl) for a in (20., 40.5) for ttl in (10., 15., 30.)}
    with pytest.raises(ValueError, match='dev'):
        create_manifest(phase='eval', frozen_config=frozen(tmp_path), aging_grid=[20], **kwargs)
    with pytest.raises(ValueError, match='aging'):
        create_manifest(phase='dev', aging_grid=[0], **kwargs)


def test_eval_adds_one_isolated_reference_run_first(tmp_path):
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path), **inputs(tmp_path))
    isolated = [r for r in manifest['runs'] if r['label'] == 'isolated']
    assert len(isolated) == 1 and manifest['runs'][0]['label'] == 'isolated'
    assert isolated[0]['config']['concurrency'] == isolated[0]['config']['jobs_active'] == 1
    assert isolated[0]['seed'] == 1 and isolated[0]['trial'] == 0
    without = create_manifest(phase='eval', frozen_config=frozen(tmp_path), isolated=False,
                              **inputs(tmp_path))
    assert not any(r['label'] == 'isolated' for r in without['runs'])


def test_full_frozen_configuration_keeps_measured_hash_and_rejects_tampering(tmp_path):
    from phase2.state import PolicyConfig
    config = PolicyConfig.for_policy('joint-checkpoint', aging_s=80., lease_max_s=37.)
    policy = dict(label='joint-checkpoint', policy='joint-checkpoint', config=config.to_dict(),
                  param_hash=config.param_hash, evidence={'param_hash': config.param_hash})
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path, policies=[policy]), **inputs(tmp_path))
    run = next(r for r in manifest['runs'] if r['label'] == 'joint-checkpoint')
    assert run['config'] == config.to_dict() and run['param_hash'] == config.param_hash
    policy['config']['aging_s'] = 90.
    with pytest.raises(ValueError, match='hash|evidence'):
        create_manifest(phase='eval', frozen_config=frozen(tmp_path, policies=[policy]), **inputs(tmp_path))


def test_dev_evidence_cannot_be_reused_with_a_different_cost_profile(tmp_path):
    kwargs = inputs(tmp_path)
    config = frozen(tmp_path, evidence={'comparison_identity': {'cost_hash': 'other-profile'}})
    with pytest.raises(ValueError, match='cost profile'):
        create_manifest(phase='eval', frozen_config=config, **kwargs)


def test_trial_labels_repeat_only_key_rows(tmp_path):
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path),
                               trial_labels=['joint'], **inputs(tmp_path))
    counts = {}
    for run in manifest['runs']:
        counts.setdefault(run['label'], set()).add((run['seed'], run['trial']))
    assert len(counts['joint']) == 4 and len(counts['fixed-best']) == 2
    with pytest.raises(ValueError, match='trial labels'):
        create_manifest(phase='eval', frozen_config=frozen(tmp_path), trial_labels=['nope'],
                        **inputs(tmp_path))


def test_mixes_reweight_source_tasks_and_keep_reference_identity(tmp_path):
    from phase2.plan import make_mix
    kwargs = inputs(tmp_path)
    manifest = create_manifest(phase='sensitivity', frozen_config=frozen(tmp_path),
                               conditions=['mix-short', 'mix-long'], only_labels=['joint'], **kwargs)
    assert {r['label'] for r in manifest['runs']} == {'joint', 'isolated'}
    for kind, favoured in [('short', 'a'), ('long', 'b')]:
        run = next(r for r in manifest['runs'] if r['condition'] == f'mix-{kind}')
        data = json.loads(Path(run['workload']['path']).read_text())
        ids = [task['task_id'] for task in data['tasks']]
        assert ids.count(favoured) == 3 and len(ids) == 4
        assert data['source_workload_hash'] == manifest['workload']['sha256']
        assert run['command'][run['command'].index('--workload') + 1] == run['workload']['path']
    with pytest.raises(ValueError, match='both sides'):
        make_mix({'tasks': [{'task_id': 'a', 'steps': [{'output': 1}]}]}, 'h', 20, 'short')
    with pytest.raises(ValueError, match='unknown conditions'):
        create_manifest(phase='sensitivity', frozen_config=frozen(tmp_path), conditions=['x'], **kwargs)


def test_host_cache_runs_require_an_explicit_fresh_backend(tmp_path):
    manifest = create_manifest(phase='dev', **dict(inputs(tmp_path), trials=1))
    server = tmp_path / 'server.json'
    server.write_text(json.dumps(dict(parallel=2, ctx_size=24576, kv_unified=False,
                                      cache_ram_mib=3072, cache_idle_slots=True)))
    runner = lambda command, **kw: CompletedProcess(command, 0)
    with pytest.raises(ValueError, match='restart-server'):
        execute_manifest(manifest, execute=True, server_record=server, runner=runner,
                         server_group='P2-C24576-RAM3072-idle1')
    restarts = []
    result = execute_manifest(manifest, execute=True, server_record=server, runner=runner,
                              server_group='P2-C24576-RAM3072-idle1', restart_server='ram-cache',
                              restarter=restarts.append)
    assert result['completed'] == 2 and restarts == [server, server]
    labelled = execute_manifest(manifest, labels=['joint'])
    assert labelled['planned'] == 2


def _finish(run, manifest):
    raw = Path(run['results_dir']) / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    (raw / 'macro_runs.jsonl').write_text(json.dumps(dict(
        schema_version=2, tasks=manifest['num_jobs'], completed=manifest['num_jobs'], failed=0,
        makespan_s=10., jobs_per_minute=manifest['num_jobs'] * 6,
        param_hash=run['param_hash'], workload_hash=run['workload']['sha256'],
        cost_hash=manifest['cost_profile']['sha256'], seed=run['seed'], trial=run['trial'],
        gap_mode=run['gap'], arrival=run['arrival'])) + '\n')


def test_resume_skips_finished_runs_and_honours_stop_file_and_deadline(tmp_path):
    from phase2.plan import run_done
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path), **inputs(tmp_path))
    server = tmp_path / 'server.json'
    server.write_text(json.dumps(dict(parallel=2, ctx_size=24576, kv_unified=False,
                                      cache_ram_mib=0, cache_idle_slots=False)))
    first = manifest['runs'][0]
    _finish(first, manifest)
    assert run_done(first, manifest)
    other = dict(first, param_hash='changed')
    assert not run_done(other, manifest)  # a different identity in the same directory is not done
    calls = []
    def runner(command, **options):
        calls.append(command[command.index('--policy-label') + 1])
        return CompletedProcess(command, 0)
    stop = tmp_path / 'STOP'
    stop.write_text('')
    result = execute_manifest(manifest, execute=True, server_record=server, runner=runner, stop_file=stop)
    assert result['skipped'] == 1 and result['stopped'] == 'stop_file' and not calls
    stop.unlink()
    ticks = iter([0., 0., 100., 100.])
    result = execute_manifest(manifest, execute=True, server_record=server, runner=runner,
                              deadline_unix=150., run_estimate_s=10., clock=lambda: next(ticks))
    # The first launched run took 100 s; the next would end after the deadline.
    assert result['completed'] == 1 and result['stopped'] == 'deadline' and len(calls) == 1


def test_aborted_metadata_and_partial_jobs_do_not_mark_a_run_done(tmp_path):
    from phase2.plan import run_done
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path), **inputs(tmp_path))
    run = manifest['runs'][0]
    raw = Path(run['results_dir']) / 'raw'
    raw.mkdir(parents=True)
    common = dict(schema_version=2, tasks=manifest['num_jobs'], num_jobs=manifest['num_jobs'],
                  param_hash=run['param_hash'], workload_hash=run['workload']['sha256'],
                  cost_hash=manifest['cost_profile']['sha256'], seed=run['seed'], trial=run['trial'],
                  gap_mode=run['gap'], arrival=run['arrival'])
    (raw / 'runs_meta.jsonl').write_text(json.dumps(dict(common, event='aborted',
        error_type='BackendStateUncertain', backend_state_uncertain=True, done_unix=123.)) + '\n')
    (raw / 'macro_jobs.jsonl').write_text(json.dumps(dict(common, job=0, status='error',
        error_type='BackendStateUncertain', steps=1, expected_steps=2)) + '\n')
    assert not run_done(run, manifest)


def test_completed_terminal_with_failed_jobs_stays_done_without_filtering_failures(tmp_path):
    from phase2.plan import run_done
    manifest = create_manifest(phase='eval', frozen_config=frozen(tmp_path), **inputs(tmp_path))
    run = manifest['runs'][0]
    _finish(run, manifest)
    path = Path(run['results_dir']) / 'raw/macro_runs.jsonl'
    record = json.loads(path.read_text())
    record.update(completed=manifest['num_jobs'] - 1, failed=1)
    path.write_text(json.dumps(record) + '\n')
    assert run_done(run, manifest)
