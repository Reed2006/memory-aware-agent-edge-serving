import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import phase2.campaign as campaign
from phase2.plan import main as plan_main


EXPECTED_MODEL = campaign.MODEL_SHA.read_text().split()[0]


def environment(**changes):
    value = dict(llama_server=f'version: 0.7.0 (build 11669, commit {campaign.PINNED_COMMIT})',
                 model_sha256=EXPECTED_MODEL, python='3.13', packages_sha256='p',
                 workloads={'dev': 'd'}, sources={'phase2/scheduler.py': 'a'})
    value.update(changes)
    return value


class Fake:
    """Simulates every child command; plan create runs for real (it is model-free)."""

    def __init__(self, root):
        self.root = Path(root)
        self.calls = []
        self.limit = None  # finish at most this many runs per plan invocation
        self.fail = {}
        self.during = {}

    def __call__(self, command):
        # The real runner logs with shlex.join and passes argv to Popen: strings only.
        assert all(isinstance(a, (str, Path)) for a in command), command
        argv = [str(a) for a in command]
        module = argv[2]
        action = argv[3] if len(argv) > 3 else ''
        name = module + ':' + (action if action in ('stop', 'calibrate', 'create', 'run') else
                               (argv[argv.index('--suite') + 1] if '--suite' in argv else ''))
        self.calls.append(name)
        if name in self.during:
            self.during.pop(name)()
        if self.fail.get(name):
            self.fail[name] -= 1
            return 1
        value = lambda flag: Path(argv[argv.index(flag) + 1])
        if module == 'phase2.probes' and action == 'calibrate':
            value('--output').parent.mkdir(parents=True, exist_ok=True)
            value('--output').write_text(json.dumps({'schema': 'phase2-cost-profile-v1',
                'prefill': {'2': {'seconds_per_token': .006}}}))
        elif module == 'phase2.probes':
            raw = value('--results-dir') / 'raw'
            raw.mkdir(parents=True, exist_ok=True)
            (raw / 'probes.jsonl').write_text(json.dumps({'status': 'ok'}) + '\n')
        elif module == 'phase2.bench':
            raw = value('--results-dir') / 'raw'
            raw.mkdir(parents=True, exist_ok=True)
            (raw / 'policy_decisions.jsonl').write_text('')
        elif module == 'phase2.plan' and action == 'create':
            plan_main(argv[3:])
        elif module == 'phase2.plan' and action == 'run':
            manifest = json.loads(Path(argv[4]).read_text())
            group = argv[argv.index('--server-group') + 1]
            finished = 0
            for run in manifest['runs']:
                if run['server_group'] != group or (self.limit is not None and finished >= self.limit):
                    continue
                raw = Path(run['results_dir']) / 'raw'
                raw.mkdir(parents=True, exist_ok=True)
                if not (raw / 'macro_runs.jsonl').exists():
                    finished += 1
                (raw / 'macro_runs.jsonl').write_text(json.dumps(dict(
                    param_hash=run['param_hash'], workload_hash=run['workload']['sha256'],
                    cost_hash=manifest['cost_profile']['sha256'], seed=run['seed'],
                    trial=run['trial'], gap_mode=run['gap'], arrival=run['arrival'])) + '\n')
        elif module == 'phase2.freeze':
            value('--output').write_text(json.dumps(dict(
                schema='phase2-frozen-config-v1', frozen=True, selected_on='dev', short_cutoff=782.5,
                notes=['test'], policies=[
                    dict(label='fifo', policy='fifo', config={}),
                    dict(label='best-fixed', policy='fixed', config={'concurrency': 2, 'jobs_active': 2}),
                    dict(label='affinity', policy='affinity', config={}),
                    dict(label='joint', policy='joint', config={'aging_s': 40.}),
                    dict(label='ram-cache', policy='ram-cache', config={},
                         resources={'cache_ram_mib': 3072, 'cache_idle_slots': True})])))
        return 0


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'POINTER', tmp_path / 'pointer')
    monkeypatch.setattr(campaign.psutil, 'virtual_memory', lambda: SimpleNamespace(available=8 * 2**30))
    monkeypatch.setattr(campaign.psutil, 'process_iter', lambda *a, **k: [])
    root = tmp_path / 'campaign-root'
    fake = Fake(root)
    state = dict(env=environment(), power="Now drawing from 'AC Power'",
                 shared=dict(aging_s=40., queue_rule='continuation'))
    monkeypatch.setattr(campaign, 'select_shared', lambda root: dict(state['shared']), raising=False)
    make = lambda: campaign.Campaign(root, runner=fake, fingerprinter=lambda: state['env'],
                                     power=lambda: state['power'])
    make().init('m4', 'm4-test')
    return SimpleNamespace(root=root, fake=fake, state=state, make=make)


def sessions(root):
    return [json.loads(line) for line in (root / 'campaign/sessions.jsonl').read_text().splitlines()]


def test_campaign_resumes_across_nights_and_waits_for_freeze_review(setup):
    setup.fake.limit = 5
    reason = setup.make().run()
    assert reason == 'deadline'  # dev runs stopped part way, as when a deadline is reached
    first = list(setup.fake.calls)
    assert first.count('phase2.probes:affinity') == 1 and first[-1] == 'phase2.server:stop'
    counts = setup.make().progress(next(s for s in setup.make().stages() if s.name == 'dev-runs'))
    assert counts == (5, 36)

    setup.fake.limit = None
    setup.fake.calls.clear()
    reason = setup.make().run()
    assert reason.startswith('waiting for review')
    assert 'phase2.probes:affinity' not in setup.fake.calls  # finished stages never rerun
    assert 'phase2.probes:ram-cache' in setup.fake.calls
    assert (setup.root / 'manifests/frozen.json').exists()

    setup.make().approve()
    assert setup.make().run() == 'complete'
    trials = json.loads((setup.root / 'manifests/eval-trials.json').read_text())
    assert {run['label'] for run in trials['runs']} == {'fifo', 'best-fixed', 'joint', 'isolated'}
    sensitivity = json.loads((setup.root / 'manifests/sensitivity.json').read_text())
    assert 'isolated' not in {run['label'] for run in sensitivity['runs']}
    assert [s['reason'] for s in sessions(setup.root)][-1] == 'complete'
    status = setup.make().status()
    assert 'sensitivity-runs' in status and 'pending' not in status
    # Changing the frozen config after approval re-closes the gate.
    frozen = setup.root / 'manifests/frozen.json'
    frozen.write_text(frozen.read_text().replace('40.0', '80.0'))
    gate = next(s for s in setup.make().stages() if s.kind == 'gate')
    assert not setup.make().done(gate)


def test_environment_change_refuses_until_relocked(setup):
    setup.state['env'] = environment(sources={'phase2/scheduler.py': 'b'})
    with pytest.raises(SystemExit, match='code changed: phase2/scheduler.py'):
        setup.make().run()
    setup.make().relock('fixed a scheduler bug before dev')
    history = (setup.root / 'campaign/lock-history.jsonl').read_text()
    assert 'scheduler' in history and 'fixed a scheduler bug' in history
    setup.state['power'] = "Now drawing from 'Battery Power'"
    with pytest.raises(SystemExit, match='battery'):
        setup.make().run()


def test_unplugging_aborts_the_current_stage_and_stops_the_backend(setup):
    session = setup.make()
    setup.fake.during['phase2.probes:incremental-recompute'] = lambda: session.interrupt('charger unplugged')
    assert session.run() == 'charger unplugged'
    assert setup.fake.calls[-1] == 'phase2.server:stop'
    assert not (setup.root / 'campaign/done/probe-incremental-recompute').exists()
    assert not (setup.root / 'campaign/campaign.pid').exists()


def test_deadline_prevents_starting_work_that_cannot_finish(setup):
    session = setup.make()
    reason = session.run(until_unix=session.clock() + 60)
    assert reason.startswith('deadline') and 'probe-affinity' in reason
    assert not any(call.startswith('phase2.probes') for call in setup.fake.calls)


def test_failed_stage_is_retried_once_then_session_stops(setup):
    setup.fake.fail['phase2.bench:'] = 2
    reason = setup.make().run()
    assert reason == 'smoke failed twice; see campaign.log'
    assert setup.fake.calls.count('phase2.bench:') == 2


def test_partial_probe_logs_are_set_aside_and_stale_snapshots_removed(setup):
    partial = setup.root / 'probes/affinity/raw'
    partial.mkdir(parents=True)
    (partial / 'probes.jsonl').write_text('{"status": "ok"}\n')
    cache = setup.root / 'backend/cache'
    cache.mkdir(parents=True)
    for name in ['phase2-abc.bin', 'probe-x.bin', 'unrelated.bin']:
        (cache / name).write_bytes(b'x')
    setup.fake.fail['phase2.bench:'] = 2
    setup.make().run()
    assert list((setup.root / 'probes/affinity').glob('raw-interrupted-*'))
    assert sorted(path.name for path in cache.iterdir()) == ['unrelated.bin']


def test_until_means_the_next_occurrence_of_that_time():
    night = dt.datetime(2026, 11, 17, 23, 0)
    assert dt.datetime.fromtimestamp(campaign.parse_until('07:30', None, night)) == dt.datetime(2026, 11, 18, 7, 30)
    assert dt.datetime.fromtimestamp(campaign.parse_until(None, 2, night)) == dt.datetime(2026, 11, 18, 1, 0)
    assert campaign.parse_until(None, None, night) is None


def test_dev_runs_command_is_all_strings_with_a_deadline(setup):
    setup.fake.limit = 1
    reason = setup.make().run(until_unix=setup.make().clock() + 10 * 3600)
    assert reason == 'deadline'
    assert 'phase2.plan:run' in setup.fake.calls


def test_internal_error_is_never_recorded_as_complete(setup):
    def broken(command):
        raise TypeError('boom')
    session = campaign.Campaign(setup.root, runner=broken, fingerprinter=lambda: setup.state['env'],
                                power=lambda: setup.state['power'])
    with pytest.raises(TypeError):
        session.run()
    assert sessions(setup.root)[-1]['reason'].startswith('internal error: TypeError')


def test_each_session_warms_up_once_before_its_first_measured_run(setup):
    setup.fake.limit = 3
    assert setup.make().run() == 'deadline'
    calls = setup.fake.calls
    first_run = calls.index('phase2.plan:run')
    assert calls[first_run - 1] == 'phase2.bench:'  # the unmeasured warm-up
    assert calls.count('phase2.bench:') == 3  # smoke, collect-waits, warm-up
    warmups = list((setup.root / 'campaign/warmup').iterdir())
    assert len(warmups) == 1 and (warmups[0] / 'raw').exists()
    setup.fake.calls.clear()
    setup.make().run()  # resumes dev: warms up again, once, before measuring
    assert setup.fake.calls.count('phase2.bench:') == 1
    assert setup.fake.calls.index('phase2.bench:') < setup.fake.calls.index('phase2.plan:run')


def test_trials_and_sensitivity_follow_the_frozen_final_system(setup):
    assert setup.make().run().startswith('waiting for review')
    frozen_path = setup.root / 'manifests/frozen.json'
    frozen = json.loads(frozen_path.read_text())
    frozen['policies'] += [dict(label='fixed-ttl', policy='ttl', config={'ttl_s': 10.}),
                           dict(label='joint-checkpoint', policy='joint-checkpoint', config={})]
    frozen['final_label'] = 'joint-checkpoint'
    frozen_path.write_text(json.dumps(frozen))
    setup.make().approve()
    assert setup.make().run() == 'complete'
    trials = json.loads((setup.root / 'manifests/eval-trials.json').read_text())
    assert {run['label'] for run in trials['runs']} == {'fifo', 'best-fixed', 'joint-checkpoint', 'isolated'}
    sensitivity = json.loads((setup.root / 'manifests/sensitivity.json').read_text())
    assert {run['label'] for run in sensitivity['runs']} == {'fifo', 'best-fixed', 'joint-checkpoint',
                                                             'fixed-ttl'}


def test_new_campaign_uses_a_new_revision_and_unseen_eval_seeds(setup):
    config = setup.make().config
    assert config['experiment_revision'] == 2
    assert config['seeds']['eval'] == '1101,1102,1103'
    assert config['seeds']['sensitivity'] == '2101,2102'
    assert config['sensitivity_conditions'] == 'gap-recorded,gap-long'


def test_dev_screens_joint_then_confirms_only_its_measured_aging(setup):
    setup.state['shared']['aging_s'] = 80.
    assert setup.make().run().startswith('waiting for review')
    screen = json.loads((setup.root / 'manifests/dev.json').read_text())
    confirm = json.loads((setup.root / 'manifests/dev-confirm.json').read_text())
    assert len(screen['runs']) == 39
    assert {r['label'] for r in screen['runs'] if r['policy'] == 'joint'} == {
        'joint-a20', 'joint-a40', 'joint-a80'}
    assert not any(r['policy'] in {'queue', 'cache', 'joint-checkpoint', 'ttl'} for r in screen['runs'])
    assert len(confirm['runs']) == 18
    assert {r['label'] for r in confirm['runs']} == {
        'queue-a80', 'cache-a80', 'joint-checkpoint-a80',
        'ttl-10-a80', 'ttl-15-a80', 'ttl-30-a80'}
    assert {r['config']['aging_s'] for r in confirm['runs']} == {80.}
    assert confirm['seeds'] == screen['seeds']
    assert confirm['cost_profile'] == screen['cost_profile']
    stages = [stage.name for stage in setup.make().stages()]
    assert stages.index('dev-screen-summary') < stages.index('dev-confirm-manifest')
    assert stages.index('dev-confirm-runs') < stages.index('dev-summary') < stages.index('freeze')


def test_strong_affinity_baseline_gets_repeats_sensitivity_and_summary(setup):
    assert setup.make().run().startswith('waiting for review')
    path = setup.root / 'manifests/frozen.json'
    frozen = json.loads(path.read_text())
    frozen['policies'].append(dict(label='affinity-fixed-best', policy='affinity-fixed',
                                   config={'concurrency': 2, 'jobs_active': 2}))
    path.write_text(json.dumps(frozen))
    stages = {stage.name: stage for stage in setup.make().stages()}
    setup.make().approve()
    assert setup.make().run() == 'complete'
    trials = json.loads((setup.root / 'manifests/eval-trials.json').read_text())
    b2f = [r for r in trials['runs'] if r['label'] == 'affinity-fixed-best']
    assert len(b2f) == 6 and {r['trial'] for r in b2f} == {0, 1}  # one added trial per seed
    sensitivity = json.loads((setup.root / 'manifests/sensitivity.json').read_text())
    # One strong fixed baseline under changed waits: the affinity-aware cap replaces best-fixed.
    assert {r['label'] for r in sensitivity['runs']} == {'fifo', 'affinity-fixed-best', 'joint'}
    assert {r['condition'] for r in sensitivity['runs']} == {'gap-recorded', 'gap-long'}
    assert len(sensitivity['runs']) == 3 * 2 * 2
    for name in ('eval-summary', 'sensitivity-summary'):
        command = stages[name].command()
        assert 'affinity-fixed-best' in command
        assert command[command.index('--confirmatory-trials') + 1] == '0,1'
    assert len(set(setup.make()._key_labels(['fifo']))) == len(setup.make()._key_labels(['fifo']))


def test_legacy_campaign_can_be_inspected_but_not_resumed_or_relocked(setup):
    path = setup.root / 'campaign/lock.json'
    lock = json.loads(path.read_text())
    lock['config'].pop('experiment_revision', None)
    path.write_text(json.dumps(lock))
    original = path.read_bytes()
    assert 'pilot' in setup.make().status().lower()
    with pytest.raises(SystemExit, match='new campaign'):
        setup.make().run()
    with pytest.raises(SystemExit, match='new campaign'):
        setup.make().relock('must not mix old measurements')
    assert not setup.fake.calls
    assert path.read_bytes() == original


def test_environment_cannot_be_relocked_after_measured_dev_started(setup):
    raw = setup.root / 'dev/dev/main/fifo/S101-T0/raw'
    raw.mkdir(parents=True)
    (raw / 'macro_runs.jsonl').write_text('{}\n')
    original = (setup.root / 'campaign/lock.json').read_bytes()
    setup.state['env'] = environment(sources={'phase2/scheduler.py': 'new'})
    with pytest.raises(SystemExit, match='new campaign'):
        setup.make().relock('changed runtime')
    assert (setup.root / 'campaign/lock.json').read_bytes() == original
    assert not (setup.root / 'campaign/lock-history.jsonl').exists()


def test_init_refuses_to_adopt_an_existing_results_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'POINTER', tmp_path / 'pointer')
    root = tmp_path / 'old-pilot'
    root.mkdir()
    (root / 'results.txt').write_text('preserve this')
    c = campaign.Campaign(root, fingerprinter=environment)
    with pytest.raises(SystemExit, match='empty'):
        c.init('m4', 'm4-test')
    assert (root / 'results.txt').read_text() == 'preserve this'
    assert not campaign.POINTER.exists()


def test_warmup_failure_stops_before_measured_runs(setup):
    runner = setup.fake

    def fail_warmup(command):
        if 'phase2.bench' in command and '--results-dir' in command:
            target = Path(command[command.index('--results-dir') + 1])
            if 'warmup' in target.parts:
                return 1
        return runner(command)

    session = campaign.Campaign(setup.root, runner=fail_warmup,
        fingerprinter=lambda: setup.state['env'], power=lambda: setup.state['power'])
    assert session.run() == 'warm-up failed'
    assert 'phase2.plan:run' not in setup.fake.calls


def test_ram_resume_warmup_matches_the_actual_backend_flags(setup):
    from phase2.bench import validate_server
    from phase2.state import PolicyConfig

    seen = []

    def validate_warmup(command):
        policy = command[command.index('--policies') + 1]
        validate_server(PolicyConfig.for_policy(policy), dict(
            parallel=2, ctx_size=24576, ctx_per_slot=12288, kv_unified=False,
            cache_ram_mib=3072, cache_idle_slots=True))
        seen.append(policy)
        return 0

    session = campaign.Campaign(setup.root, runner=validate_warmup)
    session.server_kind = 'ram'
    session._warmup()
    assert seen == ['ram-cache'] and session.abort is None


@pytest.mark.parametrize('artifact', [
    'dev/dev/main/fifo/S101-T0/raw/runs_meta.jsonl',
    'probes/affinity/raw/probes.jsonl', 'profiles/probe.json',
])
def test_relock_cannot_reuse_partial_measurements_or_old_calibration(setup, artifact):
    path = setup.root / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}\n')
    setup.state['env'] = environment(sources={'phase2/scheduler.py': 'new'})
    with pytest.raises(SystemExit, match='new campaign'):
        setup.make().relock('changed code after observations')


@pytest.mark.parametrize('changes', [
    {'llama_server': 'unreviewed build'}, {'model_sha256': 'different model'},
])
def test_relock_revalidates_pinned_runtime_and_model_before_any_measurements(setup, changes):
    setup.state['env'] = environment(**changes)
    original = (setup.root / 'campaign/lock.json').read_bytes()
    with pytest.raises(SystemExit, match='pinned|model'):
        setup.make().relock('incorrect environment')
    assert (setup.root / 'campaign/lock.json').read_bytes() == original


def test_one_failed_warmup_is_retried_before_measuring(setup):
    runner, failures = setup.fake, []

    def flaky_warmup(command):
        if 'phase2.bench' in command and '--results-dir' in command:
            target = Path(command[command.index('--results-dir') + 1])
            if 'warmup' in target.parts and not failures:
                failures.append(target)
                return 1
        return runner(command)

    setup.fake.limit = 3
    session = campaign.Campaign(setup.root, runner=flaky_warmup,
        fingerprinter=lambda: setup.state['env'], power=lambda: setup.state['power'])
    assert session.run() == 'deadline'
    assert len(failures) == 1 and 'phase2.plan:run' in setup.fake.calls
    assert len(list((setup.root / 'campaign/warmup').iterdir())) == 1  # only the retry ran the fake


def test_new_campaign_reuses_pilot_setup_and_unchanged_dev_runs(setup, tmp_path):
    assert setup.make().run().startswith('waiting for review')  # the pilot finished dev
    pilot_dev = json.loads((setup.root / 'manifests/dev.json').read_text())
    failed = next(Path(r['results_dir']) for r in pilot_dev['runs'] if r['label'] == 'fifo' and r['seed'] == 102)
    record = json.loads((failed / 'raw/macro_runs.jsonl').read_text())
    (failed / 'raw/macro_runs.jsonl').write_text(json.dumps(dict(record, failed=1, completed=9, tasks=10)) + '\n')
    setup.state['env'] = environment(sources={'phase2/scheduler.py': 'b'})  # revision 2 code
    root = tmp_path / 'r2'
    make = lambda: campaign.Campaign(root, runner=setup.fake, fingerprinter=lambda: setup.state['env'],
                                     power=lambda: setup.state['power'])
    make().init('m4', 'm4-test', reuse_from=setup.root)

    imported = json.loads((root / 'campaign/imported.json').read_text())
    assert {r['policy'] for r in imported['runs']} == set(campaign.REUSABLE_POLICIES)
    assert len(imported['runs']) == 10 * 3 - 1  # 9 normal labels + RAM-cache; the failed run is redone
    assert imported['code_changes'] == ['code changed: phase2/scheduler.py']
    assert 'imported' in (root / 'campaign/lock-history.jsonl').read_text()
    for path in campaign.REUSABLE_SETUP.values():
        assert (root / path).exists()
    assert (root / 'profiles/profile.json').read_bytes() == (setup.root / 'profiles/profile.json').read_bytes()

    session = make()
    stages = {stage.name: stage for stage in session.stages()}
    assert all(session.done(stages[name]) for name in campaign.REUSABLE_SETUP)
    assert not session.done(stages['smoke'])  # new scheduler code is checked end to end
    setup.fake(stages['dev-manifest'].command())  # model-free: the revision-2 dev manifest
    assert session.progress(stages['dev-runs']) == (27 - 1, 36)  # only joint-a* and the failed run remain
    assert session.progress(stages['dev-runs-ram']) == (3, 3)

    setup.fake.calls.clear()
    assert make().run().startswith('waiting for review')
    assert not any(call.startswith('phase2.probes') for call in setup.fake.calls)


@pytest.mark.parametrize('change', [dict(packages_sha256='q'), dict(python='3.12'),
                                    dict(workloads={'dev': 'other'})])
def test_pilot_from_a_different_environment_is_never_reused(setup, tmp_path, change):
    assert setup.make().run().startswith('waiting for review')
    setup.state['env'] = environment(**change)
    root = tmp_path / 'r2'
    with pytest.raises(SystemExit, match='different'):
        campaign.Campaign(root, runner=setup.fake, fingerprinter=lambda: setup.state['env'],
                          power=lambda: setup.state['power']).init('m4', 'm4-test', reuse_from=setup.root)
    assert not root.exists()  # validated before anything was written
