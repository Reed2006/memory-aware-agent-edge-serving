import hashlib
import json

import pandas as pd
import pytest

from phase2.freeze import freeze, main, select_shared
from phase2.plan import create_manifest
from phase2.state import PolicyConfig


def dev(tmp_path, rows, *, synthetic=False, seeds=(101, 102)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    workload = tmp_path / 'dev.json'
    workload.write_text(json.dumps({'tasks': [
        {'task_id': str(i), 'steps': [{'delta': [1], 'output': out, 'gap_s': 0}]}
        for i, out in enumerate([10, 20, 30, 40])]}))
    base = dict(schema_version=2, platform_id='m4',
                workload_hash=hashlib.sha256(workload.read_bytes()).hexdigest(),
                model_hash='model', kv_precision='f16', P=2, ctx_size=24576,
                kv_unified=False, cost_hash=hashlib.sha256(json.dumps({
                    'schema': 'phase2-cost-profile-v1', 'prefill': {}}).encode()).hexdigest(),
                gap_mode='lognormal:2:1',
                arrival='scaled:1', source_hash='source-v2', failed=0,
                synthetic_performance=synthetic, cache_ram_mib=0, cache_idle_slots=False)
    table, runs = [], []
    for label, policy, rate, slowdown, config in rows:
        config = PolicyConfig.for_policy(policy, **config)
        common = dict(base, policy=policy, policy_label=label, param_hash=config.param_hash,
                      policy_config=config.to_dict(), k=config.concurrency, j=config.jobs_active,
                      checkpoint_budget_bytes=config.checkpoint_budget_bytes)
        if policy == 'ram-cache':
            common.update(cache_ram_mib=3072, cache_idle_slots=True)
        table.append(dict(common, jobs_per_min_mean=rate, slowdown_short_mean=slowdown,
                          p95_s=100., runs=len(seeds), seeds=len(seeds),
                          submitted=4 * len(seeds), completed=4 * len(seeds)))
        for seed in seeds:
            runs.append(dict(common, seed=seed, trial=0, tasks=4, completed=4,
                             run_id=f'{label}-S{seed}', label=f'{label}-S{seed}'))
    (tmp_path / 'summary').mkdir()
    pd.DataFrame(table).to_csv(tmp_path / 'summary/policy_summary.csv', index=False)
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw/macro_runs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in runs))
    return workload


ROWS = [
    ('fifo', 'fifo', .50, 4., {}),
    ('fixed-k1-j1', 'fixed', .40, 9., dict(concurrency=1, jobs_active=1)),
    ('fixed-k2-j2', 'fixed', .70, 9., dict(jobs_active=2)),
    ('affinity', 'affinity', .60, 5., {}),
    ('affinity-fixed-j2', 'affinity-fixed', .65, 8., dict(jobs_active=2)),
    ('ttl-10-a40', 'ttl', .55, 5., dict(ttl_s=10., aging_s=40.)),
    ('ttl-15-a40', 'ttl', .60, 5., dict(ttl_s=15., aging_s=40.)),
    ('ttl-30-a40', 'ttl', .58, 5., dict(ttl_s=30., aging_s=40.)),
    ('queue-a40', 'queue', .60, 4., dict(aging_s=40.)),
    ('attained-a40', 'attained', .62, 4., dict(aging_s=40.)),
    ('cache-a40', 'cache', .78, 6., dict(aging_s=40.)),
    ('joint-a20', 'joint', .80, 10., dict(aging_s=20.)),
    ('joint-a40', 'joint', .75, 8., dict(aging_s=40.)),
    ('joint-checkpoint-a40', 'joint-checkpoint', .85, 9.5, dict(aging_s=40.)),
    ('ram-cache', 'ram-cache', .45, float('nan'), {}),
]


def test_frozen_policies_exactly_match_dev_hashes_and_record_evidence(tmp_path):
    workload = dev(tmp_path, ROWS)
    frozen, notes = freeze(tmp_path, workload, checkpoint_budget_bytes=4 * 2**30)
    policies = {item['label']: item for item in frozen['policies']}
    runs = [json.loads(line) for line in (tmp_path / 'raw/macro_runs.jsonl').read_text().splitlines()]
    measured = {run['param_hash']: run['policy_config'] for run in runs}
    for policy in frozen['policies']:
        config = PolicyConfig.from_dict(policy['config'])
        assert config.param_hash == policy['param_hash'] == policy['evidence']['param_hash']
        assert policy['config'] == measured[policy['param_hash']]
        assert policy['evidence']['seeds'] == [101, 102]
        assert len(policy['evidence']['run_ids']) == 2
    assert policies['best-fixed']['config']['jobs_active'] == 2
    assert 'affinity-fixed-best' not in policies
    assert policies['fixed-ttl']['config']['ttl_s'] == 15.
    assert policies['joint']['config']['queue_rule'] == 'continuation'
    assert all(policies[label]['config']['aging_s'] == 40.
               for label in ['queue', 'cache', 'joint', 'joint-checkpoint', 'fixed-ttl'])
    assert frozen['final_label'] == 'cache' and frozen['short_cutoff'] == 25.
    assert frozen['evidence']['selected_aging_s'] == 40.
    assert frozen['evidence']['dev_workload']['sha256'] == hashlib.sha256(workload.read_bytes()).hexdigest()
    assert any('B2F dropped' in note for note in notes)
    output = tmp_path / 'frozen.json'
    output.write_text(json.dumps(frozen))
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'schema': 'phase2-cost-profile-v1', 'prefill': {}}))
    manifest = create_manifest(phase='eval', workload=workload, cost_profile=profile,
                               results_dir=tmp_path / 'eval', platform_id='m4', seeds=[1],
                               frozen_config=output)
    for run in manifest['runs']:
        if run['label'] != 'isolated':
            assert run['param_hash'] == policies[run['label']]['param_hash']


@pytest.mark.parametrize('policy', ['queue', 'cache', 'joint-checkpoint', 'ttl'])
def test_missing_same_aging_confirmation_fails_closed(tmp_path, policy):
    rows = [(label, name, rate, slowdown, dict(config, aging_s=30.) if name == policy else config)
            for label, name, rate, slowdown, config in ROWS]
    with pytest.raises(ValueError, match=rf'dev confirmation.*{policy}'):
        freeze(tmp_path, dev(tmp_path, rows))


def test_checkpoint_budget_cannot_be_overwritten_after_measurement(tmp_path):
    workload = dev(tmp_path, ROWS)
    with pytest.raises(ValueError, match='checkpoint.*budget'):
        freeze(tmp_path, workload, checkpoint_budget_bytes=2**30)


def test_screen_selects_shared_settings_without_inventing_confirmation_rows(tmp_path):
    screen = [row for row in ROWS if row[1] in {'fifo', 'fixed', 'joint'}]
    dev(tmp_path, screen)
    assert select_shared(tmp_path) == {'aging_s': 40., 'queue_rule': 'continuation'}
    with pytest.raises(ValueError, match='confirmation'):
        freeze(tmp_path, tmp_path / 'dev.json')


def test_mock_missing_fairness_or_missing_joint_cannot_freeze(tmp_path):
    with pytest.raises(ValueError, match='mock'):
        freeze(tmp_path / 'mock', dev(tmp_path / 'mock', ROWS, synthetic=True))
    missing = [(label, policy, rate, float('nan'), config)
               if policy == 'fixed' and config.get('jobs_active') == 2 else
               (label, policy, rate, slowdown, config)
               for label, policy, rate, slowdown, config in ROWS]
    with pytest.raises(ValueError, match='fairness'):
        freeze(tmp_path / 'fairness', dev(tmp_path / 'fairness', missing))
    with pytest.raises(ValueError, match='joint'):
        freeze(tmp_path / 'joint', dev(tmp_path / 'joint', [row for row in ROWS if row[1] != 'joint']))


def test_config_tampering_and_missing_seed_evidence_are_rejected(tmp_path):
    workload = dev(tmp_path, ROWS)
    path = tmp_path / 'raw/macro_runs.jsonl'
    runs = [json.loads(line) for line in path.read_text().splitlines()]
    runs[0]['policy_config']['aging_s'] = 999.
    path.write_text(''.join(json.dumps(row) + '\n' for row in runs))
    with pytest.raises(ValueError, match='hash'):
        freeze(tmp_path, workload)
    other = tmp_path / 'coverage'
    workload = dev(other, ROWS)
    path = other / 'raw/macro_runs.jsonl'
    runs = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(''.join(json.dumps(row) + '\n' for row in runs
                            if not (row['policy'] == 'joint-checkpoint' and row['seed'] == 102)))
    with pytest.raises(ValueError, match='coverage|seed|confirmation'):
        freeze(other, workload)


def test_freeze_rejects_mixed_source_versions_and_workload_mismatch(tmp_path):
    workload = dev(tmp_path, ROWS)
    path = tmp_path / 'summary/policy_summary.csv'
    table = pd.read_csv(path)
    table.loc[table.policy == 'joint', 'source_hash'] = 'different-code'
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match='mix|source'):
        freeze(tmp_path, workload)
    other = tmp_path / 'wrong-workload.json'
    other.write_text(json.dumps({'tasks': [{'steps': [{'output': 1}]}]}))
    with pytest.raises(ValueError, match='workload'):
        freeze(tmp_path, other)


def test_no_fair_final_is_explicit_instead_of_claiming_gate_passed(tmp_path):
    rows = [(label, policy, rate, 12. if policy in {'cache', 'joint-checkpoint'} else slowdown, config)
            for label, policy, rate, slowdown, config in ROWS]
    frozen, _ = freeze(tmp_path, dev(tmp_path, rows))
    assert frozen['final_label'] == 'joint' and frozen['evidence']['final_fairness_passed']
    other = tmp_path / 'all-unfair'
    rows = [(label, policy, rate, 12. if policy == 'joint' else slowdown, config)
            for label, policy, rate, slowdown, config in rows]
    with pytest.raises(ValueError, match='confirmation'):
        freeze(other, dev(other, rows))
    confirmed = tmp_path / 'confirmed-unfair'
    rows = [row for row in rows if row[0] != 'joint-a20']
    rows = [(label, policy, rate, 10. if policy == 'joint-checkpoint' else slowdown, config)
            for label, policy, rate, slowdown, config in rows]
    frozen, notes = freeze(confirmed, dev(confirmed, rows))
    assert frozen['final_label'] == 'joint-checkpoint'
    assert frozen['evidence']['aging_fairness_passed'] is False
    assert frozen['evidence']['final_fairness_passed'] is False
    assert any('no candidate met the fairness bound' in note for note in notes)


def test_cli_error_is_friendly_and_does_not_write_partial_config(tmp_path, capsys):
    workload = dev(tmp_path, [row for row in ROWS if row[1] != 'joint-checkpoint'])
    output = tmp_path / 'frozen.json'
    with pytest.raises(SystemExit) as error:
        main(['--dev-results', str(tmp_path), '--dev-workload', str(workload), '--output', str(output)])
    assert error.value.code == 2 and 'dev confirmation' in capsys.readouterr().err
    assert not output.exists()
