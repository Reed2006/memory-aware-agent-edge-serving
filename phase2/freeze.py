"""Freeze only complete, measured dev configurations; never synthesize a winner.

Rules (seed-averaged dev means; matched seed/trial coverage):
  best fixed   highest tasks/min among fixed and FIFO; ties use lower P95.
  B2F          keep affinity-fixed only when it beats the best fixed baseline.
  aging        fastest measured joint within best fixed's short-task slowdown;
               if none qualifies, the fairest measured joint, explicitly flagged.
  ablations    require measured Q/C/checkpoint/TTL with the chosen joint's aging
               and matching active mechanism settings. No parameter transplant.
  fixed TTL    fastest confirmed TTL at that same aging (bounded dev grid).
  final        fastest confirmed cache/joint/checkpoint within the fairness bound;
               if none qualifies, the fairest one, with fairness_passed=false.
  short cutoff median expected output tokens of the source dev tasks.

All exported policies carry their complete measured configuration, its hash,
raw run identities and seed/trial coverage. Missing confirmation stops freezing.
"""

import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import statistics

import pandas as pd

from phase2.plan import task_work
from phase2.state import PolicyConfig


IDENTITY = ['platform_id', 'workload_hash', 'model_hash', 'kv_precision', 'P', 'ctx_size',
            'kv_unified', 'gap_mode', 'arrival', 'cost_hash']
SOURCE_FIELDS = ['source_hash', 'source_fingerprint', 'code_hash', 'code_fingerprint']
COMMON_MECHANISM = ['concurrency', 'slots', 'ctx_per_slot', 'jobs_active', 'affinity',
                    'aging_s', 'checkpoint_budget_bytes']
CONFIG_FIELDS = {field.name for field in fields(PolicyConfig)}


def _file(path: Path) -> dict:
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _measured_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(Path(root).rglob('macro_runs.jsonl')):
        if path.parent.name != 'raw' or {'interrupted', 'invalid'} & set(path.parts):
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                config = row['policy_config']
                if set(config) != CONFIG_FIELDS:
                    raise ValueError('a complete measured PolicyConfig is required')
                parsed = PolicyConfig.from_dict(config)
                if parsed.param_hash != row['param_hash']:
                    raise ValueError('measured config hash does not match policy_config')
                if parsed.name != row['policy']:
                    raise ValueError('measured config name does not match policy')
                if row.get('schema_version') != 2:
                    raise ValueError('expected schema_version=2')
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'{path}:{line_number}: invalid measured config/hash: {error}') from error
            row['policy_label'] = row.get('policy_label', row['policy'])
            row['_record_path'] = str(path.resolve())
            runs.append(row)
    return runs


def _best(rows: pd.DataFrame) -> pd.Series | None:
    if rows.empty:
        return None
    ordered = rows.assign(_p95=rows.p95_s.fillna(math.inf)).sort_values(
        ['jobs_per_min_mean', '_p95', 'policy_label'], ascending=[False, True, True], kind='stable')
    return ordered.iloc[0]


def _fair_best(rows: pd.DataFrame, bound: float) -> tuple[pd.Series, str, bool]:
    if rows.empty:
        raise ValueError('missing dev confirmation for a cost-lease policy')
    if not math.isfinite(float(bound)) or rows.slowdown_short_mean.isna().any() or \
            not rows.slowdown_short_mean.map(lambda value: math.isfinite(float(value))).all():
        raise ValueError('fairness requires matching isolated dev references for every candidate')
    fair = rows.loc[rows.slowdown_short_mean <= bound]
    if not fair.empty:
        return _best(fair), 'fastest within best-fixed short-task slowdown', True
    minimum = rows.slowdown_short_mean.min()
    return _best(rows.loc[rows.slowdown_short_mean == minimum]), \
        'no candidate met the fairness bound; fairest measured configuration', False


def _coverage(runs: list[dict]) -> set[tuple[int, int]]:
    return {(row['seed'], row['trial']) for row in runs}


def _select(dev_root: Path, dev_workload: Path | None, *, checkpoint_budget_bytes: int | None = None,
            allow_synthetic: bool = False, selection_only: bool = False) -> tuple[dict, list[str]]:
    dev_root = Path(dev_root)
    dev_workload = Path(dev_workload) if dev_workload is not None else None
    summary_path = dev_root / 'summary/policy_summary.csv'
    table = pd.read_csv(summary_path)
    if table.empty:
        raise ValueError('no dev summary rows; finish and summarize the dev matrix first')
    required = set(IDENTITY + ['policy', 'param_hash', 'failed', 'synthetic_performance',
                              'jobs_per_min_mean', 'slowdown_short_mean', 'p95_s'])
    if required - set(table):
        raise ValueError(f'dev summary is missing required evidence columns: {sorted(required - set(table))}')
    if 'policy_label' not in table:
        table['policy_label'] = table.policy
    workload_info = _file(dev_workload) if dev_workload is not None else None
    if workload_info is not None and not table.workload_hash.eq(workload_info['sha256']).all():
        raise ValueError('dev workload hash does not match the summarized source workload')
    if not allow_synthetic and table.synthetic_performance.astype(bool).any():
        raise ValueError('dev summary contains mock-transport runs; they cannot select parameters')
    identity_fields = IDENTITY + [key for key in SOURCE_FIELDS if key in table]
    if table[identity_fields].drop_duplicates().shape[0] != 1:
        raise ValueError('dev summary mixes platforms, workloads, resources, profiles or source versions')
    usable = table.loc[(table.failed == 0) & table.jobs_per_min_mean.notna() &
                       table.jobs_per_min_mean.gt(0)].copy()
    raw = _measured_runs(dev_root)
    notes = []

    def records_of(row) -> list[dict]:
        records = [run for run in raw if run['param_hash'] == row.param_hash and
                   run['policy_label'] == row.policy_label and
                   all(run.get(key) == row[key] for key in identity_fields)]
        if not records:
            raise ValueError(f'missing dev confirmation for {row.policy_label}: no matching raw config/hash')
        if len(_coverage(records)) != len(records) or \
                any(run.get('failed', 0) or run.get('completed') != run.get('tasks') or
                    not run.get('tasks') for run in records):
            raise ValueError(f'missing dev confirmation for {row.policy_label}: duplicate or incomplete raw coverage')
        if 'runs' in row and len(records) != row['runs']:
            raise ValueError(f'missing dev confirmation for {row.policy_label}: raw run coverage differs from summary')
        if 'seeds' in row and len({run['seed'] for run in records}) != row['seeds']:
            raise ValueError(f'missing dev confirmation for {row.policy_label}: raw seed coverage differs from summary')
        return records

    def config_of(row) -> dict:
        records = records_of(row)
        return records[0]['policy_config'].copy()

    fifo = _best(usable.loc[usable.policy == 'fifo'])
    if fifo is None:
        raise ValueError('missing dev confirmation for FIFO baseline')
    expected_coverage = _coverage(records_of(fifo))
    if not expected_coverage:
        raise ValueError('FIFO dev seed/trial coverage is empty')
    # Unequal completed coverage must not make one candidate look like a winner.
    for _, row in usable.iterrows():
        if row.policy_label != 'isolated' and _coverage(records_of(row)) != expected_coverage:
            raise ValueError(f'missing dev confirmation for {row.policy_label}: seed/trial coverage does not match FIFO')
    fixed = usable.loc[usable.policy.isin(['fixed', 'fifo']) & (usable.policy_label != 'isolated')]
    best_fixed = _best(fixed)
    bound = float(best_fixed.slowdown_short_mean)
    if not math.isfinite(bound):
        raise ValueError('best fixed has no matching fairness reference; run isolated dev tasks first')
    notes.append(f'best fixed: {best_fixed.policy_label} ({best_fixed.jobs_per_min_mean:.4g} tasks/min)')
    selected = [('fifo', fifo)]
    if best_fixed.policy != 'fifo':
        selected.append(('best-fixed', best_fixed))
    else:
        notes.append('best fixed is FIFO itself; eval keeps one row')
    b2f = _best(usable.loc[usable.policy == 'affinity-fixed'])
    if b2f is not None and b2f.jobs_per_min_mean > best_fixed.jobs_per_min_mean:
        selected.append(('affinity-fixed-best', b2f))
        notes.append(f'B2F kept: {b2f.policy_label} beats best fixed on dev')
    elif b2f is not None:
        notes.append(f'B2F dropped: {b2f.policy_label} does not beat best fixed on dev')
    joint_rows = usable.loc[usable.policy == 'joint']
    if joint_rows.empty:
        raise ValueError('missing dev confirmation for joint aging candidates')
    joint, aging_reason, aging_fair = _fair_best(joint_rows, bound)
    target = config_of(joint)
    aging = target['aging_s']
    notes.append(f'aging: {joint.policy_label} ({aging_reason})')
    notes.append(f'queue rule: {target["queue_rule"]} (measured in selected joint, never transplanted)')
    if selection_only:
        return dict(aging_s=aging, queue_rule=target['queue_rule']), notes
    for label in ['affinity', 'ram-cache']:
        row = _best(usable.loc[usable.policy == label])
        if row is None:
            raise ValueError(f'missing dev confirmation for {label} baseline')
        selected.append((label, row))

    def matching(names, keys, **requirements):
        rows = usable.loc[usable.policy.isin(names)]
        if rows.empty:
            return rows
        return rows.loc[rows.apply(lambda row: all(config_of(row).get(key) == target[key] for key in keys)
                                  and all(config_of(row).get(key) == value
                                          for key, value in requirements.items()), axis=1)]

    common = COMMON_MECHANISM
    # Inactive parameters (e.g. lease_max_s for an LRU queue) may differ, but each
    # exported config is preserved in full. Active cost-lease settings must match.
    active_cost = common + ['lease_max_s', 'checkpoint_margin_s', 'checkpoint_use_probability']
    confirmed = {
        'queue': matching(['queue', 'attained'], common, queue_rule=target['queue_rule'],
                          cache_policy='lru', checkpoint=False),
        'cache': matching(['cache'], active_cost, queue_rule='fifo', cache_policy='cost', checkpoint=False),
        'joint-checkpoint': matching(['joint-checkpoint'], active_cost + ['queue_rule', 'ttl_s'],
                                     cache_policy='cost', checkpoint=True),
        'ttl': matching(['ttl'], common, queue_rule='fifo', cache_policy='ttl', checkpoint=False),
    }
    missing = [name for name, rows in confirmed.items() if rows.empty]
    if missing:
        raise ValueError(f'missing dev confirmation for aging_s={aging:g}: {", ".join(missing)}. '
                         'Measure those configurations on the same dev seeds/trials before freezing; '
                         'do not copy a score from another aging value.')
    chosen = {name: _best(rows) for name, rows in confirmed.items()}
    if checkpoint_budget_bytes is not None and \
            config_of(chosen['joint-checkpoint'])['checkpoint_budget_bytes'] != checkpoint_budget_bytes:
        raise ValueError('checkpoint budget differs from its measured dev configuration; '
                         'run a matching dev confirmation instead of overriding it')
    selected += [('fixed-ttl', chosen['ttl']), ('queue', chosen['queue']), ('cache', chosen['cache']),
                 ('joint', joint), ('joint-checkpoint', chosen['joint-checkpoint'])]
    final_rows = pd.DataFrame([chosen['cache'], joint, chosen['joint-checkpoint']])
    final, final_reason, final_fair = _fair_best(final_rows, bound)
    final_label = next(label for label, row in selected if row.param_hash == final.param_hash)
    notes += [f'fixed TTL: {chosen["ttl"].policy_label}', f'final system: {final_label} ({final_reason})']
    policies = []
    for label, row in selected:
        records, config = records_of(row), config_of(row)
        evidence = dict(source_policy_label=row.policy_label, param_hash=row.param_hash,
            run_ids=sorted(str(run.get('run_id', run.get('label'))) for run in records),
            seeds=sorted({run['seed'] for run in records}),
            trials_by_seed={str(seed): sorted(run['trial'] for run in records if run['seed'] == seed)
                            for seed in sorted({run['seed'] for run in records})},
            raw_records=sorted({run['_record_path'] for run in records}),
            jobs_per_min_mean=float(row.jobs_per_min_mean),
            slowdown_short_mean=float(row.slowdown_short_mean)
                if pd.notna(row.slowdown_short_mean) else None)
        policy = dict(label=label, policy=config['name'], config=config,
                      param_hash=row.param_hash, evidence=evidence)
        if row.get('cache_ram_mib', 0) or row.get('cache_idle_slots', False):
            policy['resources'] = dict(cache_ram_mib=int(row['cache_ram_mib']),
                                       cache_idle_slots=bool(row['cache_idle_slots']))
        policies.append(policy)
    tasks = json.loads(dev_workload.read_text())['tasks']
    cutoff = float(statistics.median(task_work(task) for task in tasks))
    notes.append(f'short cutoff: {cutoff:g} expected output tokens (dev median)')
    identity = {key: table.iloc[0][key].item() if hasattr(table.iloc[0][key], 'item') else table.iloc[0][key]
                for key in identity_fields}
    evidence = dict(schema='phase2-freeze-evidence-v1', dev_summary=_file(summary_path),
        dev_workload=workload_info, comparison_identity=identity, selected_aging_s=aging,
        aging_fairness_passed=aging_fair, final_fairness_passed=final_fair,
        fairness_bound=bound, coverage=[dict(seed=seed, trial=trial)
                                       for seed, trial in sorted(expected_coverage)])
    frozen = dict(schema='phase2-frozen-config-v1', frozen=True, selected_on='dev',
        short_cutoff=cutoff, policies=policies, final_label=final_label,
        selection_rules=__doc__.split('Rules', 1)[1].strip(), dev_summary=str(summary_path.resolve()),
        evidence=evidence, notes=notes)
    return frozen, notes


def select_shared(dev_root: Path, *, allow_synthetic: bool = False) -> dict:
    """Choose the measured joint aging/rule for a subsequent dev confirmation.

    Uses exactly the same baseline, coverage checks and fairness rule as freeze;
    reads only dev evidence and neither writes files nor runs any experiments.
    """
    selection, _ = _select(dev_root, None, allow_synthetic=allow_synthetic, selection_only=True)
    return selection


def freeze(dev_root: Path, dev_workload: Path, *, checkpoint_budget_bytes: int | None = None,
           allow_synthetic: bool = False) -> tuple[dict, list[str]]:
    return _select(dev_root, dev_workload, checkpoint_budget_bytes=checkpoint_budget_bytes,
                   allow_synthetic=allow_synthetic)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dev-results', type=Path, required=True)
    parser.add_argument('--dev-workload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-budget-gib', type=float,
                        help='assert the measured checkpoint budget; never override a measured configuration')
    args = parser.parse_args(argv)
    try:
        budget = None
        if args.checkpoint_budget_gib is not None:
            if not math.isfinite(args.checkpoint_budget_gib) or args.checkpoint_budget_gib < 0:
                raise ValueError('checkpoint budget must be finite and nonnegative')
            budget = int(args.checkpoint_budget_gib * 2**30)
        frozen, notes = freeze(args.dev_results, args.dev_workload, checkpoint_budget_bytes=budget)
        encoded = json.dumps(frozen, indent=2, allow_nan=False) + '\n'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print('\n'.join(notes))
    print(f'Wrote {args.output}. Review it before creating the eval manifest.')


if __name__ == '__main__':
    main()
