"""Audit work conservation and complete records after running the experiment suites."""

import json
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main():
    restored = json.loads(Path('results/restore_validation.json').read_text())
    assert restored['generated_tokens_equal']
    assert len(restored['expected_output_tokens']) == len(restored['restored_output_tokens']) == 64
    audited_live_calls = 0
    for path in Path('results/raw/live_test').glob('task_*.json'):
        for call in json.loads(path.read_text())['calls']:
            assert len(call['prompt_tokens']) == call['response']['usage']['prompt_tokens']
            audited_live_calls += 1
    micro = read('results/raw/micro_requests.jsonl')
    assert len(micro) == 54, 'Expected 36 micro cases / 54 requests'
    micro_cases = Counter((r['length'], r['output'], r['k'], r['repeat']) for r in micro)
    assert len(micro_cases) == 36
    for case, count in micro_cases.items():
        assert count == case[2]
    for row in micro:
        assert row['status'] == 'ok'
        assert row['timings']['predicted_n'] == row['output']
        assert row['timings']['cache_n'] == 0
        assert row['timings']['prompt_n'] == row['input_tokens'] == row['length']
        assert not row['truncated']

    cache = read('results/raw/cache.jsonl')
    assert len(cache) == 30
    assert len({(r['mode'], r['length'], r['repeat']) for r in cache}) == 30
    for row in cache:
        assert row.get('status', 'ok') == 'ok', row['label']
        assert row['predicted_n'] == 32

    workload = json.loads(Path('results/workload.json').read_text())
    for task in workload['tasks']:
        current_length = previous_output = 0
        for step in task['steps']:
            current_length += previous_output + len(step['delta'])
            assert current_length == step['source_prompt_n'], 'Replay input length differs from captured source'
            previous_output = step['output']
    jobs = read('results/raw/macro_jobs.jsonl')
    runs = read('results/raw/macro_runs.jsonl')
    requests = read('results/raw/macro_requests.jsonl')
    expected_steps = sum(len(t['steps']) for t in workload['tasks'])
    expected_output = sum(s['output'] for t in workload['tasks'] for s in t['steps'])
    n = len(workload['tasks'])
    assert len(runs) == 6 and len({r['label'] for r in runs}) == 6
    assert len(jobs) == 6 * n
    assert len(requests) == 6 * expected_steps
    for row in jobs:
        assert row['status'] == 'ok' and row['steps'] == row['expected_steps']
        assert row['prefix_verified']
    by_run = defaultdict(list)
    by_step = defaultdict(set)
    for row in requests:
        by_run[row['label']].append(row)
        assert row['status'] == 'ok' and not row['truncated']
        assert row['timings']['predicted_n'] == row['expected_output']
        assert row['prefix_verified']
        by_step[(row['task_id'], row['step'])].add((row['input_tokens'], row['timings']['predicted_n']))
    assert all(len(values) == 1 for values in by_step.values()), 'Unequal token work across configurations'
    for label, rows in by_run.items():
        assert sum(r['timings']['predicted_n'] for r in rows) == expected_output
        assert len(rows) == expected_steps
        # FIFO order refers to dispatch order. Completion order can differ.
        dispatched = sorted(rows, key=lambda r: r['dispatch_s'])
        assert [r['sequence'] for r in dispatched] == list(range(expected_steps)), label
        active = peak = 0
        events = [(r['dispatch_s'], 1) for r in rows] + [(r['done_s'], -1) for r in rows]
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        assert active == 0 and peak <= rows[0]['k'], 'In-flight request limit violated'
    result = dict(micro_cases=36, micro_requests=len(micro), cache_cases=len(cache),
                  audited_live_agent_calls=audited_live_calls,
                  macro_runs=len(runs), macro_jobs=len(jobs), macro_requests=len(requests),
                  steps_per_macro_batch=expected_steps, generated_tokens_per_macro_batch=expected_output,
                  all_terminal_records_present=True, all_output_quotas_match=True,
                  all_macro_token_profiles_match=True, all_prefix_invariants_hold=True,
                  all_replay_input_lengths_match_live_source=True,
                  deterministic_restore_output_verified=True,
                  all_fifo_dispatch_orders_hold=True)
    Path('results/verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
