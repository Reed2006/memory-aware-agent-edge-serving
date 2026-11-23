"""A priori load screen for the fairness/efficiency frontier supplement.

Ranks unused workload seeds by how contended their arrival pattern is, using
only the seed's arrivals, its job order and the calibrated cost model. No
measured outcome enters the ranking, so picking the heaviest seeds selects a
load regime, not a result.

Indicator: mean admission wait of a 2-server FCFS run-to-completion queue that
serves each job for its isolated model time (prefill + decode from the cost
profile, tool gaps ignored). On the seeds already run it separates the heavy
ones (1101: 50 s, 2102: 44 s) from the light ones (1102: 4 s, 1103: 8 s,
2101: 9 s).

    .venv/bin/python scripts/phase2/3_frontier_seeds.py            # screen 3101-3130
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase1.bench import arrival_times, job_sources  # noqa: E402
from phase2.costs import CostModel  # noqa: E402


def isolated_service_s(task: dict, costs: CostModel) -> float:
    total, context = 0.0, 0
    for step in task['steps']:
        new = len(step['delta'])
        total += costs.prefill_s(context + new, reused=context) + costs.decode_s(step['output'])
        context += new + step['output']
    return total


def screen(workload: dict, costs: CostModel, seed: int, jobs: int, arrival: str) -> dict:
    service = {t['task_id']: isolated_service_s(t, costs) for t in workload['tasks']}
    arrivals = arrival_times(arrival, jobs, seed)
    order = job_sources(workload['tasks'], jobs, seed)
    free, waits = [0.0, 0.0], []
    for arrived, task in zip(arrivals, order):
        server = min(range(2), key=lambda i: free[i])
        start = max(arrived, free[server])
        waits.append(start - arrived)
        free[server] = start + service[task['task_id']]
    return dict(seed=seed, span_s=arrivals[-1], mean_wait_s=statistics.mean(waits), max_wait_s=max(waits))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workload', type=Path, default=ROOT / 'runs/final/workload_eval.json')
    p.add_argument('--cost-profile', type=Path,
                   default=ROOT / 'runs/phase2/2_m4_campaign/profiles/profile.json')
    p.add_argument('--first', type=int, default=3101)
    p.add_argument('--last', type=int, default=3130)
    p.add_argument('--pick', type=int, default=3)
    p.add_argument('--jobs', type=int, default=20)
    p.add_argument('--arrival', default='scaled:1')
    args = p.parse_args(argv)
    workload = json.loads(args.workload.read_text())
    costs = CostModel.from_file(args.cost_profile)
    rows = [screen(workload, costs, s, args.jobs, args.arrival) for s in range(args.first, args.last + 1)]
    rows.sort(key=lambda r: (-r['mean_wait_s'], r['seed']))
    for r in rows:
        print(f"{r['seed']}  span={r['span_s']:6.0f}s  mean_wait={r['mean_wait_s']:6.1f}s  max_wait={r['max_wait_s']:6.0f}s")
    print('picked:', ','.join(str(r['seed']) for r in rows[:args.pick]))


if __name__ == '__main__':
    main()
