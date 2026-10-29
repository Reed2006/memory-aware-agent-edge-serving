"""Per-request features for the eviction, context-cost and memory-pressure figures.

All times are the bench's perf_counter seconds, which are comparable within one run
(one label). Inputs are the frames written by phase1.summarize (sweep_requests.csv,
sweep_jobs.csv) or the raw JSONL with the same columns.

As a command it prints the contention of every macro run in a results directory, which
is how an arrival rate is calibrated on a new machine:
    python -m phase1.curves --results-dir runs/aws-gpu-pilot
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from phase1.summarize import read, request_metrics


def active_at_dispatch(requests: pd.DataFrame, jobs: pd.DataFrame) -> pd.Series:
    """Agent tasks admitted and not yet finished when each request was dispatched, itself
    included. Tasks still waiting for admission (J) hold no KV and are not counted."""
    spans = jobs[['label', 'admitted_s', 'done_s']]
    req = requests[['label', 'dispatch_s']].reset_index()
    pairs = req.merge(spans, on='label')
    inside = (pairs.admitted_s <= pairs.dispatch_s) & (pairs.dispatch_s < pairs.done_s)
    return inside.groupby(pairs['index']).sum().reindex(requests.index, fill_value=0).astype(int)


def kv_demand(requests: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    """Step function of the KV tokens one run's active tasks need resident to keep their histories.

    A task needs its prompt from the moment its step is queued, prompt + generated tokens once
    the step finishes (the history it wants to keep across the tool call), and nothing after it
    completes. Columns: t_s (from the first arrival), tokens, active (tasks holding a history).
    """
    events = []  # (time, job, history tokens from then on)
    for r in requests.itertuples():
        events.append((r.enqueue_s, r.job, r.input_tokens))
        events.append((r.done_s, r.job, r.input_tokens + (r.expected_output if r.status == 'ok' else 0)))
    events += [(j.done_s, j.job, 0) for j in jobs.itertuples()]
    t0 = jobs.arrival_s.min()
    held, rows = {}, []
    for t, job, tokens in sorted(events, key=lambda e: (e[0], e[2] == 0)):
        held[job] = tokens
        rows.append((t - t0, sum(held.values()), sum(v > 0 for v in held.values())))
    return pd.DataFrame(rows, columns=['t_s', 'tokens', 'active'])


def time_over(demand: pd.DataFrame, budget: int) -> float:
    """Fraction of the run (first arrival to last event) during which demand exceeds the budget."""
    if len(demand) < 2:
        return np.nan
    widths = np.diff(demand.t_s.to_numpy())
    over = demand.tokens.to_numpy()[:-1] > budget
    total = widths.sum()
    return float(widths[over].sum() / total) if total > 0 else np.nan


def contention(root: Path) -> pd.DataFrame:
    """One row per macro run: time-weighted mean active tasks, share of time with more active
    tasks than slots, full-history hit rate and tasks/min."""
    runs, jobs = read(root / 'raw/macro_runs.jsonl'), read(root / 'raw/macro_jobs.jsonl')
    requests = request_metrics(read(root / 'raw/macro_requests.jsonl'))
    rows = []
    for run in runs.itertuples():
        d = kv_demand(requests[requests.label == run.label], jobs[jobs.label == run.label])
        widths = np.diff(d.t_s.to_numpy())
        active = d.active.to_numpy()[:-1]
        rows.append(dict(label=run.label, tasks_per_min=run.jobs_per_minute, makespan_s=run.makespan_s,
                         mean_active=np.average(active, weights=widths) if widths.sum() else np.nan,
                         above_slots=widths[active > run.P].sum() / widths.sum() if widths.sum() else np.nan,
                         hit_rate=requests[requests.label == run.label].full_history_hit.mean()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', required=True)
    table = contention(Path(parser.parse_args().results_dir))
    with pd.option_context('display.width', 200, 'display.max_colwidth', 70):
        print(table.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
