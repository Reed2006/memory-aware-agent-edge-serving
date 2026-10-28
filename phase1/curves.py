"""Per-request features for the eviction, context-cost and memory-pressure figures.

All times are the bench's perf_counter seconds, which are comparable within one run
(one label). Inputs are the frames written by phase1.summarize (sweep_requests.csv,
sweep_jobs.csv) or the raw JSONL with the same columns.
"""

import numpy as np
import pandas as pd


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
