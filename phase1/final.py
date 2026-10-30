"""Figures and the English Phase 1 report for a full sweep directory (default runs/final).

Every section tolerates missing data so a partially finished overnight run still
produces a coherent report; absent experiments are listed as not run.
"""

import argparse
import json
import re
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase1.curves import active_at_dispatch, busy_fraction, contention, kv_demand, time_over
from phase1.noise import spread, trial_rows
from phase1.startup import startup_rows, write_csv
from phase1.summarize import summarize

COLORS = ['#156b7c', '#e47d54', '#6878a9', '#90b7a5', '#bc9a63', '#a05a7a', '#4f8f3a', '#7a7a7a', '#d4a72c', '#3c3c64']
REPEAT_COLORS = ['#156b7c', '#7fb3bd', '#c9dfe3']   # R0, R1, R2: same task mix across configs within a repeat
KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2   # Qwen3-4B: layers x KV heads x head dim x (K,V) x f16
MAIN_GAP, MAIN_ARRIVAL = 'lognormal:2:1', 'poisson:1'
ROOT: Path
FIG: Path
NOTES: list[str] = []


# ---------------------------------------------------------------- helpers

def read_jsonl(name):
    return read_jsonl_at(ROOT, name)


def read_jsonl_at(root, name):
    path = root / 'raw' / name
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(s) for s in path.read_text().splitlines() if s])


def timings(frame, keys=('prompt_ms', 'predicted_ms', 'prompt_n', 'cache_n', 'predicted_n')):
    for key in keys:
        frame[key] = frame['timings'].map(lambda d: d.get(key, np.nan) if isinstance(d, dict) else np.nan)
    return frame


def save(fig, name, note):
    fig.text(0.01, 0.01, note, fontsize=8, color='#5a6575')
    fig.savefig(FIG / f'{name}.png', dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def server_name(row):
    name = f"P{int(row['P'])}{'u' if row.get('kv_unified') else ''}·{int(row['ctx_size']) // 1024}K"
    if row.get('cache_ram_mib'):
        name += f"+RAM{int(row['cache_ram_mib']) // 1024}G"
    return name


def config_name(row):
    j = int(row['j']) if pd.notna(row['j']) else 0
    return f"{server_name(row)} K{int(row['k'])} J{j or '∞'}"


def fmt(x, digits=1, unit=''):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f'{x:,.{digits}f}{unit}'


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def section(fn):
    """Run one figure/analysis; failures become report notes instead of aborting."""
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            NOTES.append(f'{fn.__name__} failed: ' + traceback.format_exc().strip().splitlines()[-1])
            return None
    return wrapped


# ---------------------------------------------------------------- data

def load_macro():
    if not (ROOT / 'raw/macro_jobs.jsonl').exists():
        return None
    summarize(ROOT)
    jobs = pd.read_csv(ROOT / 'summary/sweep_jobs.csv')
    requests = pd.read_csv(ROOT / 'summary/sweep_requests.csv')
    runs = read_jsonl('macro_runs.jsonl')
    for frame in (jobs, requests, runs):
        frame['j'] = frame['j'].fillna(0).astype(int)
        frame['cache_ram_mib'] = frame['cache_ram_mib'].fillna(0)
        frame['kv_unified'] = frame['kv_unified'].fillna(False).astype(bool)
        frame['config'] = frame.apply(config_name, axis=1)
    per_job = requests.groupby('label').apply(lambda g: pd.Series(dict(
        prefill_tokens=g.prompt_n.sum(), wasted=g.wasted_prefill.sum(),
        hit_rate=g.full_history_hit.mean(), fifo_wait_mean=g.queue_s.mean())), include_groups=False)
    req_job = requests.groupby(['label', 'job']).agg(model_s=('roundtrip_s', 'sum'), fifo_s=('queue_s', 'sum')).reset_index()
    jobs = jobs.merge(req_job, on=['label', 'job'], how='left')
    run_rows = []
    for label, group in jobs.groupby('label'):
        done = group[group.status == 'ok']
        meta = runs[runs.label == label].iloc[0] if (runs.label == label).any() else None
        run_rows.append(dict(
            label=label, config=group.config.iloc[0], gap_mode=group.gap_mode.iloc[0],
            arrival=group.arrival.iloc[0], P=group.P.iloc[0], k=group.k.iloc[0], j=group.j.iloc[0],
            ctx_size=group.ctx_size.iloc[0], kv_unified=group.kv_unified.iloc[0],
            cache_ram_mib=group.cache_ram_mib.iloc[0], repeat=group.repeat.iloc[0],
            trial=int(meta['trial']) if meta is not None and pd.notna(meta.get('trial')) else 0,
            submitted=len(group), completed=len(done),
            jobs_per_min=meta['jobs_per_minute'] if meta is not None else np.nan,
            makespan_s=meta['makespan_s'] if meta is not None else np.nan,
            p50=done.latency_s.median(), p95=done.latency_s.quantile(.95),
            slowdown=done.slowdown.mean(),
            wasted_frac=per_job.loc[label, 'wasted'] / per_job.loc[label, 'prefill_tokens'] if label in per_job.index and per_job.loc[label, 'prefill_tokens'] else np.nan,
            hit_rate=per_job.loc[label, 'hit_rate'] if label in per_job.index else np.nan))
    trial_runs = pd.DataFrame(run_rows).sort_values('trial')
    return dict(jobs=jobs, requests=requests, runs=average_trials(trial_runs), trial_runs=trial_runs,
                summary=pd.read_csv(ROOT / 'summary/sweep_summary.csv'))


RUN_KEYS = ['config', 'gap_mode', 'arrival', 'P', 'k', 'j', 'ctx_size', 'kv_unified', 'cache_ram_mib', 'repeat']
RUN_METRICS = ['jobs_per_min', 'makespan_s', 'p50', 'p95', 'slowdown', 'wasted_frac', 'hit_rate']


def average_trials(trial_runs):
    """One row per configuration and repeat: trials replay identical inputs, so they are averaged
    (counts summed); jobs_per_min_tsd is their spread. `label` is the first trial's run."""
    agg = dict(label=('label', 'first'), trials=('label', 'size'), submitted=('submitted', 'sum'),
               completed=('completed', 'sum'), jobs_per_min_tsd=('jobs_per_min', 'std'),
               **{m: (m, 'mean') for m in RUN_METRICS})
    return trial_runs.groupby(RUN_KEYS, as_index=False, dropna=False).agg(**agg)


def main_condition(frame):
    return frame[(frame.gap_mode == MAIN_GAP) & (frame.arrival == MAIN_ARRIVAL)]


def order_configs(names):
    def key(name):
        p = int(name.split('P')[1].split('u')[0].split('·')[0])
        return (p, 'u' in name.split('·')[0], 'RAM' in name, name)
    return sorted(set(names), key=key)


def per_repeat_bars(ax, runs, configs, metric, scale=1.0):
    """Grouped bars, one per repeat: repeats are different task mixes, so they are shown, not averaged."""
    repeats = sorted(runs.repeat.unique())
    width = .8 / len(repeats)
    x = np.arange(len(configs))
    for i, r in enumerate(repeats):
        rows = runs[runs.repeat == r].set_index('config').reindex(configs)
        # Error bars: spread over trials of the same repeat, when trials were run.
        err = rows[f'{metric}_tsd'] * scale if f'{metric}_tsd' in rows and rows[f'{metric}_tsd'].notna().any() else None
        ax.bar(x + (i - (len(repeats) - 1) / 2) * width, rows[metric] * scale, width * .95, yerr=err, capsize=2,
               color=REPEAT_COLORS[i % len(REPEAT_COLORS)], label=f'R{r}')
    ax.set_xticks(x, configs, rotation=35, ha='right', fontsize=8)


def by_config(runs, metric):
    grouped = runs.groupby('config')[metric].agg(['mean', 'std', 'count'])
    return grouped.reindex(order_configs(grouped.index))


# ---------------------------------------------------------------- figures

@section
def fig_workload(live):
    test, train = live.get('test', []), live.get('train', [])
    if not test:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for color, (name, records) in zip(COLORS, [('eval (test)', test), ('dev (train)', train)]):
        steps = [len(r['calls']) for r in records]
        axes[0].hist(steps, bins=range(0, max(steps) + 4, 2), alpha=.6, color=color, label=f'{name}, n={len(records)}')
    for record in test:
        lengths = [len(c['prompt_tokens']) for c in record['calls']]
        ok = record.get('env_reward') == 1
        axes[1].plot(range(len(lengths)), lengths, color=COLORS[0] if ok else COLORS[1], alpha=.7, lw=1.4)
    outputs = [c['response']['usage']['completion_tokens'] for r in test + train for c in r['calls']]
    axes[2].hist(outputs, bins=40, color=COLORS[2])
    axes[0].set(title='Agent model calls per task', xlabel='Calls', ylabel='Tasks')
    axes[0].legend(frameon=False)
    axes[1].set(title='Context growth (teal: ENV pass, orange: fail)', xlabel='Agent step', ylabel='Prompt tokens')
    axes[2].set(title='Generated tokens per agent call', xlabel='Tokens', ylabel='Calls')
    fig.suptitle('F1  Live τ²-bench Retail workload (Qwen3-4B agent, DeepSeek user)', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f01_workload', 'Source of the macro workload. ENV = programmatic database/environment check only.')


@section
def fig_micro():
    data = read_jsonl('micro_requests.jsonl')
    if data.empty:
        return None
    data = timings(data)
    ok = data[data.status == 'ok']
    per_run = ok.groupby(['length', 'output', 'k', 'repeat']).agg(
        prefill_s=('prompt_ms', lambda x: x.mean() / 1000), gen_s=('predicted_ms', lambda x: x.mean() / 1000),
        roundtrip_s=('roundtrip_s', 'mean'), generated=('predicted_n', 'sum'),
        start=('dispatch_s', 'min'), end=('done_s', 'max')).reset_index()
    per_run['tok_s'] = per_run.generated / (per_run.end - per_run.start)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    styles = {64: '-', 512: '--'}
    for ci, k in enumerate(sorted(per_run.k.unique())):
        for output in sorted(per_run.output.unique()):
            sub = per_run[(per_run.k == k) & (per_run.output == output)].groupby('length')
            for ax, metric in zip(axes, ['prefill_s', 'roundtrip_s', 'tok_s']):
                if metric == 'prefill_s' and output != min(per_run.output):
                    continue
                g = sub[metric].agg(['mean', 'std'])
                ax.errorbar(g.index, g['mean'], yerr=g['std'], color=COLORS[ci], ls=styles.get(output, ':'),
                            marker='o', capsize=3, label=f'K={k}, out={output}')
    axes[0].set(title='Prefill time per request', xlabel='Input tokens', ylabel='s')
    axes[1].set(title='Request round trip', xlabel='Input tokens', ylabel='s')
    axes[2].set(title='Aggregate generated tokens/s', xlabel='Input tokens', ylabel='tokens/s')
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('F2  Micro: cost of context length, output length and concurrency (prompt cache off)', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f02_micro', 'Mean ± SD over repeats. K requests submitted together on P=2 slots.')
    return per_run


@section
def fig_cache():
    data = read_jsonl('cache.jsonl')
    if data.empty:
        return None
    data = timings(data[data.get('status', pd.Series('ok', index=data.index)).fillna('ok') == 'ok'].copy())
    data['reuse'] = data.cache_n / data.input_tokens
    modes = ['retain', 'wait', 'overwrite', 'erase', 'save_restore']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    lengths = sorted(data.length.unique())
    for i, length in enumerate(lengths):
        g = data[data.length == length].groupby('mode').prompt_ms.agg(['mean', 'std']).reindex(modes) / 1000
        axes[0].bar(np.arange(5) + i * .38, g['mean'], .36, yerr=g['std'], color=COLORS[i], label=f'{length:,} tokens', capsize=3)
    axes[0].set(yscale='log', xticks=np.arange(5) + .19, ylabel='Continuation prefill (s, log)', title='Cost of losing an agent\'s history')
    axes[0].set_xticklabels(['retain', 'idle 0.5 s', 'overwrite', 'erase', 'save+restore'])
    axes[0].legend(frameon=False)
    sr = data[data['mode'] == 'save_restore'].copy()
    rows = None
    if not sr.empty:
        sr['state_mib'] = sr['save'].map(lambda d: d.get('n_written', np.nan) / 2**20 if isinstance(d, dict) else np.nan)
        sr['mechanism_s'] = sr.save_roundtrip_s + sr.erase_roundtrip_s + sr.restore_roundtrip_s + sr.prompt_ms / 1000
        rebuild = data[data['mode'] == 'erase'].groupby('length').prompt_ms.mean() / 1000
        rows = sr.groupby('length').agg(state_mib=('state_mib', 'mean'), save_s=('save_roundtrip_s', 'mean'),
                                       restore_s=('restore_roundtrip_s', 'mean'), mechanism_s=('mechanism_s', 'mean'))
        rows['rebuild_s'] = rebuild
        x = np.arange(len(rows))
        axes[1].bar(x - .2, rows.rebuild_s, .38, color=COLORS[1], label='re-prefill after erase')
        axes[1].bar(x + .2, rows.mechanism_s, .38, color=COLORS[0], label='save + erase + restore + prefill')
        for xi, (_, r) in zip(x, rows.iterrows()):
            axes[1].text(xi + .2, r.mechanism_s, f'{r.state_mib:.0f} MiB', ha='center', va='bottom', fontsize=8)
        axes[1].set(yscale='log', xticks=x, xticklabels=[f'{l:,}' for l in rows.index], xlabel='Input tokens',
                    ylabel='s (log)', title='Checkpointing KV to SSD vs recomputing')
        axes[1].legend(frameon=False)
    fig.suptitle('F3  Cache mechanisms exposed by llama-server (slot reuse, erase, save/restore)', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f03_cache', 'Warm page cache; save is not fsync-forced. Overwrite = another request on the same slot.')
    return data, rows


@section
def fig_main(macro):
    runs = main_condition(macro['runs'])
    if runs.empty:
        return
    configs = order_configs(runs.config)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    per_repeat_bars(axes[0], runs, configs, 'jobs_per_min')
    axes[0].set(title='Completed tasks / min, per repeat', ylabel='tasks / min')
    axes[0].legend(frameon=False, title='repeat', fontsize=8)
    pivot, ratio = paired_throughput(runs)
    if ratio is not None:
        x = np.arange(len(configs))
        for i, r in enumerate(ratio.columns):
            axes[1].scatter(x, ratio.loc[configs, r], color=REPEAT_COLORS[i % 3], edgecolor='#333', s=40, zorder=3, label=f'R{r}')
        axes[1].axhline(1, color='#999', lw=1, ls='--')
        axes[1].set_xticks(x, configs, rotation=35, ha='right', fontsize=8)
        axes[1].set(title=f'Throughput relative to FIFO {REF_CONFIG} (same tasks)', ylabel='× FIFO')
        axes[1].legend(frameon=False, title='repeat', fontsize=8)
    sub = runs.groupby('config')[['submitted', 'completed']].sum().reindex(configs)
    failed = sub.submitted - sub.completed
    bars = axes[2].bar(np.arange(len(configs)), failed / sub.submitted * 100, color=COLORS[1])
    for bar, f, n in zip(bars, failed, sub.submitted):
        if f:
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{f}/{n}', ha='center', va='bottom', fontsize=8)
    axes[2].set_xticks(np.arange(len(configs)), configs, rotation=35, ha='right', fontsize=8)
    axes[2].set(title='Failed tasks (context overflow)', ylabel='% of submitted', ylim=(0, 100))
    fig.suptitle('F4  End-to-end FIFO baselines by server configuration and KV budget', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .06, 1, .93])
    save(fig, 'f04_main', f'Eval workload (test split), arrivals {MAIN_ARRIVAL}, tool delay {MAIN_GAP}. '
                          'P = slots, u = unified KV, K = request concurrency, J = active-task limit. Each repeat is a '
                          'different seeded mix of 8 tasks with its own arrival pattern (R1 fewest steps, R2 burstiest arrivals).')


@section
def fig_breakdown(macro):
    jobs = main_condition(macro['jobs'])
    jobs = jobs[jobs.status == 'ok']
    if jobs.empty:
        return None
    parts = jobs.groupby('config')[['admission_wait_s', 'fifo_s', 'model_s', 'gap_total_s', 'latency_s']].mean()
    parts = parts.reindex(order_configs(parts.index))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    bottom = np.zeros(len(parts))
    for color, (col, label) in zip(COLORS, [('admission_wait_s', 'admission wait (J)'), ('fifo_s', 'FIFO queue wait'),
                                            ('model_s', 'model round trip'), ('gap_total_s', 'tool delay')]):
        ax.bar(parts.index, parts[col], bottom=bottom, color=color, label=label)
        bottom += parts[col].values
    ax.scatter(parts.index, parts.latency_s, color='black', marker='_', s=300, label='measured completion time', zorder=3)
    ax.set(ylabel='mean seconds per completed task', title='Where does a task spend its time?')
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    fig.suptitle('F5  Completion-time breakdown', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f05_breakdown', 'Model round trip includes waiting inside llama-server when K > 1 (shared batch).')
    return parts


@section
def fig_bottleneck(macro):
    runs = main_condition(macro['runs'])
    if runs.empty:
        return
    configs = order_configs(runs.config)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric, title in [(axes[0], 'wasted_frac', 'Share of prefill tokens that recompute history'),
                              (axes[1], 'hit_rate', 'Continuations that reused their full history')]:
        per_repeat_bars(ax, runs, configs, metric, scale=100)
        ax.set(title=title, ylabel='%', ylim=(0, 105))
    axes[0].legend(frameon=False, title='repeat', fontsize=8)
    fig.suptitle('F6  The bottleneck: history eviction and re-prefill', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f06_bottleneck', 'Wasted = prompt tokens evaluated beyond the new tokens of the step (append-only replay); '
                                'hit = cache_n ≥ previous prompt + output − 1. J1 configurations waste 0%.')


@section
def fig_memory(macro):
    res = read_jsonl('resources.jsonl')
    if res.empty:
        return None
    runs = macro['trial_runs']   # every run, so the samples of all trials are kept
    res = res.merge(runs[['label', 'config', 'ctx_size', 'cache_ram_mib', 'P', 'kv_unified']], on='label')
    if res.empty:
        return None
    res['server'] = res.apply(server_name, axis=1)
    g = res.groupby('server').agg(rss_gib=('server_rss', lambda x: x.max() / 2**30),
                                  avail_gib=('system_available', lambda x: x.min() / 2**30),
                                  ctx=('ctx_size', 'first'), cache_ram=('cache_ram_mib', 'first'))
    g['kv_gib'] = g.ctx * KV_BYTES_PER_TOKEN / 2**30
    g = g.reindex(order_configs(g.index))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(g))
    ax.bar(x - .27, g.rss_gib, .26, color=COLORS[0], label='peak sampled server RSS')
    ax.bar(x, g.kv_gib, .26, color=COLORS[2], label='theoretical f16 KV pool')
    ax.bar(x + .27, g.avail_gib, .26, color=COLORS[3], label='min system available')
    ax.set(xticks=x, ylabel='GiB', title='Memory per server configuration')
    ax.set_xticklabels(g.index, rotation=20)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle('F7  Memory: the KV pool is preallocated by ctx-size, not by load', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f07_memory', 'RSS sampled every 250 ms; on Apple unified memory RSS does not capture every Metal allocation.')
    return g


def sweep_subset(macro, column, values):
    runs = macro['runs']
    base = runs[(runs.P == 2) & (runs.k == 2) & (runs.j == 0) & (~runs.kv_unified) & (runs.cache_ram_mib == 0)]
    other = 'arrival' if column == 'gap_mode' else 'gap_mode'
    fixed = MAIN_ARRIVAL if column == 'gap_mode' else MAIN_GAP
    sub = base[(base[other] == fixed) & base[column].isin(values)]
    # Repeats draw different task mixes, so compare only repeats that every value has.
    shared = set.intersection(*(set(g.repeat) for _, g in sub.groupby(column))) if not sub.empty else set()
    sub = sub[sub.repeat.isin(shared)]
    return sub if sub[column].nunique() > 1 else pd.DataFrame()


@section
def fig_sweep(macro, column, values, labels, name, title, note):
    sub = sweep_subset(macro, column, values)
    if sub.empty:
        return None
    present = [v for v in values if v in set(sub[column])]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    metrics = [('jobs_per_min', 'Tasks / min', 1, None), ('p95', 'P95 completion (s)', 1, None),
               ('wasted_frac', 'Wasted prefill share (%)', 100, (0, 100)), ('hit_rate', 'Full-history hit rate (%)', 100, (0, 105))]
    for ax, (metric, t, scale, ylim) in zip(axes, metrics):
        for i, (r, g) in enumerate(sub.groupby('repeat')):
            g = g.set_index(column)[metric].reindex(present) * scale
            ax.plot(range(len(present)), g, marker='o', color=REPEAT_COLORS[i % 3], mec='#333', label=f'R{r}')
        mean = sub.groupby(column)[metric].mean().reindex(present) * scale
        ax.plot(range(len(present)), mean, color='black', lw=2.2, label='mean')
        ax.set(title=t, xticks=range(len(present)))
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticklabels([labels[values.index(v)] for v in present], fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .92])
    save(fig, name, note + ' Thin lines: individual repeats (paired task mixes); thick line: mean.')
    return sub.groupby(column)[['jobs_per_min', 'p50', 'p95', 'wasted_frac', 'hit_rate', 'slowdown']].mean().reindex(present)


@section
def fig_fairness(macro):
    jobs = main_condition(macro['jobs'])
    jobs = jobs[(jobs.status == 'ok') & jobs.slowdown.notna()]
    if jobs.empty:
        return None
    configs = order_configs(jobs.config)
    cut = jobs.expected_output_tokens.median()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for color, c in zip(COLORS * 2, configs):
        s = np.sort(jobs[jobs.config == c].slowdown)
        axes[0].step(s, np.arange(1, len(s) + 1) / len(s), where='post', color=color, label=c)
    axes[0].set(xscale='log', xlabel='slowdown (log)', ylabel='fraction of tasks', title='Slowdown distribution')
    axes[0].legend(frameon=False, fontsize=7)
    long = jobs.assign(kind=np.where(jobs.expected_output_tokens > cut, 'long', 'short'))
    g = long.groupby(['config', 'kind']).slowdown.mean().unstack().reindex(configs)
    x = np.arange(len(g))
    for i, kind in enumerate(['short', 'long']):
        if kind in g:
            axes[1].bar(x + (i - .5) * .38, g[kind], .36, color=COLORS[i], label=f'{kind} tasks')
    axes[1].set(xticks=x, ylabel='mean slowdown', title='Short vs long tasks: admission limits make short tasks wait')
    axes[1].set_xticklabels(g.index, rotation=35, ha='right', fontsize=8)
    axes[1].legend(frameon=False)
    fig.suptitle('F8  Fairness: slowdown relative to running alone', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f08_fairness', 'Slowdown = completion time / (isolated compute time from J=1,K=1 runs + own tool delays). Long = above-median generated tokens.')
    return g


@section
def fig_timeline(macro):
    req = macro['requests']
    candidates = main_condition(macro['runs'])
    candidates = candidates[(candidates.P == 2) & (candidates.k == 2) & (candidates.j == 0) & (candidates.cache_ram_mib == 0) & (~candidates.kv_unified)]
    if candidates.empty:
        return
    label = candidates.sort_values('repeat').label.iloc[0]
    sub = req[req.label == label]
    t0 = sub.enqueue_s.min()
    fig, ax = plt.subplots(figsize=(14, 5))
    jobs = sorted(sub.job.unique())
    for y, job in enumerate(jobs):
        rows = sub[sub.job == job]
        for _, r in rows.iterrows():
            ax.broken_barh([(r.enqueue_s - t0, r.queue_s)], (y - .35, .7), color='#d6dbe2')
            hit = r.full_history_hit == 1 or r.step == 0
            ax.broken_barh([(r.dispatch_s - t0, r.roundtrip_s)], (y - .35, .7), color=COLORS[0] if hit else COLORS[1])
    ax.set(yticks=range(len(jobs)), yticklabels=[f'job {j}' for j in jobs], xlabel='seconds',
           title='teal: model call with history reused · orange: history re-prefilled · grey: FIFO wait')
    fig.suptitle(f'F9  One run: {label}', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f09_timeline', 'Blank intervals are tool delays and arrivals.')


ACTIVE_BINS = range(1, 7)   # 6 = six or more active tasks


@section
def fig_eviction(macro):
    req = macro['requests'].assign(active=active_at_dispatch(macro['requests'], macro['jobs']))
    cont = req[(req.step > 0) & (req.status == 'ok')].copy()
    if cont.empty:
        return None
    cont['server'] = cont.apply(server_name, axis=1)
    cont['bin'] = cont.active.clip(upper=max(ACTIVE_BINS))
    g = cont.groupby(['server', 'bin']).full_history_hit.agg(['mean', 'count'])
    g = g[g['count'] >= 10]   # fewer continuations than this give no usable rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    servers = order_configs(g.index.get_level_values('server'))
    for color, server in zip(COLORS, servers):
        s = g.loc[server]
        axes[0].plot(s.index, s['mean'] * 100, marker='o', color=color, mec='#333', label=server)
    ticks = dict(xticks=list(ACTIVE_BINS), xticklabels=[str(b) for b in ACTIVE_BINS][:-1] + [f'{max(ACTIVE_BINS)}+'])
    axes[0].set(xlabel='active agent tasks when the step was dispatched', ylabel='full-history hit rate (%)',
                ylim=(-3, 105), title='Hit rate vs active tasks (all runs, pooled)', **ticks)
    axes[0].legend(frameon=False, fontsize=8)
    main = main_condition(cont)
    pair = [('P2·24K K2 J∞', COLORS[1]), ('P2·24K K2 J2', COLORS[0])]
    for i, (name, color) in enumerate(pair):
        share = main[main.config == name].bin.value_counts(normalize=True).reindex(ACTIVE_BINS, fill_value=0)
        axes[1].bar(np.array(ACTIVE_BINS) + (i - .5) * .38, share * 100, .36, color=color, label=name)
    axes[1].set(xlabel='active agent tasks when the step was dispatched', ylabel='% of continuations',
                title='How often each level occurs (P2, main condition)', **ticks)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('F12  Eviction cliff: histories survive only while active tasks fit the slots', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f12_eviction', 'Active = admitted and not finished (tasks waiting for admission hold no KV). Pooled over K, J, '
                              'delays and arrivals, so load is a confounder: descriptive, not controlled. Points need ≥ 10 continuations.')
    return g['mean'].unstack('bin').reindex(servers), g['count'].unstack('bin').reindex(servers)


@section
def fig_context_cost(macro):
    req = main_condition(macro['requests'])
    cont = req[(req.step > 0) & (req.status == 'ok') & (req.P == 2) & (req.k == 2) & (~req.kv_unified)
               & (req.cache_ram_mib == 0)].copy()
    if cont.empty:
        return None
    prefix = shared_prefix_tokens() or 0
    cont['prefill_s'] = cont.prompt_ms / 1000
    cont['hit'] = cont.full_history_hit == 1
    edges = np.arange(cont.input_tokens.min() // 1000 * 1000, cont.input_tokens.max() + 1000, 1000)
    cont['ctx'] = pd.cut(cont.input_tokens, edges, right=False)
    med = cont.groupby(['hit', 'ctx'], observed=True).prefill_s.median().unstack('hit')
    mids = [b.left + 500 for b in med.index]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for hit, color, name in [(True, COLORS[0], 'history reused'), (False, COLORS[1], 'history re-prefilled')]:
        s = cont[cont.hit == hit]
        axes[0].scatter(s.input_tokens, s.prefill_s, s=9, alpha=.35, color=color)
        if hit in med:
            axes[0].plot(mids, med[hit], marker='o', color=color, mec='#333', lw=2, label=f'{name} (median per 1K)')
    axes[0].set(xlabel='context length at the step (tokens)', ylabel='prefill time of the step (s)',
                title='A miss costs more the longer the history')
    axes[0].legend(frameon=False, fontsize=8)
    miss = cont[~cont.hit]
    axes[1].scatter(miss.input_tokens, miss.prompt_n, s=9, alpha=.4, color=COLORS[1], label='misses')
    x = np.array([cont.input_tokens.min(), cont.input_tokens.max()])
    axes[1].plot(x, x - prefix, color='#333', lw=1.2, ls='--', label=f'context − {prefix:,}-token shared prefix')
    axes[1].set(xlabel='context length at the step (tokens)', ylabel='tokens prefilled on a miss',
                title='What a miss recomputes: the task-specific history')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('F13  Context growth: hits stay cheap, misses grow with the history', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f13_context_cost', 'P2·24K K2 (J∞ and J2), main condition, all repeats. Prefill time includes sharing the batch '
                                  'with the other slot; the shared system prompt usually survives a miss in the slot.')
    slope = np.polyfit(miss.prompt_n, miss.prefill_s, 1)[0] if len(miss) > 2 else np.nan
    return dict(median=med, hit_s=cont[cont.hit].prefill_s.median(), slope=slope, misses=len(miss),
                continuations=len(cont), prefix=prefix, recompute_median=miss.prompt_n.median())


@section
def fig_kv_demand(macro):
    runs = main_condition(macro['trial_runs'])
    runs = runs[(runs.P == 2) & (runs.k == 2) & (~runs.kv_unified) & (runs.cache_ram_mib == 0) & runs.j.isin([0, 2])]
    if runs.empty:
        return None
    req, jobs = macro['requests'], macro['jobs']
    rows, curves = [], {}
    for r in runs.itertuples():
        d = kv_demand(req[req.label == r.label], jobs[jobs.label == r.label])
        widths = np.diff(d.t_s.to_numpy())
        curves[r.label] = d
        rows.append(dict(label=r.label, config=r.config, repeat=r.repeat, trial=r.trial, budget=r.ctx_size,
                         peak=d.tokens.max(), over=time_over(d, r.ctx_size),
                         mean_active=np.average(d.active[:-1], weights=widths), hit_rate=r.hit_rate, jobs_per_min=r.jobs_per_min))
    per_run = pd.DataFrame(rows)
    tab = per_run.groupby(['config', 'repeat'], as_index=False).agg(
        trials=('label', 'size'), budget=('budget', 'first'), peak=('peak', 'mean'), over=('over', 'mean'),
        mean_active=('mean_active', 'mean'), hit_rate=('hit_rate', 'mean'), jobs_per_min=('jobs_per_min', 'mean'))
    tab = tab.sort_values(['repeat', 'config'])
    fifo = tab[tab.config == 'P2·24K K2 J∞']
    if fifo.empty:
        return tab
    repeat = fifo.sort_values('over').repeat.iloc[-1]   # the repeat where FIFO is under the most pressure
    # Plot the first trial of each configuration in that repeat.
    shown = per_run[per_run.repeat == repeat].sort_values('trial').drop_duplicates('config').sort_values('config', ascending=False)
    fig, axes = plt.subplots(len(shown), 1, figsize=(14, 2.6 * len(shown) + .6), sharex=True, sharey=True, squeeze=False)
    for ax, r, color in zip(axes[:, 0], shown.itertuples(), [COLORS[1], COLORS[0]]):
        d = curves[r.label]
        ax.fill_between(d.t_s, d.tokens / 1000, step='post', color=color, alpha=.35, lw=0)
        ax.step(d.t_s, d.tokens / 1000, where='post', color=color, lw=1.4)
        ax.axhline(r.budget / 1000, color='#333', ls='--', lw=1)
        ax.text(0, r.budget / 1000, f' KV budget {r.budget:,} tokens', va='bottom', fontsize=8)
        ax.set(ylabel='thousand tokens', title=f'{r.config}: demand above budget {fmt(r.over * 100, 0, "%")} of the run, '
                                                f'mean {fmt(r.mean_active, 1)} active tasks, full-history hits {fmt(r.hit_rate * 100, 0, "%")}')
    axes[-1, 0].set_xlabel('seconds since the first arrival')
    fig.suptitle(f'F14  Memory pressure in tokens: KV the active tasks need to keep their histories (R{repeat})',
                 x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f14_kv_demand', 'Demand = sum over admitted, unfinished tasks of prompt (+ generated tokens once the step is '
                               'done). The pool is preallocated (F7), so demand above the budget means histories are evicted.')
    return tab


ADMISSION = [('P2·24K K1 J1', '1 (K1)'), ('P2·24K K2 J2', '2'), ('P2·24K K2 J3', '3'), ('P2·24K K2 J4', '4'), ('P2·24K K2 J∞', '∞')]


@section
def fig_admission(macro):
    """Admission curve: P2 under the main condition as the active-task limit J grows."""
    present = [(c, l) for c, l in ADMISSION if c in set(macro['trial_runs'].config)]
    if len(present) < 3:
        return None
    trial_runs = main_condition(macro['trial_runs'])
    req, jobs = macro['requests'], macro['jobs']
    trial_runs = trial_runs[trial_runs.config.isin([c for c, _ in present])].copy()
    trial_runs['busy'] = [busy_fraction(req[req.label == l], jobs[jobs.label == l]) for l in trial_runs.label]
    per = trial_runs.groupby(['config', 'repeat']).agg(tpm=('jobs_per_min', 'mean'), tpm_sd=('jobs_per_min', 'std'),
                                                       p95=('p95', 'mean'), hit=('hit_rate', 'mean'), busy=('busy', 'mean'))
    done = main_condition(jobs[(jobs.status == 'ok') & jobs.slowdown.notna()])
    short = done[done.expected_output_tokens <= done.expected_output_tokens.median()].groupby('config').slowdown.mean()
    x = np.arange(len(present))
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.4))
    panels = [('tpm', 'Completed tasks / min', 1), ('p95', 'P95 completion (s)', 1), ('hit', 'Full-history hit rate (%)', 100),
              ('busy', 'Server busy (% of run)', 100)]
    for ax, (metric, title, scale) in zip(axes, panels):
        for i, r in enumerate(sorted(per.index.get_level_values('repeat').unique())):
            vals = per.xs(r, level='repeat').reindex([c for c, _ in present])
            err = vals.tpm_sd if metric == 'tpm' and vals.tpm_sd.notna().any() else None
            ax.errorbar(x, vals[metric] * scale, yerr=err, marker='o', capsize=3, color=REPEAT_COLORS[i % 3], mec='#333', label=f'R{r}')
        ax.set(title=title, xticks=x, xlabel='active-task limit J')
        ax.set_xticklabels([l for _, l in present])
    axes[0].legend(frameon=False, fontsize=8, title='repeat')
    axes[4].bar(x, short.reindex([c for c, _ in present]), color=COLORS[0])
    axes[4].set(title='Short-task slowdown (mean)', xticks=x, xlabel='active-task limit J')
    axes[4].set_xticklabels([l for _, l in present])
    fig.suptitle('F15  Admission curve: P2 under the main condition as the active-task limit grows', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f15_admission', 'J = 1 runs with K = 1; the other points with K = 2. Error bars: SD over trials of the same repeat. '
                               'Server busy = share of the run with at least one request in the server.')
    table_rows = per.groupby('config').mean().reindex([c for c, _ in present])
    table_rows['short'] = short.reindex(table_rows.index)
    table_rows['label'] = [l for _, l in present]
    return table_rows


@section
def fig_calibration():
    """Arrival-rate calibration from <results>-pilot: FIFO contention against the offered rate."""
    pilot = ROOT.parent / (ROOT.name + '-pilot')
    if not (pilot / 'raw/macro_runs.jsonl').exists():
        return None
    cont = contention(pilot)
    runs = read_jsonl_at(pilot, 'macro_runs.jsonl')[['label', 'arrival', 'repeat', 'j']]
    cont = cont.merge(runs, on='label')
    cont['rate'] = cont.arrival.str.split(':').str[1].astype(float)
    scan = cont[(cont.repeat == 0) & (cont.j.fillna(0) == 0)].sort_values('rate')
    ref = None
    initial = ROOT.parent / 'final'
    if (initial / 'raw/macro_runs.jsonl').exists():
        c0 = contention(initial)
        ref = c0[c0.label == 'macro-P2-C24576-K2-Jall-lognormal:2:1-poisson:1-R0']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (metric, title, scale) in zip(axes, [('mean_active', 'Mean active tasks', 1),
                                                 ('above_slots', 'Time with more active tasks than slots (%)', 100),
                                                 ('hit_rate', 'Full-history hit rate (%)', 100)]):
        ax.plot(scan.rate, scan[metric] * scale, marker='o', color=COLORS[0], mec='#333', label='FIFO P2 K2, R0')
        if ref is not None and not ref.empty:
            ax.axhline(ref[metric].iloc[0] * scale, color=COLORS[1], ls='--', lw=1.2, label='initial run (M4), R0 at 1/min')
        main_rate = float(MAIN_ARRIVAL.split(':')[1]) if ':' in MAIN_ARRIVAL else np.nan
        if np.isfinite(main_rate):
            ax.axvline(main_rate, color='#777', ls=':', lw=1)
        ax.set(title=title, xlabel='offered load (tasks/min, scaled arrivals)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('F16  Arrival-rate calibration on this machine (pilot runs, not part of the sweep)', x=.01, ha='left', weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .93])
    save(fig, 'f16_calibration', f'Dotted line: the chosen main rate ({MAIN_ARRIVAL}). Pilot runs are in {pilot}.')
    return cont.sort_values(['repeat', 'j', 'rate']), ref


# ---------------------------------------------------------------- report

def live_dir(split):
    """The live capture behind this directory's workload: its own raw/live_<split>, or, for a sweep
    that reused frozen workloads (WORKLOADS=...), the capture the workload file names."""
    own = ROOT / f'raw/live_{split}'
    workload = ROOT / ('workload_eval.json' if split == 'test' else 'workload_dev.json')
    if not own.exists() and workload.exists():
        source = json.loads(workload.read_text()).get('source_dir')
        if source and Path(source).exists():
            return Path(source)
    return own


def live_records(split):
    path = live_dir(split)
    return [json.loads(p.read_text()) for p in sorted(path.glob('task_*.json'))] if path.exists() else []


def meta():
    rows = read_jsonl('runs_meta.jsonl')
    return rows


@section
def load_startup():
    """One row per llama-server launch, parsed from logs/server-*.log."""
    rows = startup_rows(ROOT)
    if not rows:
        return None
    write_csv(rows, ROOT / 'summary/startup.csv')
    frame = pd.DataFrame(rows)
    frame = frame[frame.P.notna() & frame.load_s.notna()].copy()
    frame['kv_unified'] = frame.kv_unified.eq(True)
    frame['server'] = frame.apply(server_name, axis=1)
    return frame


def power_text(meta_rows):
    states = meta_rows.get('power', pd.Series(dtype=str)).dropna().str.split('\n').str[0]
    if states.empty:
        return 'not recorded'
    ac, battery = states.str.contains('AC Power').sum(), states.str.contains('Battery Power').sum()
    text = f'AC power for {ac} of {len(states)} recorded launches'
    if battery and (ROOT / 'raw/interrupted').exists():
        text += f'; the {battery} battery launch was the run stopped by hand, quarantined in `raw/interrupted/` and re-run on AC'
    elif battery:
        text += f'; {battery} on battery'
    return text


def machine_text(meta_rows):
    """Machine row of the setup table from the platform recorded with each run (older runs lack it)."""
    recorded = [p for p in meta_rows.get('platform', pd.Series(dtype=object)).dropna() if isinstance(p, dict)]
    if not recorded:
        # Runs before platform logging all come from the M4; never describe the machine generating the report.
        return 'arm64 macOS 15.3, Apple M4, 16 GB unified memory'
    p = recorded[-1]
    host = f"EC2 {p['instance_type']}" if p.get('instance_type') else f"{p['system']} {p['release']}"
    gpus = p.get('gpus') or []
    gpu = f", {gpus[0]['name']} ({gpus[0]['memory.total']}, driver {gpus[0]['driver_version']})" if gpus else ''
    cards = {g['uuid'] for r in recorded for g in (r.get('gpus') or [])}
    if len(cards) > 1:
        gpu += f'; runs used {len(cards)} physical GPUs (instance restarts), see runs_meta.jsonl'
    return f"{host}, {p.get('cpu_model') or p['machine']}, {p['cpus']} logical CPUs, {p['memory_gib']} GiB memory{gpu}"


def backend_text(meta_rows):
    commands = meta_rows.get('server_command', pd.Series(dtype=object)).dropna()
    cpu = any('--no-kv-offload' in c for c in commands if isinstance(c, list))
    platforms = [p for p in meta_rows.get('platform', pd.Series(dtype=object)).dropna() if isinstance(p, dict)]
    if cpu:
        return 'CPU only (no GPU offload)'
    return 'CUDA' if any(p.get('gpus') for p in platforms) else 'Metal'


def report(live, micro, cache, macro, parts, memory, gap_tab, arrival_tab, fairness, startup=None,
           eviction=None, context=None, demand=None, admission=None, calibration=None):
    lines = []
    add = lines.append
    noise = dev_summary = None
    meta_rows = meta()
    power = sorted({(m or '').split('\n')[0] for m in meta_rows.get('power', pd.Series(dtype=str)).dropna()})
    commits = sorted(set(meta_rows.get('git_commit', pd.Series(dtype=str)).dropna()))
    add('# Phase 1 Report — Memory-Aware Agent Edge Serving (Option 1)\n')
    add(f'_Generated {datetime.now().isoformat(timespec="minutes")} from `{ROOT}` by `phase1.final`. '
        'All numbers come from the raw logs in this directory._\n')
    add('## 1. Question and baseline\n')
    add('Multi-step agents alternate model calls with tool calls and grow their context at every step. On a memory-'
        'constrained edge machine the KV cache that holds this context is the scarce resource. Phase 1 asks: '
        '**under a fixed KV memory budget, how do request concurrency, the number of server slots, the number of '
        'simultaneously active agent tasks, tool idle time and arrival rate affect completed agent work, tail latency, '
        'memory and fairness — and what is the dominant bottleneck?**\n')
    add('The baseline is a FIFO queue with fixed concurrency K in front of one llama-server: every ready agent step '
        '(first step or continuation after a tool) is dispatched in enqueue order; tool waits do not hold K. '
        'An optional task-level admission limit J caps how many agent tasks are active at once (waiting counts toward latency).\n')
    add('## 2. Setup\n')
    add(table(['Item', 'Value'], [
        ['Machine', machine_text(meta_rows)],
        ['Runtime', f'llama.cpp llama-server build 11669 (commit 620d2225c), {backend_text(meta_rows)}, flash-attn, '
                    'continuous batching'],
        ['Model', 'Qwen3-4B Q4_K_M (sha256 7485fe6f…), f16 KV, thinking disabled; fixed in all comparisons'],
        ['KV budget', '--ctx-size tokens in total; partitioned per slot unless "u" (unified pool); f16 KV = 144 KiB/token'],
        ['Agent benchmark', 'τ²-bench v1.0.1 Retail (official tools, orchestrator, ENV evaluator)'],
        ['User simulator', 'DeepSeek V4 flash (`deepseek-flash`), live capture only; never called during timed replays'],
        ['Power during runs', power_text(meta_rows)],
        ['Code commit(s)', ', '.join(c[:8] for c in commits) or 'n/a'],
    ]))
    add('\n## 3. Workloads (micro → macro)\n')
    test, train = live.get('test', []), live.get('train', [])
    passes = sum(r.get('env_reward') == 1 for r in test)
    if test:
        steps = [len(r['calls']) for r in test]
        maxlen = [max(len(c['prompt_tokens']) for c in r['calls']) for r in test if r['calls']]
        add(f'**Real application (live).** {len(test)} eval tasks (test split) and {len(train)} dev tasks (train split), '
            'sampled with seed 291 and kept regardless of outcome, were executed end-to-end: local Qwen3-4B agent, '
            f'official Retail tools and DeepSeek user simulator. Eval tasks take {min(steps)}–{max(steps)} agent calls '
            f'(median {np.median(steps):.0f}); the longest prompt per task is {min(maxlen):,}–{max(maxlen):,} tokens. '
            f'ENV checks pass on {passes}/{len(test)} eval tasks: the 4B model is weak, and quality is reported, not filtered (F1).'
            + ('' if live_dir('test').is_relative_to(ROOT) else
               f' The capture is reused from `{live_dir("test").parent.parent}` as a frozen workload, so no replay here calls the user simulator.')
            + '\n')
    kind, _, rate = MAIN_ARRIVAL.partition(':')
    arrivals = (f'each repeat\'s 1 task/min Poisson pattern compressed to {rate} tasks/min (`{MAIN_ARRIVAL}`)' if kind == 'scaled'
                else f'Poisson arrivals at {rate} task{"" if rate == "1" else "s"}/min' if kind == 'poisson'
                else f'arrivals `{MAIN_ARRIVAL}`')
    pilot = ROOT.parent / (ROOT.name + '-pilot')
    if kind == 'scaled' and pilot.exists():
        arrivals += f', a rate chosen in a calibration pilot (`{pilot}`) for comparable contention'
    trials = int(macro['runs'].trials.max()) if macro is not None and len(macro['runs']) else 1
    add('**Macro (benchmark-derived).** Each live trajectory is converted to an append-only replay that keeps the exact '
        'number of steps, per-step output tokens and input growth, and retains tokens actually generated during the '
        f'replay. A run draws N tasks from the frozen eval set (balanced, seeded), with {arrivals}, and controlled '
        'tool delays (default log-normal, median 2 s, σ=1, heavy-tailed). Every configuration sees identical arrivals, '
        'task order and delays for the same repeat index'
        + (f', and every experiment was run {trials} times on these identical inputs (trials)' if trials > 1 else '')
        + '. The dev split is kept separate for selecting baselines.\n')
    add('**Micro (controlled).** Exact token counts with prompt cache disabled; K requests submitted together. '
        '**Cache probe.** Measures continuation cost when an agent\'s slot is retained, idle, overwritten, erased, or saved to SSD and restored.\n')
    add('**Metrics.** Completed tasks/min, P50/P95/P99 task completion time (arrival → last step, admission and queue '
        'included), FIFO and admission waits, failures and their reason, peak server RSS / min system available memory, '
        'prefill tokens that recompute already-seen history ("wasted prefill"), full-history hit rate, and fairness: '
        'slowdown = completion time / (isolated compute time + own tool delay), Jain index of 1/slowdown, and slowdown of '
        'long vs short tasks.\n')

    add('## 4. Results\n')
    if micro is not None and not micro.empty:
        add('### 4.1 Micro: prefill dominates long contexts (F2)\n')
        g = micro.groupby(['length', 'output', 'k'])[['prefill_s', 'roundtrip_s', 'tok_s']].mean().reset_index()
        add(table(['input tokens', 'output', 'K', 'prefill s', 'round trip s', 'aggregate tok/s'],
                  [[f'{r.length:,}', r.output, r.k, fmt(r.prefill_s, 2), fmt(r.roundtrip_s, 2), fmt(r.tok_s, 1)] for r in g.itertuples()]))
        one = g[(g.k == 1) & (g.output == g.output.min())]
        if len(one) > 1:
            rate = (one.length.max() - one.length.min()) / (one.prefill_s.max() - one.prefill_s.min())
            longest = g[(g.k == 1) & (g.output == g.output.max()) & (g.length == g.length.min())].iloc[0]
            decode = longest.output / (longest.roundtrip_s - longest.prefill_s)   # alone, short context
            add(f'\nPrefill runs at roughly {rate:,.0f} tokens/s, so re-processing an 8K history costs '
                f'{fmt(one.prefill_s.max(), 1)} s, as long as generating about {one.prefill_s.max() * decode:,.0f} output tokens '
                f'at the measured {decode:,.0f} tokens/s.\n')
    if cache is not None:
        data, rows = cache
        add('### 4.2 Cache mechanisms (F3)\n')
        g = data.groupby(['length', 'mode']).agg(prefill=('prompt_ms', 'mean'), reuse=('reuse', 'mean')).reset_index()
        add(table(['input', 'condition', 'continuation prefill s', 'input reused'],
                  [[f'{r.length:,}', r.mode, fmt(r.prefill / 1000, 3), fmt(r.reuse * 100, 1, '%')] for r in g.itertuples()]))
        if rows is not None:
            add('\n' + table(['input', 'state MiB', 'save s', 'restore s', 'save+erase+restore+prefill s', 're-prefill s'],
                             [[f'{i:,}', fmt(r.state_mib, 0), fmt(r.save_s, 3), fmt(r.restore_s, 3), fmt(r.mechanism_s, 3), fmt(r.rebuild_s, 2)]
                              for i, r in rows.iterrows()]))
        restore = ROOT / 'restore_validation.json'
        if restore.exists():
            rv = json.loads(restore.read_text())
            add(f'\nRestore correctness: after save → overwrite → restore, 64 greedy tokens identical to the uninterrupted run: '
                f'**{rv["generated_tokens_equal"]}**.\n')
    if macro is not None:
        runs = macro['runs']
        main = main_condition(runs)
        add('### 4.3 Macro: FIFO baselines under a fixed 24K-token KV budget (F4–F6)\n')
        if not main.empty:
            rows_out = []
            for c in order_configs(main.config):
                r = main[main.config == c]
                repeats = f'{len(r)} × {int(r.trials.max())} trials' if r.trials.max() > 1 else len(r)
                rows_out.append([c, repeats, f'{r.completed.sum()}/{r.submitted.sum()}',
                                 f'{fmt(r.jobs_per_min.mean(), 3)} ± {fmt(r.jobs_per_min.std(), 3)}',
                                 fmt(r.p50.mean(), 0), fmt(r.p95.mean(), 0),
                                 fmt(r.wasted_frac.mean() * 100, 1, '%'), fmt(r.hit_rate.mean() * 100, 1, '%'),
                                 fmt(r.slowdown.mean(), 2)])
            add(table(['config', 'repeats', 'completed', 'tasks/min', 'P50 s', 'P95 s', 'wasted prefill', 'history hit', 'mean slowdown'], rows_out))
            add('\nTasks/min is mean ± SD across repeats' + (' (each repeat first averaged over its trials)' if main.trials.max() > 1 else '')
                + '; P50/P95, wasted prefill and hit rate are means of per-run values. '
                'Latency is over completed tasks only, so configurations with failures (P4) look faster than they are: '
                'the tasks that fail are the long ones.\n')
            pivot, ratio = paired_throughput(main)
            if ratio is not None:
                mix = repeat_mix(main)
                cols = list(pivot.columns)
                add(f'**Repeats are paired, not noise.** Each repeat index draws a different seeded mix of 8 tasks, and every '
                    'configuration sees the same mix for the same index. The SD above is therefore mostly workload '
                    f'variation: FIFO P2 K2 J∞ has a median completion time of '
                    + ', '.join(f'{fmt(mix.get(c), 0)} s in R{c}' for c in cols) +
                    '. Throughput per repeat, and relative to FIFO P2 K2 J∞ on the same tasks:\n')
                add(table(['config'] + [f'R{c} tasks/min' for c in cols] + [f'R{c} vs FIFO' for c in cols],
                          [[c] + [fmt(pivot.loc[c, r], 3) for r in cols] + [fmt(ratio.loc[c, r], 2, '×') for r in cols]
                           for c in pivot.index]))
                add('\n' + paired_text(ratio))
            best = main.groupby('config').jobs_per_min.mean().idxmax()
            add(f'\nHighest completed-task throughput on the eval workload: **{best}**. (Phase 2 must re-select the best fixed '
                'baseline on the dev workload; this ranking is descriptive.)\n')
            failed = macro['jobs'][macro['jobs'].status != 'ok']
            if not failed.empty:
                reasons = macro['requests'][macro['requests'].status == 'error'].get('error', pd.Series(dtype=str)).fillna('').str.extract(r'"message":"([^"]{0,90})')[0].dropna()
                add(f'Failures ({len(failed)} tasks across all runs) are kept in the denominators. Reasons seen: '
                    + '; '.join(sorted(set(reasons.str.replace(r"\(\d+ tokens\)", "(N tokens)", regex=True)))[:4] or '; '.join(sorted(failed.status.unique()))) + '.\n')
        if parts is not None:
            add('Mean time per completed task (F5):\n')
            add(table(['config', 'admission wait', 'FIFO wait', 'model round trip', 'tool delay', 'total'],
                      [[c, fmt(r.admission_wait_s, 0), fmt(r.fifo_s, 0), fmt(r.model_s, 0), fmt(r.gap_total_s, 0), fmt(r.latency_s, 0)]
                       for c, r in parts.iterrows()]))
        if memory is not None:
            add('\n### 4.4 Memory (F7)\n')
            add(table(['server', 'theoretical KV GiB', 'peak server RSS GiB', 'min system available GiB'],
                      [[s, fmt(r.kv_gib, 2), fmt(r.rss_gib, 2), fmt(r.avail_gib, 2)] for s, r in memory.iterrows()]))
            add('\nllama-server preallocates the whole KV pool at start, so resident memory is set by the budget, not by '
                'load; "too much work" therefore shows up as slot contention (evicted histories) or as hard context '
                'overflows, not as growing RSS.' + gpu_memory_text(meta()) + '\n')
        if gap_tab is not None:
            add('### 4.5 Idle time between steps (F10)\n')
            add(table(['tool delay', 'tasks/min', 'P50 s', 'P95 s', 'wasted prefill', 'history hit', 'slowdown'],
                      [[i, fmt(r.jobs_per_min, 3), fmt(r.p50, 0), fmt(r.p95, 0), fmt(r.wasted_frac * 100, 1, '%'), fmt(r.hit_rate * 100, 1, '%'), fmt(r.slowdown, 2)]
                       for i, r in gap_tab.iterrows()]))
            add('\n' + sweep_text(gap_tab, 'tool delay'))
        if arrival_tab is not None:
            add('\n### 4.6 Arrival rate (F11)\n')
            add(table(['arrival', 'tasks/min', 'P50 s', 'P95 s', 'wasted prefill', 'history hit', 'slowdown'],
                      [[i, fmt(r.jobs_per_min, 3), fmt(r.p50, 0), fmt(r.p95, 0), fmt(r.wasted_frac * 100, 1, '%'), fmt(r.hit_rate * 100, 1, '%'), fmt(r.slowdown, 2)]
                       for i, r in arrival_tab.iterrows()]))
            add('\n' + sweep_text(arrival_tab, 'arrival rate'))
        if fairness is not None:
            add('\n### 4.7 Fairness (F8)\n')
            add(table(['config', 'short-task slowdown', 'long-task slowdown'],
                      [[c, fmt(r.get('short', np.nan), 2), fmt(r.get('long', np.nan), 2)] for c, r in fairness.iterrows()]))
        dev = ROOT.parent / (ROOT.name + '-dev')
        if (dev / 'raw/macro_runs.jsonl').exists():
            summarize(dev)
            d = pd.read_csv(dev / 'summary/sweep_summary.csv')
            d['j'] = d['j'].fillna(0).astype(int)
            dev_summary = d
            add('\n### 4.8 Dev workload (for Phase 2 baseline selection)\n')
            add(table(['P', 'K', 'J', 'runs', 'completed', 'tasks/min', 'P95 s', 'wasted prefill', 'history hit'],
                      [[int(r.P), int(r.k), int(r.j) or '∞', int(r.runs), f'{int(r.completed)}/{int(r.submitted)}',
                        fmt(r.jobs_per_min_mean, 3), fmt(r.p95_s, 0), fmt(r.wasted_prefill_fraction * 100, 1, '%'),
                        fmt(r.full_history_hit_rate * 100, 1, '%')] for r in d.itertuples()]))
            add('\n' + dev_text(d))
        slots = section(slot_use)(macro)
        if slots is not None:
            tab, prefix = slots
            add('\n### 4.9 How llama.cpp assigns slots (P = 2)\n')
            add(table(['K', 'J', 'requests', 'share on busiest slot', 'full-history hit', 'misses that kept only the shared prefix'],
                      [[int(r.k), int(r.j) or '∞', int(r.requests), fmt(r.busiest_slot * 100, 0, '%'), fmt(r.hit * 100, 1, '%'),
                        fmt(r.prefix_only * 100, 0, '%')] for r in tab.itertuples()]))
            add(f'\nAll Retail tasks start with the same {prefix:,}-token system prompt and tool schema. llama.cpp picks the idle '
                'slot whose cached prompt shares the longest prefix with the new request, and with one request in flight '
                '(K = 1) every slot is idle, so it keeps choosing the same slot: **with K = 1 the second slot is never used**, '
                'and P2 K1 behaves like P1 K1. Two active tasks (J = 2) then overwrite each other in one slot while the other '
                'stays empty. When a history is lost, the shared prefix survives and only the task-specific part is recomputed. '
                'Slot choice is not agent-aware: pinning each task to a slot (`id_slot`) is a cheap first step for Phase 2.\n')
        if startup is not None and not startup.empty:
            add('\n### 4.10 Server startup (cold start)\n')
            warmup = startup[startup.first_request_tokens <= 32]
            rows = []
            for server, g in startup.groupby('server', sort=False):
                w = warmup[warmup.server == server].first_request_ms
                rows.append([server, len(g), f'{fmt(g.load_s.median(), 2)} ({fmt(g.load_s.min(), 2)}–{fmt(g.load_s.max(), 2)})',
                             fmt(g.ready_s.median(), 2), fmt(w.median(), 0) if len(w) else 'n/a'])
            add(table(['server', 'launches', 'model load s, median (min–max)', 'listening after s, median',
                       'warm-up request ms, median'], rows))
            add('\nOne row per `llama-server` launch is in `summary/startup.csv`, parsed from `logs/server-*.log` '
                '(`--cache-ram` is not visible in the log, so those launches are grouped by slots and budget). '
                f'Loading the model takes {fmt(startup.load_s.median(), 2)} s at the median and at most '
                f'{fmt(startup.load_s.max(), 2)} s: the weights are memory-mapped, so a launch whose model file is not '
                'already in the OS page cache (for example the first launch after a reboot) is slower. The first request '
                'after each launch is the bench warm-up (5 prompt + 16 generated tokens). Neither model load nor the '
                'warm-up enters any measurement above, so all results describe a warm server; on an edge device that '
                'starts the server on demand, both would add to the first task.\n')
        if eviction is not None:
            add('\n' + eviction_text(*eviction))
        if context is not None:
            add('\n' + context_text(context))
        if demand is not None and not demand.empty:
            add('\n' + demand_text(demand))
        if admission is not None:
            add('\n' + admission_text(admission))
        if calibration is not None:
            add('\n' + calibration_text(*calibration))
        noise = section(noise_text)(main, macro['trial_runs'])
        if noise:
            add('\n' + noise)
        add('\n## 5. Dominant bottleneck\n')
        add(bottleneck_text(macro, micro, cache, context))
    add('\n## 6. Phase 2 direction\n')
    if macro is not None:
        add(phase2_text(macro, cache, fairness, dev_summary))
    add('## 7. Limitations\n')
    add('- Replay is append-only and benchmark-derived, not a byte-faithful trace; it does not re-execute customer-service semantics. '
        'Completion means "all recorded steps executed", not task success.\n'
        '- 8 tasks per run and 3 repeats (2 for P4·36K and for the tool-delay / arrival sweeps); P95/P99 are descriptive. '
        'Error bars are SD across repeats, which mostly reflects the different task mix of each repeat (§4.3). '
        + ('Run-to-run noise on identical inputs is in §4.16.\n' if noise
           else 'Run-to-run noise on identical inputs is not measured here (`TRIALS` or `SCOPE=noise` in the pipeline measures it).\n') +
        '- Latency of configurations with failures is over completed tasks only.\n'
        '- Each macro run is 8 tasks starting from erased slots (an empty KV cache), so it includes a ramp-up and a drain; '
        'there is no separate steady-state window, and tasks/min includes the drain tail. Server startup is reported in §4.10.\n'
        + environment_text(meta_rows) +
        (f'- The main arrival `{MAIN_ARRIVAL}` was calibrated on this machine; absolute numbers are not comparable '
         'with runs on other machines.\n' if MAIN_ARRIVAL.startswith('scaled') else '') +
        '- The 600 s per-request HTTP deadline is a client limit, not a production SLO.\n'
        '- The user simulator is DeepSeek, not TritonAI; it only shapes the captured trajectories.\n')
    add('## 8. Reproduce\n')
    add(reproduce_text())
    add('Setup, options and the AWS procedure are in the project `README.md`; definitions and validity rules in `docs/DESIGN.md`.\n')
    add('Raw logs: `raw/`; tables: `summary/`; figures: `figures/` (PNG + PDF); pipeline log: `logs/pipeline.log`.\n')
    if NOTES:
        add('\n## Generation notes\n' + '\n'.join(f'- {n}' for n in NOTES))
    (ROOT / 'PHASE1_REPORT.md').write_text('\n'.join(lines))


REF_CONFIG = 'P2·24K K2 J∞'


def admission_text(tab):
    text = '### 4.14 Admission limit (F15)\n\n'
    text += table(['J', 'tasks/min (mean of repeats)', 'P95 s', 'full-history hit', 'server busy', 'short-task slowdown'],
                  [[r.label, fmt(r.tpm, 3), fmt(r.p95, 0), fmt(r.hit * 100, 0, '%'), fmt(r.busy * 100, 0, '%'), fmt(r.short, 2)]
                   for r in tab.itertuples()])
    best = tab.tpm.idxmax()
    text += (f'\nP2 under the main condition, J = 1 with K = 1 and the rest with K = 2. As J grows, the full-history hit rate '
             f'{change(tab.hit, pct=True)}, while the share of each run with the server busy {change(tab.busy, pct=True)}; '
             f'the highest mean throughput is at J = {tab.loc[best, "label"]}.\n')
    return text


def calibration_text(cont, ref):
    text = '### 4.15 Arrival-rate calibration (F16)\n\n'
    text += table(['run', 'tasks/min offered', 'mean active tasks', 'time above slots', 'full-history hit', 'tasks/min completed'],
                  [[f'R{int(r.repeat)} J{int(r.j) or "∞"}', fmt(r.rate, 2), fmt(r.mean_active, 2), fmt(r.above_slots * 100, 0, '%'),
                    fmt(r.hit_rate * 100, 0, '%'), fmt(r.tasks_per_min, 3)] for r in cont.itertuples()])
    if ref is not None and not ref.empty:
        r = ref.iloc[0]
        text += (f'\nTarget from the initial run (M4, FIFO P2 K2 on R0 at 1 task/min): {fmt(r.mean_active, 2)} active tasks, '
                 f'{fmt(r.above_slots * 100, 0, "%")} of the time above the slot count, {fmt(r.hit_rate * 100, 0, "%")} hits. ')
    return text + (f'The main rate `{MAIN_ARRIVAL}` was chosen from these pilot runs, which are not part of the sweep.\n')


def change(values, unit='', digits=3, pct=False):
    """'rises from a to b to c' / 'falls ...' / 'goes from ...' for a short ordered series."""
    v = [x for x in values if pd.notna(x)]
    word = 'rises' if all(b > a for a, b in zip(v, v[1:])) else 'falls' if all(b < a for a, b in zip(v, v[1:])) else 'goes'
    shown = [fmt(x * 100, 0, '%') if pct else fmt(x, digits, unit) for x in v]
    return f'{word} from ' + ' to '.join(shown)


def paired_text(ratio):
    """Best configuration per repeat and where the slot-count task limit lands, from the paired ratios."""
    parts = []
    for r in ratio.columns:
        col = ratio[r].dropna()
        best = col.idxmax()
        near = [c for c in col.index if c != best and col[c] >= col[best] * 0.99]
        parts.append(f'R{r} {best} ({fmt(col[best], 2, "×")}' + (f'; within 1%: {", ".join(near)}' if near else '') + ')')
    text = 'Highest throughput per repeat: ' + '; '.join(parts) + '. '
    if 'P2·24K K2 J2' in ratio.index:
        text += ('A task limit equal to the slot count (P2 K2 J2) reaches '
                 + ', '.join(f'{fmt(ratio.loc["P2·24K K2 J2", r], 2, "×")} FIFO in R{r}' for r in ratio.columns) + '. ')
    return text + ('Throughput counts completed tasks only, so a configuration whose long tasks fail (P4) can look fast '
                   'on the tasks that remain.\n')


def sweep_text(tab, what):
    """Direction of throughput, tail latency, hit rate and slowdown along a one-parameter sweep."""
    if len(tab) < 2:
        return ''
    labels = ' → '.join(str(i) for i in tab.index)
    return (f'Compared on the repeats every {what} has ({labels}). Throughput {change(tab.jobs_per_min)} tasks/min, '
            f'P95 completion time {change(tab.p95, " s", 0)}, the full-history hit rate {change(tab.hit_rate, pct=True)} '
            f'and mean slowdown {change(tab.slowdown, digits=2)}.\n')


def dev_text(d):
    """Best fixed baseline on the dev workload, relative to FIFO P2 K2 J∞."""
    d = d[d.P == 2]
    best = d.loc[d.jobs_per_min_mean.idxmax()]
    fifo = d[(d.k == 2) & (d.j == 0)].jobs_per_min_mean
    text = (f'On the dev tasks the highest-throughput fixed baseline is **P{int(best.P)} K{int(best.k)} J{int(best.j) or "∞"}** '
            f'({fmt(best.jobs_per_min_mean, 3)} tasks/min'
            + (f', {fmt(best.jobs_per_min_mean / fifo.iloc[0], 2, "×")} FIFO P2 K2 J∞' if len(fifo) else '') + '). ')
    k1 = d[d.k == 1].set_index('j').jobs_per_min_mean
    if 0 in k1.index and 2 in k1.index:
        text += (f'With K = 1, a task limit of 2 reaches {fmt(k1[2] / k1[0], 2, "×")} K1 J∞: with one request in flight, '
                 'llama.cpp keeps using a single slot (§4.9).')
    return text + '\n'


def gpu_memory_text(meta_rows):
    """GPU memory in use from logs/machine.log (nvidia-smi), which server RSS does not include."""
    if backend_text(meta_rows) != 'CUDA' or not (ROOT / 'logs/machine.log').exists():
        return ''
    # Samples between server restarts read ~0 MiB; keep those taken while a server held the model.
    used = [u for u in map(int, re.findall(r', (\d+) MiB \|', (ROOT / 'logs/machine.log').read_text())) if u > 1024]
    if not used:
        return ''
    return (f' On a CUDA GPU the weights and the KV pool live in GPU memory, which server RSS does not include: '
            f'nvidia-smi reported {min(used):,}–{max(used):,} MiB in use over the sweep (`logs/machine.log`), '
            'set by the server configuration rather than by load.')


def eviction_text(rate, count):
    rows = [[s] + [f'{fmt(rate.loc[s, b] * 100, 0, "%")} ({int(count.loc[s, b])})' if pd.notna(rate.loc[s, b]) else '–'
                   for b in rate.columns] for s in rate.index]
    labels = [str(b) for b in rate.columns][:-1] + [f'{rate.columns[-1]}+']
    text = '### 4.11 Eviction cliff: hit rate vs active tasks (F12)\n\n'
    text += table(['server'] + [f'{b} active' for b in labels], rows)
    text += ('\nFull-history hit rate of continuations (number of continuations), by how many agent tasks were admitted '
             'and unfinished when the step was dispatched. Pooled over all runs of each server, so it is descriptive. ')
    if 'P2·24K' in rate.index:
        p2 = rate.loc['P2·24K']
        text += (f'With two slots, a returning task finds its history {fmt(p2.get(1) * 100, 0, "%")} of the time when it '
                 f'is alone and {fmt(p2.get(2) * 100, 0, "%")} with two active tasks, but only '
                 f'{fmt(p2.get(3) * 100, 0, "%")} with three, and almost never with more. ')
    text += ('More slots move the cliff to more tasks, at the price of smaller per-slot contexts (§4.3). '
             'Eviction is set by how many tasks overlap relative to the slots, which is what an admission limit controls.\n')
    return text


def context_text(c):
    text = '### 4.12 Context growth and the cost of a miss (F13)\n\n'
    med = c['median']
    rows = [[f'{b.left:,}–{b.right - 1:,}', fmt(med.loc[b].get(True, np.nan), 1), fmt(med.loc[b].get(False, np.nan), 1)]
            for b in med.index]
    text += table(['context tokens', 'median prefill s, history reused', 'median prefill s, history re-prefilled'], rows)
    text += (f'\nP2·24K K2, main condition: {c["continuations"]} continuations, {c["misses"]} misses. A step that reuses its '
             f'history prefills only its new tokens (median {fmt(c["hit_s"], 1)} s overall). A miss usually keeps '
             f'the {c["prefix"]:,}-token shared prefix in the slot and recomputes the task-specific history (median '
             f'{fmt(c["recompute_median"], 0)} tokens), at about {fmt(c["slope"] * 1000, 1)} s per 1,000 tokens under contention. '
             'The cost of each eviction therefore grows roughly linearly as the task\'s context grows, while a hit stays a '
             'matter of its new tokens and of sharing the batch. This is observational: long '
             'contexts are also late steps of long tasks, which overlap more other tasks; a controlled growth sweep is future work.\n')
    return text


def demand_text(tab):
    text = '### 4.13 Memory pressure in tokens (F14)\n\n'
    text += table(['config', 'repeat', 'peak demand (tokens)', 'time above the 24K budget', 'mean active tasks',
                   'full-history hit', 'tasks/min'],
                  [[r.config, f'R{r.repeat}', f'{int(r.peak):,}', fmt(r.over * 100, 0, '%'), fmt(r.mean_active, 1),
                    fmt(r.hit_rate * 100, 0, '%'), fmt(r.jobs_per_min, 3)] for r in tab.itertuples()])
    text += ('\nDemand is the KV the admitted tasks would need to keep their whole histories resident. Resident memory '
             'cannot show this (F7: the pool is preallocated), but the demand can: under FIFO it exceeds the budget whenever '
             'several tasks overlap, and every token above the budget belongs to a history that must be evicted and later '
             're-prefilled. With partitioned slots eviction starts even earlier, as soon as more tasks than slots are active '
             '(F12). With J = 2 the demand never exceeds two histories, which always fit.\n')
    return text


def trials_text(trial_runs):
    """Run-to-run noise when every experiment was run as several trials in this directory."""
    tr = main_condition(trial_runs)
    cell = tr.groupby(['config', 'repeat']).jobs_per_min.agg(['size', 'mean', 'std'])
    cell['cv'] = cell['std'] / cell['mean']
    fifo = tr[tr.config == REF_CONFIG].set_index(['repeat', 'trial']).jobs_per_min
    tr = tr.assign(vs=tr.jobs_per_min.to_numpy() / fifo.reindex(pd.MultiIndex.from_frame(tr[['repeat', 'trial']])).to_numpy())
    rows = []
    for c in order_configs(tr.config):
        cv = cell.loc[c].cv
        paired = []
        for r, g in tr[tr.config == c].groupby('repeat'):
            v = g.vs.dropna()
            if c != REF_CONFIG and len(v):
                side = ' (all above 1)' if (v > 1).all() else ' (all below 1)' if (v < 1).all() else ''
                paired.append(f'R{r} {fmt(v.min(), 2)}–{fmt(v.max(), 2)}×{side}')
        rows.append([c, int(cell.loc[c]['size'].max()), fmt(cv.mean() * 100, 1, '%'), fmt(cv.max() * 100, 1, '%'),
                     '; '.join(paired) or '–'])
    text = '### 4.16 Run-to-run noise (every experiment repeated as trials)\n\n'
    text += table(['config', 'trials per repeat', 'tasks/min CV, mean over repeats', 'max CV', 'vs FIFO per trial, range per repeat'], rows)
    worst = cell.cv.max()
    text += ('\nTrials replay a repeat with identical tasks, arrivals and tool delays, so their spread is run-to-run noise; '
             'the tables above average them first. Each trial restarted the servers and ran the configurations in a rotated '
             f'order. The largest CV of tasks/min is {fmt(worst * 100, 1, "%")}: a paired difference smaller than about '
             f'{fmt(2 * worst * 100, 0, "%")} in a single run would not be distinguishable from noise. A comparison is '
             'consistent when every trial falls on the same side of FIFO.\n')
    return text


def noise_text(main, trial_runs=None):
    """Run-to-run noise: trials in this directory, else from <results>-noise (one repeat replayed)."""
    if trial_runs is not None and trial_runs.trial.nunique() > 1:
        return trials_text(trial_runs)
    root = ROOT.parent / (ROOT.name + '-noise')
    if not (root / 'raw/macro_runs.jsonl').exists():
        return None
    rows = trial_rows(root)
    summary = spread(rows)
    summary['config'] = summary.apply(config_name, axis=1)
    text = '### 4.16 Run-to-run noise (same inputs, repeated)\n\n'
    text += table(['config', 'repeat', 'trials', 'completed', 'tasks/min mean ± SD', 'CV', 'min–max', 'P50 s (CV)',
                   'history hit', 'vs FIFO, per trial'],
                  [[r.config, f'R{int(r.repeat)}', r.trials, f'{int(rows[rows.j == r.j].completed.sum())}/{int(rows[rows.j == r.j].tasks.sum())}',
                    f'{fmt(r.jobs_per_minute_mean, 3)} ± {fmt(r.jobs_per_minute_sd, 3)}', fmt(r.jobs_per_minute_cv * 100, 1, '%'),
                    f'{fmt(r.jobs_per_minute_min, 3)}–{fmt(r.jobs_per_minute_max, 3)}',
                    f'{fmt(r.p50_s_mean, 0)} ({fmt(r.p50_s_cv * 100, 0, "%")})', fmt(r.full_history_hit_rate_mean * 100, 0, '%'),
                    ', '.join(fmt(v, 2, '×') for v in rows[rows.j == r.j].sort_values('trial').vs_fifo)]
                   for r in summary.itertuples()])
    machines = sorted({json.dumps(m.get('platform'), sort_keys=True) for m in
                       [json.loads(s) for s in (root / 'raw/runs_meta.jsonl').read_text().splitlines() if s]}) \
        if (root / 'raw/runs_meta.jsonl').exists() else []
    cv = summary.jobs_per_minute_cv.max()
    text += (f'\nEach trial replays repeat R{int(summary.repeat.iloc[0])} with identical tasks, arrivals and tool delays '
             f'after a server restart, alternating which configuration runs first; trials are in `{root}`. The largest '
             f'coefficient of variation of tasks/min is {fmt(cv * 100, 1, "%")}, so a paired difference within one repeat '
             f'needs to exceed roughly {fmt(2 * cv * 100, 0, "%")} (2 SD) to stand out from run-to-run noise.')
    _, ratio = paired_throughput(main)
    if ratio is not None and np.isfinite(cv):
        small = [f'{c} R{r} ({fmt(ratio.loc[c, r], 2, "×")})' for c in ratio.index for r in ratio.columns
                 if c != REF_CONFIG and pd.notna(ratio.loc[c, r]) and abs(ratio.loc[c, r] - 1) < 2 * cv]
        if small:
            text += ' Paired differences in §4.3 within that band: ' + ', '.join(small) + '.'
    if len(machines) == 1 and machines[0] != 'null':
        text += f' Trials ran on: `{machines[0]}`; compare only with runs from the same machine.'
    return text + '\n'


def paired_throughput(main):
    """Tasks/min per repeat (each repeat is a different seeded task mix) and the ratio to FIFO P2 K2 J∞ in the same repeat."""
    pivot = main.pivot_table(index='config', columns='repeat', values='jobs_per_min').reindex(order_configs(main.config))
    if REF_CONFIG not in pivot.index:
        return pivot, None
    ratio = pivot.div(pivot.loc[REF_CONFIG], axis=1)
    return pivot, ratio


def repeat_mix(main):
    """Median completion time of the reference config per repeat, as a proxy for how long that repeat's tasks are."""
    ref = main[main.config == REF_CONFIG].set_index('repeat')
    return ref.p50.round(0).to_dict()


def shared_prefix_tokens():
    path = ROOT / 'workload_eval.json'
    if not path.exists():
        return None
    firsts = [t['steps'][0]['delta'] for t in json.loads(path.read_text())['tasks']]
    n = 0
    while all(len(f) > n for f in firsts) and len({f[n] for f in firsts}) == 1:
        n += 1
    return n


def slot_use(macro):
    """Share of requests on the busiest slot, and misses that kept only the prefix shared by all tasks."""
    req = macro['requests']
    req = req[(req.P == 2) & (req.status == 'ok') & req.slot.notna()]
    if req.empty:
        return None
    prefix = shared_prefix_tokens() or 0
    rows = []
    for (k, j), g in req.groupby(['k', 'j']):
        busiest = g.groupby('label').slot.agg(lambda s: s.value_counts(normalize=True).iloc[0]).mean()
        miss = g[g.full_history_hit == 0]
        rows.append(dict(k=k, j=j, requests=len(g), busiest_slot=busiest,
                         hit=g.full_history_hit.mean(),
                         prefix_only=(miss.cache_n <= prefix + 64).mean() if len(miss) else np.nan))
    return pd.DataFrame(rows), prefix


def run_costs(macro, configs):
    """Per run of the given configurations (main condition): GPU busy share and the share of model time
    (prefill + decode) spent re-prefilling history the task had already computed."""
    req, jobs = macro['requests'], macro['jobs']
    rows = []
    for r in main_condition(macro['trial_runs']).itertuples():
        if r.config not in configs:
            continue
        q, j = req[(req.label == r.label) & (req.status == 'ok')], jobs[jobs.label == r.label]
        model_s = (q.prompt_ms.sum() + q.predicted_ms.sum()) / 1000
        reprefill_s = (q.prompt_ms * q.wasted_prefill / q.prompt_n.where(q.prompt_n > 0)).sum() / 1000
        rows.append(dict(config=r.config, repeat=r.repeat, busy=busy_fraction(req[req.label == r.label], j),
                         reprefill_share=reprefill_s / model_s if model_s else np.nan))
    return pd.DataFrame(rows)


def bottleneck_text(macro, micro, cache, context=None):
    main = main_condition(macro['runs'])
    if main.empty:
        return 'Main-condition runs are missing; see the tables above.\n'
    g = main.groupby('config')[['wasted_frac', 'hit_rate', 'jobs_per_min', 'p95']].mean()
    worst, best = g.wasted_frac.idxmax(), g.wasted_frac.idxmin()
    corr = g[['wasted_frac', 'jobs_per_min']].corr().iloc[0, 1] if len(g) > 2 else np.nan
    text = (f'Across baselines the share of prefill tokens spent recomputing history ranges from {fmt(g.wasted_frac.min() * 100, 1, "%")} '
            f'({best}) to {fmt(g.wasted_frac.max() * 100, 1, "%")} ({worst}); full-history hit rates range '
            f'{fmt(g.hit_rate.min() * 100, 0, "%")}–{fmt(g.hit_rate.max() * 100, 0, "%")}. ')
    if np.isfinite(corr):
        text += f'Across configurations, wasted-prefill share and completed tasks/min correlate at r = {corr:.2f}. '
    costs = section(run_costs)(macro, [REF_CONFIG, 'P2·24K K2 J2'])
    if costs is not None and not costs.empty:
        c = costs.groupby('config')[['busy', 'reprefill_share']].mean()
        if REF_CONFIG in c.index:
            text += (f'Under FIFO, re-prefill takes {fmt(c.loc[REF_CONFIG, "reprefill_share"] * 100, 0, "%")} of model time '
                     f'(prefill + decode) and the server is busy {fmt(c.loc[REF_CONFIG, "busy"] * 100, 0, "%")} of each run')
            if 'P2·24K K2 J2' in c.index:
                text += f'; with J = 2 the server is busy {fmt(c.loc["P2·24K K2 J2", "busy"] * 100, 0, "%")} of the time'
            text += '. '
    pivot, ratio = paired_throughput(main)
    if ratio is not None and 'P2·24K K2 J2' in ratio.index:
        mix = repeat_mix(main)
        most, least = max(mix, key=mix.get), min(mix, key=mix.get)
        gain = ratio.loc['P2·24K K2 J2', most]
        text += (f'On the same tasks, a task limit equal to the slot count (P2 K2 J2) reaches {fmt(gain, 2, "×")} FIFO in the '
                 f'most contended repeat (R{most}, FIFO median completion {fmt(mix[most], 0)} s) and '
                 f'{fmt(ratio.loc["P2·24K K2 J2", least], 2, "×")} in the least contended (R{least}). ')
        slope = context.get('slope') if context else np.nan
        cost = f' ({fmt(slope * 1000, 1)} s per 1,000 tokens, §4.12)' if np.isfinite(slope) else ''
        if gain > 1.1:
            text += (f'Where tasks overlap, preventing eviction pays off, because a lost history is expensive to rebuild{cost}. ')
        elif gain < 0.95:
            text += (f'Even where tasks overlap most, the limit costs throughput: a lost history is cheap to rebuild{cost}, '
                     'while the admitted tasks spend much of their time in tool calls and leave the server idle. ')
    text += ('Resident memory stays flat because the pool is preallocated, so the memory limit acts through the '
             'number and size of slots: more slots per budget means smaller per-slot contexts (overflow failures for long '
             'Retail histories), fewer slots means agents evict one another\'s histories while their tools run. ')
    p4 = main[main.P == 4].groupby('config')[['completed', 'submitted']].sum()
    if {'P4·24K K4 J∞', 'P4·36K K4 J∞'} <= set(p4.index):
        done = p4.completed / p4.submitted
        text += (f'Raising the budget from 24K to 36K tokens at P4 raises completed tasks from '
                 f'{fmt(done["P4·24K K4 J∞"] * 100, 0, "%")} to {fmt(done["P4·36K K4 J∞"] * 100, 0, "%")}.')
    return text + '\n'


def phase2_text(macro, cache, fairness, dev):
    """Findings that set the Phase 2 baselines and mechanism, computed from this directory."""
    items = []
    if dev is not None and not dev.empty:
        d = dev[dev.P == 2]
        best = d.loc[d.jobs_per_min_mean.idxmax()]
        name = f'P2 K{int(best.k)} J{int(best.j) or "∞"}'
        fifo = d[(d.k == 2) & (d.j == 0)].jobs_per_min_mean
        rel = f', {fmt(best.jobs_per_min_mean / fifo.iloc[0], 2, "×")} FIFO' if len(fifo) else ''
        items.append(f'**The best fixed baseline on the dev workload is {name}** ({fmt(best.jobs_per_min_mean, 3)} tasks/min{rel}). '
                     'Phase 2 has to beat it, not only FIFO, on the eval workload.')
    else:
        name = 'the best fixed K/J'
    if fairness is not None and {'P2·24K K2 J2', REF_CONFIG} <= set(fairness.index):
        items.append(f'**A tight task limit is unfair to short tasks**: their mean slowdown is '
                     f'{fmt(fairness.loc["P2·24K K2 J2", "short"], 2)} under P2 K2 J2 against '
                     f'{fmt(fairness.loc[REF_CONFIG, "short"], 2)} under FIFO, because they wait for admission behind long tasks (§4.7).')
    slots = section(slot_use)(macro)
    if slots is not None:
        tab = slots[0].set_index(['k', 'j'])
        if (1, 0) in tab.index:
            items.append(f'**Slot placement is not agent-aware** (§4.9): with one request in flight, '
                         f'{fmt(tab.loc[(1, 0), "busiest_slot"] * 100, 0, "%")} of requests go to the same slot. '
                         'Slot affinity per task is a cheap first mechanism.')
    if cache is not None and cache[1] is not None:
        r = cache[1].iloc[-1]
        length = cache[1].index[-1]
        verdict = ('so checkpointing a slot is cheaper than rebuilding it' if r.save_s + r.restore_s < r.rebuild_s else
                   'so writing a checkpoint costs more than rebuilding, and a checkpoint helps only if it is written '
                   'while the task waits on a tool, off the critical path')
        items.append(f'**Retention by save/restore**: saving one slot\'s {length:,}-token state takes {fmt(r.save_s, 2)} s and '
                     f'restoring it {fmt(r.restore_s, 2)} s, against {fmt(r.rebuild_s, 2)} s to re-prefill (§4.2), {verdict}.')
    main = main_condition(macro['runs'])
    _, ratio = paired_throughput(main)
    if ratio is not None and 'P2·24K+RAM3G K2 J∞' in ratio.index:
        items.append(f'**llama.cpp\'s host prompt cache** (`--cache-ram 3072`) reaches '
                     f'{fmt(ratio.loc["P2·24K+RAM3G K2 J∞"].mean(), 2, "×")} FIFO on average: a fixed cache does not address eviction.')
    text = 'The measured findings that set the Phase 2 baselines and mechanism:\n\n' + ''.join(f'{i}. {t}\n' for i, t in enumerate(items, 1))
    return text + ('\nThe Phase 2 mechanism is therefore an adaptive admission and retention policy: (1) admit agent tasks '
                   'while the KV budget can hold their growing histories and while the server would otherwise idle during '
                   'tool calls, (2) give ready continuations priority with per-task slot affinity, with aging so neither '
                   'short nor long tasks starve, and (3) when a tool call is long, decide between keeping the slot, '
                   f'checkpointing it, or evicting it. Required baselines: FIFO, {name} (best fixed on dev), and '
                   '`--cache-ram` as the simple fixed cache policy; plus queue-only and cache-only ablations, policy '
                   'overhead, and low-load / zero-delay cases where little benefit is expected.\n')


def environment_text(meta_rows):
    """Limitations bullet about the machine the runs came from."""
    if backend_text(meta_rows) == 'CUDA':
        log = ROOT / 'logs/machine.log'
        clocks = [int(m) for m in re.findall(r'GPU-[^,]+, (\d+) MHz', log.read_text())] if log.exists() else []
        span = f'; its SM clock ranged {min(clocks):,}–{max(clocks):,} MHz (`logs/machine.log`)' if clocks else ''
        return ('- One dedicated EC2 GPU instance with nothing else running; the GPU ran at its board power cap under load'
                f'{span}. Server RSS excludes GPU memory (§4.4).\n')
    return '- One laptop: thermal state and background processes are not fully controlled; RSS under-counts Metal allocations.\n'


def reproduce_text():
    log = ROOT / 'logs/pipeline.log'
    starts = [s for s in log.read_text().splitlines() if '=== pipeline start' in s] if log.exists() else []
    settings = starts[-1].split('commit ', 1)[-1] if starts else 'see logs/pipeline.log'
    if ROOT.name == 'final':
        run = 'HOURS=9 nohup caffeinate -ims bash scripts/overnight.sh > /dev/null 2>&1 &   # resumable full pipeline'
    else:
        run = f'# pipeline settings for this directory (logs/pipeline.log): commit {settings}'
    return (f'```bash\n{run}\n.venv/bin/python -m phase1.final --results-dir {ROOT} --main-arrival {MAIN_ARRIVAL}'
            '   # figures + this report\n```\n')


def main():
    global ROOT, FIG, MAIN_GAP, MAIN_ARRIVAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='runs/final')
    parser.add_argument('--main-gap', default=MAIN_GAP)
    parser.add_argument('--main-arrival', default=MAIN_ARRIVAL)
    args = parser.parse_args()
    ROOT, MAIN_GAP, MAIN_ARRIVAL = Path(args.results_dir), args.main_gap, args.main_arrival
    FIG = ROOT / 'figures'
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .15, 'pdf.fonttype': 42})
    live = {'test': live_records('test'), 'train': live_records('train')}
    fig_workload(live)
    micro = fig_micro()
    cache = fig_cache()
    macro = section(load_macro)()
    parts = memory = gap_tab = arrival_tab = fairness = eviction = context = demand = admission = None
    if macro is not None:
        fig_main(macro)
        parts = fig_breakdown(macro)
        fig_bottleneck(macro)
        memory = fig_memory(macro)
        gap_tab = fig_sweep(macro, 'gap_mode', ['zero', MAIN_GAP, 'fixed:15'], ['none', 'log-normal med 2 s', 'fixed 15 s'],
                            'f10_tool_delay', 'F10  Effect of tool idle time (P2 K2)', 'Same arrivals and tasks; only the delay between steps changes.')
        if ':' in MAIN_ARRIVAL:
            kind, rate = MAIN_ARRIVAL.split(':')
            rates = [float(rate) / 2, float(rate), float(rate) * 2]   # the pipeline sweeps half and double the main rate
            pattern = 'one pattern per repeat, compressed' if kind == 'scaled' else 'Poisson arrivals'
            arrival_tab = fig_sweep(macro, 'arrival', [f'{kind}:{rates[0]:g}', MAIN_ARRIVAL, f'{kind}:{rates[2]:g}'],
                                    [f'{r:g}/min' for r in rates], 'f11_arrival', 'F11  Effect of arrival rate (P2 K2)',
                                    f'{pattern}; tool delay log-normal median 2 s.')
        fairness = fig_fairness(macro)
        fig_timeline(macro)
        eviction = fig_eviction(macro)
        context = fig_context_cost(macro)
        demand = fig_kv_demand(macro)
        admission = fig_admission(macro)
    startup = load_startup()
    calibration = fig_calibration()
    report(live, micro, cache, macro, parts, memory, gap_tab, arrival_tab, fairness, startup,
           eviction, context, demand, admission, calibration)
    print('Wrote', ROOT / 'PHASE1_REPORT.md', 'and figures in', FIG, '| notes:', NOTES or 'none')


if __name__ == '__main__':
    main()
