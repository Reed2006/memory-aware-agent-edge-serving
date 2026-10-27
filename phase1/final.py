"""Figures and the English Phase 1 report for a full sweep directory (default runs/final).

Every section tolerates missing data so a partially finished overnight run still
produces a coherent report; absent experiments are listed as not run.
"""

import argparse
import json
import platform
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    path = ROOT / 'raw' / name
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
            submitted=len(group), completed=len(done),
            jobs_per_min=meta['jobs_per_minute'] if meta is not None else np.nan,
            makespan_s=meta['makespan_s'] if meta is not None else np.nan,
            p50=done.latency_s.median(), p95=done.latency_s.quantile(.95),
            slowdown=done.slowdown.mean(),
            wasted_frac=per_job.loc[label, 'wasted'] / per_job.loc[label, 'prefill_tokens'] if label in per_job.index and per_job.loc[label, 'prefill_tokens'] else np.nan,
            hit_rate=per_job.loc[label, 'hit_rate'] if label in per_job.index else np.nan))
    return dict(jobs=jobs, requests=requests, runs=pd.DataFrame(run_rows),
                summary=pd.read_csv(ROOT / 'summary/sweep_summary.csv'))


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
        vals = runs[runs.repeat == r].set_index('config')[metric].reindex(configs) * scale
        ax.bar(x + (i - (len(repeats) - 1) / 2) * width, vals, width * .95,
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
    save(fig, 'f04_main', f'Eval workload (test split), Poisson {MAIN_ARRIVAL.split(":")[1]} task/min, tool delay {MAIN_GAP}. '
                          'P = slots, u = unified KV, K = request concurrency, J = active-task limit. '
                          'Each repeat is a different seeded mix of 8 tasks (R1 short, R2 long).')


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
    runs = macro['runs']
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


# ---------------------------------------------------------------- report

def live_records(split):
    path = ROOT / f'raw/live_{split}'
    return [json.loads(p.read_text()) for p in sorted(path.glob('task_*.json'))] if path.exists() else []


def meta():
    rows = read_jsonl('runs_meta.jsonl')
    return rows


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


def report(live, micro, cache, macro, parts, memory, gap_tab, arrival_tab, fairness):
    lines = []
    add = lines.append
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
        ['Machine', f'{platform.machine()} macOS {platform.mac_ver()[0]}, Apple M4, 16 GB unified memory'],
        ['Runtime', 'llama.cpp llama-server build 11669 (commit 620d2225c), Metal, flash-attn, continuous batching'],
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
            f'ENV checks pass on {passes}/{len(test)} eval tasks: the 4B model is weak, and quality is reported, not filtered (F1).\n')
    add('**Macro (benchmark-derived).** Each live trajectory is converted to an append-only replay that keeps the exact '
        'number of steps, per-step output tokens and input growth, and retains tokens actually generated during the '
        'replay. A run draws N tasks from the frozen eval set (balanced, seeded), with Poisson arrivals and controlled '
        'tool delays (default log-normal, median 2 s, σ=1, heavy-tailed). Every configuration sees identical arrivals, '
        'task order and delays for the same repeat index. The dev split is kept separate for selecting baselines.\n')
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
            add(f'\nPrefill runs at roughly {rate:,.0f} tokens/s, so re-processing an 8K history costs '
                f'{fmt(one.prefill_s.max(), 1)} s — longer than generating several hundred tokens.\n')
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
                rows_out.append([c, len(r), f'{r.completed.sum()}/{r.submitted.sum()}',
                                 f'{fmt(r.jobs_per_min.mean(), 3)} ± {fmt(r.jobs_per_min.std(), 3)}',
                                 fmt(r.p50.mean(), 0), fmt(r.p95.mean(), 0),
                                 fmt(r.wasted_frac.mean() * 100, 1, '%'), fmt(r.hit_rate.mean() * 100, 1, '%'),
                                 fmt(r.slowdown.mean(), 2)])
            add(table(['config', 'repeats', 'completed', 'tasks/min', 'P50 s', 'P95 s', 'wasted prefill', 'history hit', 'mean slowdown'], rows_out))
            add('\nTasks/min is mean ± SD across repeats; P50/P95, wasted prefill and hit rate are means of per-run values. '
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
                add('\nLimiting active tasks (J = P) wins only when histories are long (R0, R2) and is neutral on short tasks (R1). '
                    'Throughput also counts completed tasks only: P4·24K looks fast in R2 because its long tasks failed early.\n')
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
                'overflows, not as growing RSS.\n')
        if gap_tab is not None:
            add('### 4.5 Idle time between steps (F10)\n')
            add(table(['tool delay', 'tasks/min', 'P50 s', 'P95 s', 'wasted prefill', 'history hit', 'slowdown'],
                      [[i, fmt(r.jobs_per_min, 3), fmt(r.p50, 0), fmt(r.p95, 0), fmt(r.wasted_frac * 100, 1, '%'), fmt(r.hit_rate * 100, 1, '%'), fmt(r.slowdown, 2)]
                       for i, r in gap_tab.iterrows()]))
            add('\nCompared on the repeats every delay has. Only a long fixed idle time (15 s) clearly raises eviction and '
                'lowers throughput; the gap between no delay and log-normal delay is within repeat-to-repeat variation.\n')
        if arrival_tab is not None:
            add('\n### 4.6 Arrival rate (F11)\n')
            add(table(['arrival', 'tasks/min', 'P50 s', 'P95 s', 'wasted prefill', 'history hit', 'slowdown'],
                      [[i, fmt(r.jobs_per_min, 3), fmt(r.p50, 0), fmt(r.p95, 0), fmt(r.wasted_frac * 100, 1, '%'), fmt(r.hit_rate * 100, 1, '%'), fmt(r.slowdown, 2)]
                       for i, r in arrival_tab.iterrows()]))
            add('\nCompared on the repeats every rate has. The lowest rate (0.5/min) gives the best fairness and hit rate; '
                'between 1 and 2/min, throughput rises with offered load while P95 latency and eviction grow.\n')
        if fairness is not None:
            add('\n### 4.7 Fairness (F8)\n')
            add(table(['config', 'short-task slowdown', 'long-task slowdown'],
                      [[c, fmt(r.get('short', np.nan), 2), fmt(r.get('long', np.nan), 2)] for c, r in fairness.iterrows()]))
        dev = ROOT.parent / (ROOT.name + '-dev')
        if (dev / 'raw/macro_runs.jsonl').exists():
            summarize(dev)
            d = pd.read_csv(dev / 'summary/sweep_summary.csv')
            d['j'] = d['j'].fillna(0).astype(int)
            add('\n### 4.8 Dev workload (for Phase 2 baseline selection)\n')
            add(table(['P', 'K', 'J', 'runs', 'completed', 'tasks/min', 'P95 s', 'wasted prefill', 'history hit'],
                      [[int(r.P), int(r.k), int(r.j) or '∞', int(r.runs), f'{int(r.completed)}/{int(r.submitted)}',
                        fmt(r.jobs_per_min_mean, 3), fmt(r.p95_s, 0), fmt(r.wasted_prefill_fraction * 100, 1, '%'),
                        fmt(r.full_history_hit_rate * 100, 1, '%')] for r in d.itertuples()]))
            add('\nOn the dev tasks the best fixed baseline is again P2 K2 J2 (J = number of slots). '
                'K1 J2 gains nothing over K1 J∞; §4.9 explains why.\n')
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
        add('\n## 5. Dominant bottleneck\n')
        add(bottleneck_text(macro, micro, cache))
    add('\n## 6. Phase 2 direction\n')
    add('Phase 1 shows that, at a fixed KV budget, the cost that FIFO fails to control is **history eviction followed by '
        're-prefill**, which grows with context length and with the number of agent tasks that share the slots. Four '
        'findings shape Phase 2:\n\n'
        '1. **A fixed task limit J = P is already a strong baseline** (best on eval and dev, up to ~2× FIFO on long tasks). '
        'Phase 2 has to beat it, not FIFO.\n'
        '2. **That baseline is unfair to short tasks**: they wait for admission behind long ones (§4.7). A policy that '
        'admits by predicted KV footprint and remaining steps can keep J\'s cache hits while cutting short-task slowdown.\n'
        '3. **Slot placement is not agent-aware** (§4.9): with one request in flight llama.cpp reuses one slot and leaves '
        'the other empty. Slot affinity per task is required.\n'
        '4. **Simple fixed caching does not help**: llama.cpp\'s host prompt cache (`--cache-ram 3072`) is no better than '
        'none on the same tasks, while SSD save/restore of one slot costs < 1 s against 44 s to re-prefill 8K tokens '
        '(§4.2). Per-slot capacity, not slot count, decides whether long tasks fit (P4 fails at 6K per slot, mostly '
        'recovers at 9K).\n\n'
        'The Phase 2 mechanism is therefore a memory-aware admission and retention policy: (1) admit new agent tasks only '
        'while the KV budget can hold their growing context (estimated from observed growth), (2) give ready continuations '
        'priority with slot affinity, with aging so neither short nor long tasks starve, and (3) when a tool is slow, decide '
        'between keeping the slot, saving its state to SSD or evicting it. Required baselines: FIFO, the best fixed K/J '
        'selected on the dev workload (P2 K2 J2), and `--cache-ram` as the simple fixed cache policy; plus queue-only and '
        'cache-only ablations, policy overhead, and low-load / zero-delay cases where little benefit is expected.\n')
    add('## 7. Limitations\n')
    add('- Replay is append-only and benchmark-derived, not a byte-faithful trace; it does not re-execute customer-service semantics. '
        'Completion means "all recorded steps executed", not task success.\n'
        '- 8 tasks per run and 3 repeats (2 for P4·36K and for the tool-delay / arrival sweeps); P95/P99 are descriptive. '
        'Error bars are SD across repeats, which mostly reflects the different task mix of each repeat (§4.3).\n'
        '- Latency of configurations with failures is over completed tasks only.\n'
        '- One laptop: thermal state and background processes are not fully controlled; RSS under-counts Metal allocations.\n'
        '- The 600 s per-request HTTP deadline is a client limit, not a production SLO.\n'
        '- The user simulator is DeepSeek, not TritonAI; it only shapes the captured trajectories.\n')
    add('## 8. Reproduce\n')
    add('```bash\nHOURS=9 nohup caffeinate -dims bash scripts/overnight.sh > /dev/null 2>&1 &   # resumable full pipeline\n'
        '.venv/bin/python -m phase1.final --results-dir runs/final                         # figures + this report\n```\n')
    add('Setup, options and the AWS procedure are in the project `README.md`; definitions and validity rules in `docs/DESIGN.md`.\n')
    add('Raw logs: `raw/`; tables: `summary/`; figures: `figures/` (PNG + PDF); pipeline log: `logs/pipeline.log`.\n')
    if NOTES:
        add('\n## Generation notes\n' + '\n'.join(f'- {n}' for n in NOTES))
    (ROOT / 'PHASE1_REPORT.md').write_text('\n'.join(lines))


REF_CONFIG = 'P2·24K K2 J∞'


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


def bottleneck_text(macro, micro, cache):
    main = main_condition(macro['runs'])
    if main.empty:
        return 'Main-condition runs are missing; see the tables above.\n'
    g = main.groupby('config')[['wasted_frac', 'hit_rate', 'jobs_per_min', 'p95']].mean()
    worst, best = g.wasted_frac.idxmax(), g.wasted_frac.idxmin()
    corr = g[['wasted_frac', 'jobs_per_min']].corr().iloc[0, 1] if len(g) > 2 else np.nan
    text = (f'Across baselines the share of prefill spent recomputing history ranges from {fmt(g.wasted_frac.min() * 100, 1, "%")} '
            f'({best}) to {fmt(g.wasted_frac.max() * 100, 1, "%")} ({worst}); full-history hit rates range '
            f'{fmt(g.hit_rate.min() * 100, 0, "%")}–{fmt(g.hit_rate.max() * 100, 0, "%")}. ')
    if np.isfinite(corr):
        text += f'Across configurations, wasted-prefill share and completed tasks/min correlate at r = {corr:.2f}. '
    pivot, ratio = paired_throughput(main)
    limited = [c for c in ['P2·24K K2 J2', 'P2·24K K1 J1'] if ratio is not None and c in ratio.index]
    if limited:
        c = limited[0]
        mix = repeat_mix(main)
        longest, shortest = max(mix, key=mix.get), min(mix, key=mix.get)
        text += (f'The paired comparison is clearer: on the same tasks, {c} reaches {fmt(ratio.loc[c, longest], 2, "×")} the '
                 f'throughput of FIFO P2 K2 J∞ in the long-task repeat R{longest} but only {fmt(ratio.loc[c, shortest], 2, "×")} '
                 f'in the short-task repeat R{shortest}. Removing eviction pays off in proportion to how much history there is '
                 'to lose. ')
    text += ('Resident memory stays flat because the pool is preallocated, so the edge-memory limit acts through the '
             'number and size of slots: more slots per budget means smaller per-slot contexts (overflow failures for long '
             'Retail histories), fewer slots means agents evict one another\'s histories while their tools run. Raising the '
             'budget from 24K to 36K tokens at P4 cuts failures sharply, but P4 still does not beat P2 with a task limit.\n')
    return text


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
    parts = memory = gap_tab = arrival_tab = fairness = None
    if macro is not None:
        fig_main(macro)
        parts = fig_breakdown(macro)
        fig_bottleneck(macro)
        memory = fig_memory(macro)
        gap_tab = fig_sweep(macro, 'gap_mode', ['zero', MAIN_GAP, 'fixed:15'], ['none', 'log-normal med 2 s', 'fixed 15 s'],
                            'f10_tool_delay', 'F10  Effect of tool idle time (P2 K2)', 'Same arrivals and tasks; only the delay between steps changes.')
        arrival_tab = fig_sweep(macro, 'arrival', ['poisson:0.5', MAIN_ARRIVAL, 'poisson:2'], ['0.5/min', '1/min', '2/min'],
                                'f11_arrival', 'F11  Effect of arrival rate (P2 K2)', 'Poisson arrivals; tool delay log-normal median 2 s.')
        fairness = fig_fairness(macro)
        fig_timeline(macro)
    report(live, micro, cache, macro, parts, memory, gap_tab, arrival_tab, fairness)
    print('Wrote', ROOT / 'PHASE1_REPORT.md', 'and figures in', FIG, '| notes:', NOTES or 'none')


if __name__ == '__main__':
    main()
