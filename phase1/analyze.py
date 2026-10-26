"""Derive tables and export publication-quality plots from raw observations."""

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW, OUT, FIG = Path('results/raw'), Path('results/summary'), Path('figures')
COLORS = ['#156b7c', '#e47d54', '#6878a9', '#90b7a5', '#bc9a63']


def backend_label():
    path = OUT.parent / 'server.json'
    command = json.loads(path.read_text())['command'] if path.exists() else []
    if '--n-gpu-layers' in command and command[command.index('--n-gpu-layers')+1] == '0':
        return f'{platform.machine()} CPU'
    return 'M4 · Metal' if sys.platform == 'darwin' else 'GPU offload'


def read(name):
    path = RAW / name
    return pd.DataFrame([json.loads(s) for s in path.read_text().splitlines() if s]) if path.exists() else pd.DataFrame()


def save(fig, name, note):
    fig.text(0.01, 0.015, note, fontsize=8, color='#5a6575')
    fig.savefig(FIG / f'{name}.png', dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def timings(frame):
    for key in ['prompt_ms', 'predicted_ms', 'prompt_n', 'cache_n', 'predicted_n']:
        frame[key] = frame['timings'].map(lambda d: d.get(key, np.nan) if isinstance(d, dict) else np.nan)
    return frame


def micro():
    data = read('micro_requests.jsonl')
    if data.empty:
        return
    data = timings(data)
    data.to_csv(OUT / 'micro_requests.csv', index=False)
    valid = data[data.status == 'ok'].copy()
    per_run = valid.groupby(['length', 'output', 'k', 'repeat']).agg(
        prefill_ms=('prompt_ms', 'mean'), decode_ms=('predicted_ms', 'mean'),
        roundtrip_s=('roundtrip_s', 'mean'), generated=('predicted_n', 'sum'),
        start=('dispatch_s', 'min'), end=('done_s', 'max')).reset_index()
    per_run['generated_tokens_s'] = per_run.generated / (per_run.end - per_run.start)
    per_run.to_csv(OUT / 'micro_runs.csv', index=False)
    per_run.groupby(['length', 'output', 'k']).agg(['mean', 'std']).to_csv(OUT / 'micro_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for color, k in zip(COLORS, [1, 2]):
        subset = per_run[(per_run.output == 64) & (per_run.k == k)]
        for ax, metric, divisor in zip(axes, ['prefill_ms', 'decode_ms', 'generated_tokens_s'], [1000, 1000, 1]):
            grouped = subset.groupby('length')[metric].agg(['mean', 'std']) / divisor
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                        color=color, marker='o', lw=2, capsize=3, label=f'K = {k}')
    for ax, title, label in zip(axes, ['Processing the input', 'Generating 64 tokens', 'Combined output rate'],
                               ['Prefill time (s)', 'Generation interval per request (s)', 'Generated tokens / s']):
        ax.set(title=title, xlabel='Input tokens', ylabel=label)
        ax.set_xticks([512, 2048, 4096], ['512', '2,048', '4,096'])
    axes[0].legend(frameon=False)
    fig.suptitle('01  /  Local inference cost', x=.07, ha='left', weight='bold', fontsize=18)
    fig.tight_layout(rect=[0, .07, 1, .9])
    save(fig, '01_micro_cost', f'{backend_label()} · Qwen3-4B Q4_K_M · cache disabled · mean ± SD across 3 trials; no confidence-interval claim')


def cache():
    data = read('cache.jsonl')
    if data.empty:
        return
    data = timings(data)
    data['reuse_fraction'] = data.cache_n / data.input_tokens
    data.to_csv(OUT / 'cache.csv', index=False)
    modes = ['retain', 'wait', 'overwrite', 'erase', 'save_restore']
    labels = ['Retain', 'Wait 0.5 s\n(no competition)', 'Overwrite', 'Erase', 'Save + restore']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for color, length, offset in zip(COLORS, [2048, 4096], [-.18, .18]):
        sub = data[data.length == length]
        for ax, field, scale in [(axes[0], 'prompt_ms', 1000), (axes[1], 'reuse_fraction', .01)]:
            grouped = sub.groupby('mode')[field].agg(['mean', 'std']).reindex(modes) / scale
            ax.bar(np.arange(5) + offset, grouped['mean'], .34, color=color,
                   yerr=grouped['std'], capsize=3, label=f'{length:,} input tokens')
            ax.set_xticks(np.arange(5), labels)
    axes[0].set(ylabel='Continuation prefill time (s)', title='The cost of losing history')
    axes[0].set_ylim(bottom=0)
    axes[1].set(ylabel='Input tokens reused (%)', ylim=(0, 105), title='Did the continuation reuse its prefix?')
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('02  /  What survives an agent pause?', x=.07, ha='left', weight='bold', fontsize=18)
    fig.tight_layout(rect=[0, .08, 1, .91])
    save(fig, '02_cache_reuse', 'Same-slot mechanism probe · f16 KV · extra RAM cache disabled · error bars: SD over 3 trials')

    checkpoint = data[data['mode'] == 'save_restore'].copy()
    if not checkpoint.empty:
        checkpoint['handling_and_prefill_s'] = (checkpoint.save_roundtrip_s
            + checkpoint.erase_roundtrip_s + checkpoint.restore_roundtrip_s
            + checkpoint.prompt_ms/1000)
        checkpoint['saved_mib'] = checkpoint['save'].map(lambda d: d['n_written']/2**20)
        checkpoint.to_csv(OUT / 'checkpoint_cost.csv', index=False)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for x, length in enumerate([2048, 4096]):
            rebuild = data[(data.length == length) & (data['mode'] == 'erase')].prompt_ms/1000
            restored = checkpoint[checkpoint.length == length].handling_and_prefill_s
            ax.bar(x-.17, rebuild.mean(), .3, color=COLORS[1], yerr=rebuild.std(), capsize=4,
                   label='Rebuild prefix' if x == 0 else None)
            ax.bar(x+.17, restored.mean(), .3, color=COLORS[0], yerr=restored.std(), capsize=4,
                   label='Save + erase + restore + cached prefill' if x == 0 else None)
            ax.text(x-.17, (rebuild.mean()+rebuild.std())*1.2, f'{rebuild.mean():.2f} s', ha='center', fontsize=10)
            ax.text(x+.17, (restored.mean()+restored.std())*1.2, f'{restored.mean():.2f} s', ha='center', fontsize=10)
        ax.set(yscale='log', ylabel='Mechanism cost (s, log scale)', xlabel='Original input tokens',
               xticks=[0, 1], xticklabels=['2,048', '4,096'], ylim=(.1, 60),
               title='08  /  Checkpoint handling versus rebuilding history')
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout(rect=[0, .09, 1, 1])
        save(fig, '08_checkpoint_cost', 'Warm file cache · excludes common decode and unrelated work · descriptive mechanism comparison, not scheduler speedup')


def macro():
    runs, jobs, requests = read('macro_runs.jsonl'), read('macro_jobs.jsonl'), read('macro_requests.jsonl')
    if runs.empty:
        return
    requests = timings(requests)
    requests = requests.sort_values(['label', 'task_id', 'step'])
    previous_input = requests.groupby(['label', 'task_id']).input_tokens.shift()
    previous_output = requests.groupby(['label', 'task_id']).predicted_n.shift()
    requests['full_history_threshold'] = previous_input + previous_output - 1
    requests['continuation_history_hit'] = np.where(requests.step > 0,
        (requests.cache_n >= requests.full_history_threshold).astype(float), np.nan)
    jobs.to_csv(OUT / 'macro_jobs.csv', index=False)
    requests.to_csv(OUT / 'macro_requests.csv', index=False)
    runs.to_csv(OUT / 'macro_runs.csv', index=False)
    run_latency = jobs[jobs.status == 'ok'].groupby('label').latency_s.agg(p50_s='median', p95_s=lambda x: x.quantile(.95),
                                                    max_s='max').reset_index()
    summary = runs.merge(run_latency, on='label')
    summary.to_csv(OUT / 'macro_summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for color, mode, title in zip(COLORS, ['recorded', 'zero'], ['Recorded external gaps', 'Zero-gap sensitivity']):
        sub = summary[summary.gap_mode == mode]
        if sub.empty:
            continue
        for ax, field in zip(axes, ['jobs_per_minute', 'p50_s', 'p95_s']):
            grouped = sub.groupby('k')[field].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                        color=color, marker='o', lw=2, capsize=4, label=title)
            for _, row in sub.iterrows():
                ax.scatter(row.k, row[field], color=color, alpha=.4, s=18)
    for ax, title, label in zip(axes, ['Completed replay jobs', 'Typical completion', 'Tail completion'],
                               ['Jobs / minute', 'P50 task time (s)', 'P95 task time (s)']):
        ax.set(title=title, xlabel='Request concurrency K', ylabel=label, xticks=[1, 2])
        ax.set_xlim(.8, 2.2)
    axes[0].legend(frameon=False, fontsize=8, loc='best')
    fig.suptitle('03  /  Multi-step Retail serving', x=.07, ha='left', weight='bold', fontsize=18)
    fig.tight_layout(rect=[0, .08, 1, .9])
    save(fig, '03_macro_performance', '4 complete source tasks / batch · 3 trials per K · fixed P = 2 and KV budget · dots: trials; bars: SD · replay is not semantic success')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for color, k in zip(COLORS, [1, 2]):
        sub = jobs[(jobs.k == k) & (jobs.gap_mode == 'recorded') & (jobs.status == 'ok')].sort_values('latency_s')
        axes[0].step(sub.latency_s, np.arange(1, len(sub)+1) / len(sub),
                     where='post', color=color, label=f'K = {k}', lw=2)
        q = requests[(requests.k == k) & (requests.gap_mode == 'recorded')]
        groups = [q[q.step == 0].queue_s.to_numpy(), q[q.step > 0].queue_s.to_numpy()]
        for i, values in enumerate(groups):
            axes[1].scatter(np.full(len(values), i + (k-1.5)*.16), values,
                            color=color, alpha=.3, s=15)
            if len(values):
                axes[1].plot([i+(k-1.5)*.16-.05, i+(k-1.5)*.16+.05],
                             [np.median(values)]*2, color=color, lw=3)
    axes[0].set(xlabel='Task completion time (s)', ylabel='Fraction of replay jobs completed',
                title='Completion-time distribution', ylim=(0, 1.02))
    axes[0].legend(frameon=False)
    axes[1].set(xticks=[0, 1], xticklabels=['First step', 'Continuation'], ylabel='External FIFO wait (s)',
                title='Who waits in the FIFO queue?')
    fig.suptitle('04  /  Tails and waiting', x=.08, ha='left', weight='bold', fontsize=18)
    fig.tight_layout(rect=[0, .08, 1, .9])
    save(fig, '04_macro_distribution', 'Recorded-gap workload · pooled repeated jobs are shown descriptively; observations are not independent unique tasks')

    chosen = runs[(runs.k == 2) & (runs.gap_mode == 'recorded')].iloc[0].label
    sub = requests[requests.label == chosen]
    origin = sub.enqueue_s.min()
    ids = list(sub.task_id.unique())
    fig, ax = plt.subplots(figsize=(12, max(4.2, len(ids)*.48)))
    for y, task_id in enumerate(ids):
        for _, row in sub[sub.task_id == task_id].iterrows():
            ax.broken_barh([(row.enqueue_s-origin, row.queue_s)], (y-.25, .5), facecolors='#d8dde4')
            ax.broken_barh([(row.dispatch_s-origin, row.roundtrip_s)], (y-.25, .5), facecolors=COLORS[0])
    ax.set(yticks=range(len(ids)), yticklabels=[f'Task {x}' for x in ids], xlabel='Seconds from first ready step',
           title='05  /  One batch, from first request to final step')
    ax.grid(axis='x', alpha=.25)
    fig.tight_layout(rect=[0, .08, 1, 1])
    save(fig, '05_execution_timeline', f'{chosen} · teal: model round trip · gray: FIFO waiting · blank intervals: external gaps')

    cache_runs = requests.groupby(['label', 'k', 'repeat']).agg(
        cached_tokens=('cache_n', 'sum'), input_tokens=('input_tokens', 'sum'),
        continuation_history_hit=('continuation_history_hit', 'mean')).reset_index()
    cache_runs['input_reuse_fraction'] = cache_runs.cached_tokens/cache_runs.input_tokens
    cache_runs.to_csv(OUT / 'macro_cache.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, metric, title in zip(axes, ['input_reuse_fraction', 'continuation_history_hit'],
                                 ['All input tokens reused', 'Continuations retaining full history']):
        grouped = cache_runs.groupby('k')[metric].agg(['mean', 'std']) * 100
        ax.bar([0, 1], grouped['mean'].reindex([1, 2]), color=COLORS[:2], width=.55,
               yerr=grouped['std'].reindex([1, 2]), capsize=4)
        for position, k in enumerate([1, 2]):
            mean, std = grouped.loc[k, ['mean', 'std']]
            ax.text(position, mean + std + 2, f'{mean:.1f}%', ha='center', fontsize=10)
        ax.set(xticks=[0, 1], xticklabels=['K = 1', 'K = 2'], ylabel='Percent (%)',
               title=title, ylim=(0, 105))
    fig.suptitle('09  /  Shared instructions can hide lost task history', x=.08,
                 ha='left', weight='bold', fontsize=17)
    fig.tight_layout(rect=[0, .09, 1, .9])
    save(fig, '09_macro_cache', 'Full-history hit: reused tokens cover the previous input + generated tokens − 1 · bars: mean ± SD over 3 trials')


def resources():
    data = read('resources.jsonl')
    if data.empty:
        return
    summary = data.groupby('label').agg(
        peak_server_rss=('server_rss', 'max'), min_system_available=('system_available', 'min'),
        swap_used_start=('swap_used', 'first'), swap_used_end=('swap_used', 'last'),
        swap_out_start=('swap_out', 'first'), swap_out_end=('swap_out', 'last'))
    summary.to_csv(OUT / 'resources.csv')
    macro = data[data.label.str.startswith('macro')].copy()
    if macro.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    origin = macro.time_s.min()
    x = (macro.time_s-origin)/60
    axes[0].plot(x, macro.server_rss / 2**30, color=COLORS[0], lw=1.5, label='Server RSS')
    axes[0].plot(x, macro.system_available / 2**30, color=COLORS[2], lw=1.5, label='System available')
    axes[0].set(ylabel='GiB', title='06  /  Memory during the measured replay runs')
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(x, (macro.swap_out-macro.swap_out.iloc[0]) / 2**30, color=COLORS[1])
    axes[1].set(xlabel='Elapsed minutes (including gaps between runs)', ylabel='Cumulative swap-out Δ (GiB)')
    fig.tight_layout(rect=[0, .07, 1, 1])
    save(fig, '06_memory', 'Process + system memory · sampled every 250 ms · RSS is not total physical footprint; allocations may be preallocated')


def live():
    rows = []
    for path in sorted((RAW / 'live_test').glob('task_*.json')):
        record = json.loads(path.read_text())
        calls = record['calls']
        tool_events = record.get('tool_events', [])
        rows.append(dict(task_id=record['task_id'], status=record['status'],
            env_reward=record.get('env_reward', 0), agent_calls=len(calls),
            generated_tokens=sum(c['response'].get('timings', {}).get('predicted_n', 0) for c in calls),
            max_prompt=max((len(c['prompt_tokens']) for c in calls), default=0),
            normalization_count=sum(c.get('normalized_tool_json', False) for c in calls),
            tool_calls=len(tool_events), tool_errors=sum(t['error'] for t in tool_events),
            termination=record.get('simulation', {}).get('termination_reason'),
            elapsed_s=record['elapsed_s']))
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'live_quality.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(frame))
    axes[0].bar(x, frame.agent_calls, color=[COLORS[0] if r == 1 else '#a5aebb' for r in frame.env_reward])
    for i, row in frame.iterrows():
        axes[0].text(i, row.agent_calls+.3, 'PASS' if row.env_reward == 1 else 'FAIL',
                     ha='center', fontsize=8, color=COLORS[0] if row.env_reward == 1 else '#586273')
    axes[0].set(xticks=x, xticklabels=frame.task_id, xlabel='Frozen test task ID',
                ylabel='Agent model calls', title='Real tasks, including failed executions')
    axes[1].bar(x, frame.max_prompt/1000, color=COLORS[2])
    axes[1].axhline(12.288, color=COLORS[1], ls='--', lw=1.5, label='Per-slot context capacity')
    axes[1].set(xticks=x, xticklabels=frame.task_id, xlabel='Frozen test task ID',
                ylabel='Maximum observed input (thousand tokens)', title='How much history does each task retain?')
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle('07  /  Live Retail validation', x=.07, ha='left', weight='bold', fontsize=18)
    fig.tight_layout(rect=[0, .08, 1, .9])
    save(fig, '07_live_validation', 'Local Qwen agent + DeepSeek user · programmatic ENV checks only; natural-language assertions not evaluated')


def main():
    global RAW, OUT, FIG
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    root = Path(parser.parse_args().results_dir)
    # The original pilot keeps its figures at the project root.
    RAW, OUT = root / 'raw', root / 'summary'
    FIG = Path('figures') if root == Path('results') else root / 'figures'
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#344154', 'text.color': '#15263b',
        'axes.titleweight': 'bold', 'axes.titlepad': 14,
        'axes.grid': True, 'grid.alpha': .15, 'grid.linestyle': '-',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'pdf.fonttype': 42, 'savefig.pad_inches': .2})
    micro()
    cache()
    macro()
    resources()
    live()
    print(f'Wrote tables to {OUT} and figures to {FIG}', flush=True)


if __name__ == '__main__':
    main()
