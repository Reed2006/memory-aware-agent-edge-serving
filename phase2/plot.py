"""Plot model-free Phase 2 CSV summaries, including empty or all-failed sweeps."""

import argparse
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(root: Path) -> list[Path]:
    root = Path(root)
    t0 = 'policy_summary_t0.csv'
    if not (root / 'summary' / t0).exists():
        return _plot_view(root, 'policy_summary.csv', '', 'All-trial descriptive (legacy sample)')
    paths = _plot_view(root, t0, '', 'T0 ablations: trial 0 only')
    if (root / 'summary/policy_summary_confirmatory.csv').exists():
        paths += _plot_view(root, 'policy_summary_confirmatory.csv', '_confirmatory',
                            'Confirmatory repeats: complete predeclared trial sets')
    return paths


def _plot_view(root: Path, table_name: str, suffix: str, scope_title: str) -> list[Path]:
    root = Path(root)
    path = root / 'summary' / table_name
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    output = root / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    panels = _panels(table)
    specifications = [
        ('throughput', [('jobs_per_min_mean', 'Completed tasks/min')]),
        ('completion_time', [('jct_mean_s', 'Mean JCT (s)'), ('p95_s', 'Mean per-run P95 JCT (s)')]),
        ('fairness', [('slowdown_short_mean', 'Short-task slowdown'),
                      ('slowdown_long_mean', 'Long-task slowdown')]),
        ('action_overhead', [('decision_cpu_s', 'Decision CPU (s/run)'), ('io_s', 'Save/restore (s/run)')]),
        ('history', [('full_history_hit_rate', 'Full-history hit rate'),
                     ('wasted_prefill_fraction', 'Re-prefilled fraction of prompt work')]),
        ('memory', [('peak_server_rss_gib', 'Peak server RSS (GiB)'),
                    ('min_system_available_gib', 'Min system available (GiB)')]),
    ]
    with plt.rc_context({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}):
        for name, metrics in specifications:
            fig, axes = plt.subplots(len(panels), 1, figsize=(max(7, min(14, len(table) * .5)),
                                                            3.7 * len(panels)), squeeze=False)
            fig.suptitle(scope_title + '\nEach panel has identical seed/trial IDs; labels show seed/run/trial N',
                         fontsize=10)
            for ax, (key, group) in zip(axes[:, 0], panels):
                if group.empty:
                    ax.text(.5, .5, 'No completed runs / no measurements', ha='center', va='center',
                            transform=ax.transAxes)
                    ax.set_xticks([])
                    continue
                labels = [_label(row) for row in group.itertuples()]
                x = np.arange(len(group))
                width = .8 / len(metrics)
                has_values = False
                for index, (metric, label) in enumerate(metrics):
                    values = pd.to_numeric(group[metric], errors='coerce')
                    has_values |= bool(values.notna().any())
                    ax.bar(x + (index - (len(metrics) - 1) / 2) * width, values, width=width, label=label)
                ax.set_xticks(x, labels, rotation=35, ha='right')
                title = ' / '.join(str(value) if len(str(value)) <= 18 else str(value)[:12] + '...'
                                   for value in key)
                ax.set_title(textwrap.fill(title, width=100))
                ax.set_ylim(bottom=0)
                if not has_values:
                    ax.text(.5, .5, 'No matched reference / no completed measurements',
                            transform=ax.transAxes, ha='center', va='center')
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(axis='y', alpha=.2)
                ax.set_axisbelow(True)
            fig.tight_layout(rect=(0, 0, 1, .94))
            for extension in ('png', 'pdf'):
                target = output / f'{name}{suffix}.{extension}'
                fig.savefig(target, dpi=180, bbox_inches='tight')
                figures.append(target)
            plt.close(fig)
        figures += _frontier(panels, output, len(table), suffix, scope_title)
    return figures


def _panels(table: pd.DataFrame):
    # Never combine machines, budgets, workloads or gap/arrival conditions into one bar.
    keys = ['platform_id', 'workload_hash', 'model_hash', 'kv_precision',
                            'P', 'ctx_size', 'kv_unified', 'cost_hash', 'gap_mode', 'arrival',
                            'cache_ram_mib', 'cache_idle_slots', 'checkpoint_budget_bytes',
                            'synthetic_performance']
    # Different seed/trial samples get separate panels; no intersection or
    # unequal trial averages are silently used to construct comparable bars.
    keys += [key for key in ['sample_scope', 'seed_ids', 'trials_by_seed', 'runs', 'seeds']
             if key in table]
    groups = table.groupby(keys, dropna=False, sort=True) if not table.empty else [((), table)]
    return list(groups)


def _label(row) -> str:
    name = getattr(row, 'policy_label', None)
    name = row.policy if not isinstance(name, str) or not name else name
    counts = (f'S={getattr(row, "seeds", "?")} R={getattr(row, "runs", "?")} '
              f'T={getattr(row, "trial_ids", "?")}')
    return f'{name}\n{str(row.param_hash)[:6]}\n{counts}'


def _frontier(panels, output: Path, rows: int, suffix='', scope_title='') -> list[Path]:
    """Throughput against the fairness cost it is bought with, one point per policy."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 4.6 * len(panels)), squeeze=False)
    fig.suptitle(scope_title + '\nIdentical seed/trial sample within each panel', fontsize=10)
    for ax, (key, group) in zip(axes[:, 0], panels):
        x = pd.to_numeric(group.get('slowdown_short_mean', pd.Series(dtype=float)), errors='coerce')
        y = pd.to_numeric(group.get('jobs_per_min_mean', pd.Series(dtype=float)), errors='coerce')
        valid = x.notna() & y.notna()
        if group.empty or not valid.any():
            ax.text(.5, .5, 'Needs completed runs and an isolated K=1/J=1 reference',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.scatter(x[valid], y[valid], s=36, color='#2a6fdb', edgecolor='white', linewidth=1.5, zorder=3)
        for row, xv, yv in zip(group.loc[valid].itertuples(), x[valid], y[valid]):
            ax.annotate(_label(row).split('\n')[0], (xv, yv), xytext=(5, 4),
                        textcoords='offset points', fontsize=8, color='#333333')
        ax.set_xlabel('Short-task mean slowdown (lower is fairer)')
        ax.set_ylabel('Completed tasks/min (higher is better)')
        ax.set_title(textwrap.fill(' / '.join(str(v)[:12] for v in key), width=100), fontsize=8)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, .94))
    paths = []
    for extension in ('png', 'pdf'):
        target = output / f'frontier{suffix}.{extension}'
        fig.savefig(target, dpi=180, bbox_inches='tight')
        paths.append(target)
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    for path in plot(args.results_dir):
        print(path)


if __name__ == '__main__':
    main()
