"""Plot measured validation trajectories from report_experiment.py output.

Optional analysis dependency: matplotlib. Does not alter the training image.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(analysis, output):
    colors = {'lora': '#2367a6', 'simplex': '#cc6519', 'dora': '#16836b'}
    styles = {17: '-', 42: '--', 123: ':'}
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'simplexbench'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for r in analysis['runs']:
        rows = r['evaluations']
        for ax, xkey in zip(axes, ('step', 'elapsed_seconds')):
            ax.plot([x[xkey] for x in rows], [x['macro_f1'] for x in rows],
                    color=colors[r['method']], linestyle=styles[r['seed']],
                    label=f"{r['method']} / {r['seed']}", linewidth=1.7)
    target = analysis['manifest']['config']['target_f1']
    for ax in axes:
        ax.axhline(target, color='#555555', linewidth=1, linestyle='-.', label=f'target {target}')
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_xlabel('Optimizer step (rank search included)')
    axes[1].set_xlabel('Training + validation elapsed seconds')
    axes[0].set_ylabel('Validation macro-F1')
    axes[1].legend(fontsize=8, loc='lower right', ncol=2)
    steps = analysis['manifest']['config']['steps']
    fig.suptitle(f'Qwen2.5-0.5B / AG News: {steps} steps, three paired seeds')
    fig.tight_layout()
    fig.savefig(output, metadata={'Date': None}, bbox_inches='tight')
    plt.close(fig)


def plot_adaptation(analysis, output):
    """Separate measured DoRA updates from Simplex search and rank rounding."""
    styles = {17: '-', 42: '--', 123: ':'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for run in analysis['runs']:
        style = styles[run['seed']]
        label = f"seed {run['seed']}"
        if run['method'] == 'dora':
            rows = run['magnitude_trajectory']
            axes[0].plot([0] + [r['step'] for r in rows],
                         [0] + [r['delta_abs_max'] for r in rows],
                         color='#16836b', linestyle=style, label=label)
        elif run['method'] == 'simplex':
            rows = run['rank_history']
            axes[1].plot([0] + [r['step'] for r in rows],
                         [0] + [max(abs(w - 1) for w in r['weights']) for r in rows],
                         color='#cc6519', linestyle=style, marker='o', label=label)
            initial = analysis['manifest']['config']['rank'] * len(rows[0]['ranks'])
            axes[2].step([0] + [r['step'] for r in rows] + [analysis['manifest']['config']['steps']],
                         [initial] + [sum(r['ranks'].values()) for r in rows] + [sum(rows[-1]['ranks'].values())],
                         where='post', color='#cc6519', linestyle=style, label=label)
    labels = [('DoRA magnitude update', 'max |m(step) − m(initial)|'),
              ('Simplex importance at search events', 'max |importance − 1|'),
              ('Simplex allocated rank', 'Sum of layer ranks')]
    for ax, (title, ylabel) in zip(axes, labels):
        ax.set(title=title, xlabel='Optimizer step', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle('Measured adaptation diagnostics — separate scales, three paired seeds')
    fig.tight_layout()
    fig.savefig(output, metadata={'Date': None}, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('analysis')
    p.add_argument('--output', required=True)
    p.add_argument('--adaptation-output', help='Optional magnitude/importance/rank figure')
    a = p.parse_args()
    analysis = json.loads(Path(a.analysis).read_text())
    plot(analysis, a.output)
    if a.adaptation_output:
        plot_adaptation(analysis, a.adaptation_output)
