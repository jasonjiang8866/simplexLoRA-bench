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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('analysis')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plot(json.loads(Path(a.analysis).read_text()), a.output)
