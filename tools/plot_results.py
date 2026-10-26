"""Create publication-style SVGs directly from extracted main-table values."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'static/data/results.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#cdd7d0', 'axes.labelcolor': '#526158', 'xtick.color': '#657169', 'ytick.color': '#657169', 'text.color': '#3f5046', 'svg.fonttype': 'none'})
styles = {
    'Random': ('#c0c8c3', 'v', (0, (2, 3))),
    'Uniform': ('#879dbe', 's', '-'),
    'K-Center': ('#a4aa99', 'D', (0, (5, 2))),
    'Herding': ('#b9a699', 'x', (0, (1, 2))),
    'SD3.5': ('#a78261', '^', '-'),
    'UniDD*': ('#b1a6bd', 'P', (0, (4, 2))),
    'S²-RL': ('#226c5c', 'o', '-'),
}
for key, dataset in DATA['tables'].items():
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    fig.subplots_adjust(left=.13, right=.97, top=.96, bottom=.31)
    x = np.array(dataset['ratios'])
    for row in dataset['rows']:
        color, marker, dash = styles[row['name']]
        ours = row['name'] == 'S²-RL'
        ax.errorbar(x, row['mean'], yerr=row['std'], color=color, marker=marker, linestyle=dash, linewidth=2.6 if ours else 1.3, markersize=6 if ours else 4.5, elinewidth=1.2 if ours else .8, capsize=2.4, label=row['name'], zorder=8 if ours else 3)
        if ours:
            for xi, yi in zip(x, row['mean']):
                ax.annotate(f'{yi:.2f}', (xi, yi), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=10, color=color, weight='bold')
    min_y = min(v-e for row in dataset['rows'] for v,e in zip(row['mean'],row['std']))
    max_y = max(v+e for row in dataset['rows'] for v,e in zip(row['mean'],row['std']))
    ax.set_ylim(min_y-1.6, max_y+3.2)
    ax.set_xlim(x.min()-(x.max()-x.min())*.08, x.max()+(x.max()-x.min())*.08)
    ax.set_xticks(x, [f'{v:g}%' for v in x])
    ax.set_xlabel('Synthetic dataset budget', labelpad=9)
    ax.set_ylabel('mIoU (%) ↑', labelpad=9)
    ax.yaxis.grid(True, color='#eaf0eb', linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=3, width=.7, labelsize=10)
    handles, labels = ax.get_legend_handles_labels()
    order = [6, 4, 1, 2, 0, 3, 5]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower center', bbox_to_anchor=(.52,.002), ncol=4, frameon=False, fontsize=9, columnspacing=1.2, handlelength=1.8, handletextpad=.55, labelspacing=.9)
    for ext in ['svg', 'png']:
        fig.savefig(ROOT / f'static/images/figures/results-{key}.{ext}', dpi=170, facecolor='white', metadata={'Creator': 'S2RL project page; values from the current manuscript main tables'} if ext == 'svg' else None)
    plt.close(fig)
    print(f'{key}: plotted {len(dataset["rows"])} methods at {len(x)} budgets with reported uncertainty')
