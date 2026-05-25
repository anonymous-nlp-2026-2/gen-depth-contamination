import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rates = [10, 30, 50, 100]

hellaswag = {
    'no_filter':  [0.6652, 0.6625, 0.6582, 0.6217],
    'binary':     [0.6709, 0.6702, 0.6700, None],
    'graduated':  [0.6672, 0.6640, 0.6606, 0.6215],
}

arc_c = {
    'no_filter':  [0.4471, 0.4334, 0.4300, 0.3891],
    'binary':     [0.4462, 0.4480, 0.4505, None],
    'graduated':  [0.4428, 0.4334, 0.4292, 0.3933],
}

colors = {'binary': '#2196F3', 'graduated': '#FF9800', 'no_filter': '#9E9E9E'}
markers = {'binary': 'o', 'graduated': '^', 'no_filter': 's'}
linestyles = {'binary': '-', 'graduated': '--', 'no_filter': ':'}
labels = {'binary': 'Binary Filter', 'graduated': 'Graduated Filter', 'no_filter': 'No Filter'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

def plot_panel(ax, data, ylabel, title, annotate_info=None):
    for strat in ['no_filter', 'graduated', 'binary']:
        vals = data[strat]
        valid_rates = [r for r, v in zip(rates, vals) if v is not None]
        valid_vals = [v for v in vals if v is not None]
        ax.plot(valid_rates, valid_vals, color=colors[strat], marker=markers[strat],
                linestyle=linestyles[strat], label=labels[strat], linewidth=1.8,
                markersize=7, zorder=3)
        if strat == 'binary' and vals[-1] is None:
            last_valid = valid_vals[-1]
            ax.plot(100, last_valid, marker='x', color=colors[strat], markersize=9,
                    markeredgewidth=2, zorder=4)
            ax.plot([valid_rates[-1], 100], [last_valid, last_valid],
                    color=colors[strat], linestyle=':', linewidth=1.2, alpha=0.5, zorder=2)
            ax.annotate('N/A\n(100% filtered)', xy=(100, last_valid),
                        xytext=(100, last_valid - 0.012), fontsize=8, color=colors[strat],
                        ha='center', va='top')

    if annotate_info:
        x_pt = annotate_info['x']
        binary_val = data['binary'][rates.index(x_pt)]
        nofilter_val = data['no_filter'][rates.index(x_pt)]
        delta = (binary_val - nofilter_val) * 100
        tx, ty = annotate_info['text_offset']
        ax.annotate(f'$\\Delta$={delta:+.2f}pp',
                    xy=(x_pt, binary_val),
                    xytext=(x_pt + tx, binary_val + ty),
                    fontsize=9, color=colors['binary'], fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=colors['binary'], lw=1.5),
                    ha='center', va='top')

    ax.set_xlabel('Contamination Rate (%)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(rates)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plot_panel(ax1, hellaswag, 'HellaSwag', '(a) HellaSwag',
           annotate_info={'x': 50, 'text_offset': (20, -0.008)})
plot_panel(ax2, arc_c, 'ARC-Challenge', '(b) ARC-Challenge',
           annotate_info={'x': 50, 'text_offset': (20, -0.008)})

plt.tight_layout()

out_dir = '/root/autodl-tmp/gen-depth-contamination/results/contamination_sweep'
fig.savefig(f'{out_dir}/fig_contamination_sweep.pdf', bbox_inches='tight', dpi=300)
fig.savefig(f'{out_dir}/fig_contamination_sweep.png', bbox_inches='tight', dpi=300)
print('Saved PDF and PNG.')
