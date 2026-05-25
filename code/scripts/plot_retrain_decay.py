import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

pairs_retrain = ['0v1', '1v2', '2v3', '3v4', '4v5', '5v6', '6v7']
auc_retrain   = [0.984, 0.897, 0.810, 0.760, 0.746, 0.702, 0.641]

pairs_prompt  = ['0v1', '1v2', '2v3', '3v4', '4v5']
auc_prompt    = [0.975, 0.629, 0.549, 0.524, 0.515]

x_retrain = np.arange(len(pairs_retrain))
x_prompt  = np.arange(len(pairs_prompt))

fig, ax = plt.subplots(figsize=(3.5, 2.4))

ax.plot(x_retrain, auc_retrain, '-o', color='#1f77b4', markersize=4.5,
        linewidth=1.4, label='Retrain-chain', zorder=3)
ax.plot(x_prompt, auc_prompt, '--s', color='#d95319', markersize=4.5,
        linewidth=1.4, label='Prompt-chain', zorder=3, dashes=(4, 2.5))

ax.axhline(y=0.60, color='#a0a0a0', linestyle=':', linewidth=0.9,
           label=r'$\theta = 0.60$', zorder=2)

# Retrain K* annotation: place text above the curve, pointing down to 6v7
ax.annotate(r'Retrain $K^*\!\geq 7$',
            xy=(6, 0.641), xytext=(4.0, 0.55),
            fontsize=7.5, color='#1f77b4',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=0.8))

# Prompt K* annotation: point to the crossing region between 1v2 and 2v3
ax.annotate(r'Prompt $K^*\!=\!2$',
            xy=(1.5, 0.589), xytext=(2.6, 0.74),
            fontsize=7.5, color='#d95319',
            arrowprops=dict(arrowstyle='->', color='#d95319', lw=0.8))

ax.set_xticks(x_retrain)
ax.set_xticklabels(pairs_retrain)
ax.set_xlabel('Depth Pair')
ax.set_ylabel('Pairwise AUC')
ax.set_ylim(0.45, 1.05)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper right', frameon=False, handlelength=1.8)

plt.tight_layout(pad=0.3)

out_dir = '/root/autodl-tmp/gen-depth-contamination/figures'
fig.savefig(f'{out_dir}/retrain_decay_curve.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{out_dir}/retrain_decay_curve.png', dpi=300, bbox_inches='tight')
print('Done')
