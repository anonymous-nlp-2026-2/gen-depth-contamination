import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np

with open('/root/autodl-tmp/gen-depth-contamination/results/feature_importance/importance_results.json') as f:
    data = json.load(f)

name_map = {
    'surp_mean': r'Surprisal $\mu$',
    'surp_std': r'Surprisal $\sigma$',
    'surp_skew': 'Surprisal Skew',
    'surp_kurt': 'Surprisal Kurt.',
    'surp_d1_mean': r'$\Delta^1$ Surprisal $\mu$',
    'surp_d1_std': r'$\Delta^1$ Surprisal $\sigma$',
    'surp_d1_skew': r'$\Delta^1$ Surprisal Skew',
    'surp_d2_mean': r'$\Delta^2$ Surprisal $\mu$',
    'surp_d2_std': r'$\Delta^2$ Surprisal $\sigma$',
    'ttr': 'TTR',
    'hapax_ratio': 'Hapax Ratio',
    'self_bleu': 'Self-BLEU',
    'freq_kurtosis': 'Freq. Kurt.',
    'freq_entropy': 'Freq. Entropy',
    'low_freq_ratio': 'Low-Freq Ratio',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

for ax, key, title_label in [
    (ax1, '0v1', 'Gen-0 vs Gen-1'),
    (ax2, '1v2', 'Gen-1 vs Gen-2'),
]:
    section = data[key]
    baseline = section['baseline_auc']
    imp = section['importance']

    sorted_feats = sorted(imp.keys(), key=lambda f: imp[f]['mean'])

    means = [imp[f]['mean'] for f in sorted_feats]
    stds = [imp[f]['std'] for f in sorted_feats]
    labels = [name_map[f] for f in sorted_feats]

    y_pos = np.arange(len(sorted_feats))

    colors = ['#2166ac' if m >= 0 else '#b2182b' for m in means]

    ax.barh(y_pos, means, xerr=stds, height=0.65,
            color=colors, edgecolor='black', linewidth=0.4,
            error_kw={'linewidth': 0.6, 'capsize': 1.5, 'capthick': 0.6},
            zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r'$\Delta$AUC (permutation importance)')
    ax.set_title(f'{title_label} (AUC = {baseline:.3f})', fontsize=9, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5, zorder=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(w_pad=2.5)
plt.savefig('/root/autodl-tmp/gen-depth-contamination/docs/paper/figures/feature_importance.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('/root/autodl-tmp/gen-depth-contamination/docs/paper/figures/feature_importance.png',
            bbox_inches='tight', dpi=300)
print('Done.')
