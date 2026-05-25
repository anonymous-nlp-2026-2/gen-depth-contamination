import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data: Retrain-chain (Qwen2.5-1.5B, depth 0->7)
retrain_pairs = ['0 vs 1', '1 vs 2', '2 vs 3', '3 vs 4', '4 vs 5', '5 vs 6', '6 vs 7']
retrain_auc   = [0.9837, 0.8971, 0.8098, 0.7600, 0.7459, 0.7017, 0.6410]
retrain_ci_lo = [0.9821, 0.8949, 0.8052, 0.7528, 0.7379, 0.6958, 0.6376]
retrain_ci_hi = [0.9854, 0.8993, 0.8144, 0.7672, 0.7538, 0.7076, 0.6445]

# Data: Prompt-chain (Qwen2.5-1.5B on C4)
prompt_pairs = ['0 vs 1', '1 vs 2', '2 vs 3', '3 vs 4', '4 vs 5']
prompt_auc   = [0.975, 0.629, 0.549, 0.524, 0.515]
prompt_ci_lo = [0.972, 0.619, 0.539, 0.514, 0.505]
prompt_ci_hi = [0.978, 0.639, 0.559, 0.535, 0.527]

retrain_x = np.arange(len(retrain_pairs))
prompt_x  = np.arange(len(prompt_pairs))

retrain_auc = np.array(retrain_auc)
retrain_err_lo = retrain_auc - np.array(retrain_ci_lo)
retrain_err_hi = np.array(retrain_ci_hi) - retrain_auc

prompt_auc = np.array(prompt_auc)
prompt_err_lo = prompt_auc - np.array(prompt_ci_lo)
prompt_err_hi = np.array(prompt_ci_hi) - prompt_auc

# Plot
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Retrain-chain
ax.errorbar(retrain_x, retrain_auc,
            yerr=[retrain_err_lo, retrain_err_hi],
            fmt='o-', color='#2166ac', markersize=4.5, linewidth=1.4,
            ecolor='#2166ac', elinewidth=0.8, capsize=2.5, capthick=0.8,
            label='Retrain-chain', zorder=3)

# Prompt-chain
ax.errorbar(prompt_x, prompt_auc,
            yerr=[prompt_err_lo, prompt_err_hi],
            fmt='s--', color='#e66101', markersize=4.5, linewidth=1.4,
            ecolor='#e66101', elinewidth=0.8, capsize=2.5, capthick=0.8,
            label='Prompt-chain', zorder=3)

# Theta threshold
ax.axhline(y=0.60, color='#b2182b', linestyle=':', linewidth=1.0, zorder=1)
ax.text(6.05, 0.608, r'$\theta$ = 0.60', fontsize=7.5, color='#b2182b',
        va='bottom', ha='left')

# Annotations
ax.annotate(r'Retrain $K^*\!\geq\!7$',
            xy=(6, 0.6410), xytext=(4.6, 0.56),
            fontsize=7, color='#2166ac',
            arrowprops=dict(arrowstyle='->', color='#2166ac', lw=0.8),
            zorder=4)

ax.annotate(r'Prompt $K^*\!=\!2$',
            xy=(1, 0.629), xytext=(2.2, 0.72),
            fontsize=7, color='#e66101',
            arrowprops=dict(arrowstyle='->', color='#e66101', lw=0.8),
            zorder=4)

# Axes
ax.set_xticks(retrain_x)
ax.set_xticklabels(retrain_pairs, fontsize=8)
ax.set_xlabel('Depth pair', fontsize=9)
ax.set_ylabel('Pairwise AUC', fontsize=9)
ax.set_ylim(0.45, 1.02)
ax.tick_params(axis='y', labelsize=8)

# Remove grid, keep spines clean
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
ax.legend(fontsize=7.5, loc='upper right', frameon=False)

plt.tight_layout(pad=0.3)

# Save
fig.savefig('/root/autodl-tmp/gen-depth-contamination/figures/retrain_decay_curve_v2.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig('/root/autodl-tmp/gen-depth-contamination/figures/retrain_decay_curve_v2.png',
            bbox_inches='tight', dpi=300)
plt.close()
print('Done.')
