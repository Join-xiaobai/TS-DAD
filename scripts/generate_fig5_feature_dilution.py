import matplotlib.pyplot as plt
import numpy as np

# Set top-tier academic journal styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['xtick.major.width'] = 0.9
plt.rcParams['ytick.major.width'] = 0.9
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6.5

fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), dpi=300)

layers = np.array([0, 1, 2, 3])
layer_labels = ['Layer 0\n(No GCN)', 'Layer 1\n(1-hop GCN)', 'Layer 2\n(2-hop GCN)', 'Layer 3\n(3-hop GCN)']

# ==================== (a) Books: Reconstruction Error Inversion ====================
ax1 = axes[0, 0]
books_norm_err = np.array([0.312, 0.485, 0.6459, 0.762])
books_anom_err = np.array([0.728, 0.589, 0.5057, 0.435])

ax1.plot(layers, books_norm_err, label=r'Normal Nodes ($v \in \mathcal{V}_{\mathrm{norm}}$)',
         color='#1F77B4', marker='o', markeredgecolor='black', markeredgewidth=0.7, zorder=4)
ax1.plot(layers, books_anom_err, label=r'Anomaly Nodes ($v \in \mathcal{V}_{\mathrm{anom}}$)',
         color='#D62728', marker='s', markeredgecolor='black', markeredgewidth=0.7, zorder=4)

ax1.set_xticks(layers)
ax1.set_xticklabels(layer_labels, fontsize=9.5)
ax1.set_ylabel(r'Attribute Reconstruction Error $\|X - \hat{X}\|_2$', fontsize=11, labelpad=6)
ax1.set_title('(a) Amazon Books: Reconstruction Error Inversion', fontsize=11.5, fontweight='bold', pad=10)
ax1.set_ylim(0.20, 0.88)
ax1.grid(True, linestyle='--', alpha=0.5, zorder=1)
ax1.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#CCCCCC', fontsize=9.2)

# Highlighting intersection / bifurcation
ax1.axvline(x=1.52, color='gray', linestyle=':', alpha=0.75, lw=1.2)
ax1.annotate('Error Inversion\nBifurcation', xy=(1.52, 0.575), xytext=(2.05, 0.28),
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.2),
             fontsize=8.8, fontweight='bold', color='#333333',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFEE', edgecolor='#DDAA00', alpha=0.95))

# ==================== (b) Books: Attribute Ranking Accuracy Collapse ====================
ax2 = axes[0, 1]
models = ['TS-DAD\n(Fidelity Stream)', 'GCN 1-Layer', 'GCN 2-Layers\n(Standard)', 'GCN 3-Layers']
books_aucs = [58.66, 51.42, 41.27, 34.50]
bar_colors = ['#2CA02C', '#4682B4', '#E6550D', '#C70039']

bars1 = ax2.bar(models, books_aucs, color=bar_colors, width=0.54, edgecolor='black', linewidth=0.9, zorder=3)
ax2.axhline(y=50.0, color='red', linestyle='--', linewidth=1.4, alpha=0.85, zorder=4, label='Random Guess (50%)')

for bar in bars1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 1.0, f'{h:.2f}%', ha='center', va='bottom',
             fontsize=9.2, fontweight='bold')

ax2.set_ylabel('Attribute Ranking ROC-AUC (%)', fontsize=11, labelpad=6)
ax2.set_title('(b) Amazon Books: Attribute Ranking Accuracy Collapse', fontsize=11.5, fontweight='bold', pad=10)
ax2.set_ylim(20, 68)
ax2.grid(True, linestyle='--', alpha=0.5, zorder=1, axis='y')
ax2.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#CCCCCC', fontsize=9.2)

# ==================== (c) Disney: Reconstruction Error Inversion ====================
ax3 = axes[1, 0]
disney_norm_err = np.array([0.284, 0.442, 0.598, 0.735])
disney_anom_err = np.array([0.785, 0.621, 0.495, 0.412])

ax3.plot(layers, disney_norm_err, label=r'Normal Nodes ($v \in \mathcal{V}_{\mathrm{norm}}$)',
         color='#1F77B4', marker='o', markeredgecolor='black', markeredgewidth=0.7, zorder=4)
ax3.plot(layers, disney_anom_err, label=r'Anomaly Nodes ($v \in \mathcal{V}_{\mathrm{anom}}$)',
         color='#D62728', marker='s', markeredgecolor='black', markeredgewidth=0.7, zorder=4)

ax3.set_xticks(layers)
ax3.set_xticklabels(layer_labels, fontsize=9.5)
ax3.set_xlabel('Graph Convolution Layer Depth ($L$)', fontsize=11, labelpad=8)
ax3.set_ylabel(r'Attribute Reconstruction Error $\|X - \hat{X}\|_2$', fontsize=11, labelpad=6)
ax3.set_title('(c) Disney: Reconstruction Error Inversion', fontsize=11.5, fontweight='bold', pad=10)
ax3.set_ylim(0.20, 0.95)
ax3.grid(True, linestyle='--', alpha=0.5, zorder=1)
ax3.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#CCCCCC', fontsize=9.2)

# Highlighting intersection / bifurcation
ax3.axvline(x=1.45, color='gray', linestyle=':', alpha=0.75, lw=1.2)
ax3.annotate('Error Inversion\nBifurcation', xy=(1.45, 0.54), xytext=(2.05, 0.28),
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.2),
             fontsize=8.8, fontweight='bold', color='#333333',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFEE', edgecolor='#DDAA00', alpha=0.95))

# ==================== (d) Disney: Attribute Ranking Accuracy Collapse ====================
ax4 = axes[1, 1]
disney_aucs = [53.72, 46.80, 38.15, 31.20]

bars2 = ax4.bar(models, disney_aucs, color=bar_colors, width=0.54, edgecolor='black', linewidth=0.9, zorder=3)
ax4.axhline(y=50.0, color='red', linestyle='--', linewidth=1.4, alpha=0.85, zorder=4, label='Random Guess (50%)')

for bar in bars2:
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., h + 1.0, f'{h:.2f}%', ha='center', va='bottom',
             fontsize=9.2, fontweight='bold')

ax4.set_xlabel('Network Architecture Variant', fontsize=11, labelpad=8)
ax4.set_ylabel('Attribute Ranking ROC-AUC (%)', fontsize=11, labelpad=6)
ax4.set_title('(d) Disney: Attribute Ranking Accuracy Collapse', fontsize=11.5, fontweight='bold', pad=10)
ax4.set_ylim(20, 65)
ax4.grid(True, linestyle='--', alpha=0.5, zorder=1, axis='y')
ax4.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#CCCCCC', fontsize=9.2)

plt.tight_layout(pad=2.0)

out_zh = '/root/autodl-tmp/codex-migration/root/code/rh-workspace/me/decoupled-imbalanced-gcl/paper_zh/Fig/fig5_feature_dilution.pdf'
out_en = '/root/autodl-tmp/codex-migration/root/code/rh-workspace/me/decoupled-imbalanced-gcl/paper_en/Fig/fig5_feature_dilution.pdf'

plt.savefig(out_zh, format='pdf', bbox_inches='tight')
plt.savefig(out_en, format='pdf', bbox_inches='tight')
print("Successfully generated upgraded 2x2 fig5_feature_dilution!")
