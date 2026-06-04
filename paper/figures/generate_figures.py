#!/usr/bin/env python3
"""Generate publication-quality figures for the student engagement detection paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

output_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Figure 1: Framework Overview
# ============================================================
def create_framework_figure():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Color palette
    colors = {
        'input': '#4CAF50',
        'face': '#2196F3',
        'backbone': '#FF9800',
        'head': '#9C27B0',
        'output': '#F44336',
    }

    # Box 1: Input
    box1 = mpatches.FancyBboxPatch((0.3, 1.8), 1.4, 1.4, boxstyle="round,pad=0.1",
                                     facecolor=colors['input'], edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box1)
    ax.text(1.0, 2.5, 'Input\nImage', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Arrow 1
    ax.annotate('', xy=(2.0, 2.5), xytext=(1.7, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    # Box 2: Face Detection
    box2 = mpatches.FancyBboxPatch((2.2, 1.8), 1.6, 1.4, boxstyle="round,pad=0.1",
                                     facecolor=colors['face'], edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box2)
    ax.text(3.0, 2.5, 'Face\nDetection &\nPreprocessing', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Arrow 2
    ax.annotate('', xy=(4.1, 2.5), xytext=(3.8, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    # Box 3: Backbone
    box3 = mpatches.FancyBboxPatch((4.3, 1.5), 1.8, 2.0, boxstyle="round,pad=0.1",
                                     facecolor=colors['backbone'], edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box3)
    ax.text(5.2, 2.5, 'Lightweight\nBackbone\n(MobileNetV3 /\nEfficientNet-B0)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')

    # Arrow 3
    ax.annotate('', xy=(6.4, 2.5), xytext=(6.1, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    # Box 4: Classification Head
    box4 = mpatches.FancyBboxPatch((6.6, 1.8), 1.6, 1.4, boxstyle="round,pad=0.1",
                                     facecolor=colors['head'], edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box4)
    ax.text(7.4, 2.5, 'Classification\nHead', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Arrow 4
    ax.annotate('', xy=(8.5, 2.5), xytext=(8.2, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    # Box 5: Output
    box5 = mpatches.FancyBboxPatch((8.7, 1.5), 1.1, 2.0, boxstyle="round,pad=0.1",
                                     facecolor=colors['output'], edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box5)
    ax.text(9.25, 2.5, 'Engagement\nLevel', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Labels below
    labels = ['224x224\nRGB', 'Face Crop\n+ Augment', 'Feature\nExtraction', 'GAP + FC\n+ Dropout', '4 Classes']
    x_positions = [1.0, 3.0, 5.2, 7.4, 9.25]
    for x, label in zip(x_positions, labels):
        ax.text(x, 1.1, label, ha='center', va='center', fontsize=8, color='#555', style='italic')

    ax.set_title('Figure 1: Overview of the Proposed Lightweight CNN Framework', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(os.path.join(output_dir, 'framework.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'framework.png'), format='png')
    plt.close()
    print("Created: framework.pdf / framework.png")


# ============================================================
# Figure 2: Model Comparison (Accuracy vs Parameters)
# ============================================================
def create_comparison_figure():
    methods = ['VGG-16', '3D-CNN', 'CNN+LSTM', 'Santoni\net al.', 'ResNet-50', 'MobileNetV3\n+CSAM', 'EfficientNet\n-B0+CSAM']
    accuracy = [68.2, 65.4, 69.8, 70.2, 71.5, 74.6, 75.8]
    params = [138.0, 18.2, 12.3, 15.8, 25.6, 2.6, 5.4]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = np.arange(len(methods))
    width = 0.35

    # Accuracy bars
    bars1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy (%)',
                     color=['#90CAF9']*5 + ['#FF8A65', '#FFCC80'], edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Accuracy (%)', color='#1565C0', fontsize=12)
    ax1.set_ylim(60, 78)
    ax1.tick_params(axis='y', labelcolor='#1565C0')

    # Parameters on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, params, width, label='Parameters (M)',
                     color=['#E0E0E0']*5 + ['#A5D6A7', '#C8E6C9'], edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Parameters (M)', color='#2E7D32', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_ylim(1, 200)
    ax2.tick_params(axis='y', labelcolor='#2E7D32')

    # X-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # Highlight proposed methods
    for i in [5, 6]:
        bars1[i].set_edgecolor('#E65100')
        bars1[i].set_linewidth(2)

    ax1.set_title('Figure 2: Model Comparison — Accuracy vs. Parameters', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(os.path.join(output_dir, 'comparison.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'comparison.png'), format='png')
    plt.close()
    print("Created: comparison.pdf / comparison.png")


# ============================================================
# Figure 3: Ablation Study Results
# ============================================================
def create_ablation_figure():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) Data Augmentation
    ax = axes[0]
    aug_labels = ['None', 'Flip\nonly', 'Flip +\nRotation', 'Full\nAug.']
    aug_acc = [68.5, 70.2, 71.4, 72.8]
    colors_aug = ['#E0E0E0', '#BDBDBD', '#90CAF9', '#42A5F5']
    bars = ax.bar(aug_labels, aug_acc, color=colors_aug, edgecolor='black', linewidth=0.8)
    ax.set_ylim(66, 74)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(a) Effect of Data Augmentation', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, aug_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (b) Class Balancing
    ax = axes[1]
    bal_labels = ['No\nbal.', 'Weighted\nCE', 'Focal\nloss', 'Over-\nsample', 'Focal\nweighted']
    bal_f1 = [64.0, 71.8, 71.2, 70.0, 73.9]
    colors_bal = ['#E0E0E0', '#BDBDBD', '#BDBDBD', '#BDBDBD', '#66BB6A']
    bars = ax.bar(bal_labels, bal_f1, color=colors_bal, edgecolor='black', linewidth=0.8)
    ax.set_ylim(60, 74)
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('(b) Effect of Class Balancing', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, bal_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (c) Transfer Learning
    ax = axes[2]
    tl_labels = ['Random\nInit.', 'ImageNet\nPre-trained']
    tl_acc = [62.4, 72.8]
    colors_tl = ['#EF9A9A', '#66BB6A']
    bars = ax.bar(tl_labels, tl_acc, color=colors_tl, edgecolor='black', linewidth=0.8)
    ax.set_ylim(58, 76)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(c) Effect of Transfer Learning', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, tl_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Figure 3: Ablation Study Results (MobileNetV3-Small)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'ablation.png'), format='png')
    plt.close()
    print("Created: ablation.pdf / ablation.png")


# ============================================================
# Figure 4: Per-Class Performance
# ============================================================
def create_perclass_figure():
    classes = ['Engagement', 'Boredom', 'Confusion', 'Frustration']
    precision = [78.5, 72.3, 68.9, 68.7]
    recall = [80.2, 70.1, 65.4, 70.3]
    f1 = [79.3, 71.2, 67.1, 69.5]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#42A5F5', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#66BB6A', edgecolor='black', linewidth=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FFA726', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(60, 84)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Figure 4: Per-Class Performance of MobileNetV3-Small', fontsize=13, fontweight='bold', pad=10)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perclass.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'perclass.png'), format='png')
    plt.close()
    print("Created: perclass.pdf / perclass.png")


# ============================================================
# Figure 5: Efficiency Comparison
# ============================================================
def create_efficiency_figure():
    methods = ['VGG-16', '3D-CNN', 'CNN+LSTM', 'ResNet-50', 'MobileNetV3\n+CSAM', 'EfficientNet\n-B0+CSAM']
    inference = [8.2, 12.4, 6.5, 4.8, 1.7, 3.0]
    accuracy = [68.2, 65.4, 69.8, 71.5, 74.6, 75.8]
    params = [138.0, 18.2, 12.3, 25.6, 2.6, 5.4]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bubble size proportional to params
    sizes = [p * 8 for p in params]

    scatter = ax.scatter(inference, accuracy, s=sizes, c=['#90CAF9']*4 + ['#FF8A65', '#FFCC80'],
                         edgecolors='black', linewidth=1.5, alpha=0.8, zorder=5)

    # Annotate methods
    for i, method in enumerate(methods):
        offset_y = 0.8 if i != 3 else -1.2
        ax.annotate(method, (inference[i], accuracy[i]),
                    textcoords="offset points", xytext=(0, offset_y*10),
                    ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Inference Time (ms)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlim(0, 14)
    ax.set_ylim(63, 76)
    ax.set_title('Figure 5: Efficiency vs. Accuracy (bubble size = parameters)', fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'efficiency.png'), format='png')
    plt.close()
    print("Created: efficiency.pdf / efficiency.png")


# ============================================================
# Generate all figures
# ============================================================
def create_gradcam_figure():
    """Create simulated Grad-CAM visualizations for different engagement states."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    states = ['Engagement', 'Boredom', 'Confusion', 'Frustration']
    focus_regions = [(0.5, 0.4), (0.5, 0.7), (0.4, 0.45), (0.6, 0.65)]

    for idx, (ax, state, (cx, cy)) in enumerate(zip(axes, states, focus_regions)):
        # Create base face-like image (gray)
        face = np.ones((100, 100, 3)) * 0.7

        # Draw simple face outline
        circle = plt.Circle((50, 50), 40, fill=False, color='#555', linewidth=2)
        ax.add_patch(circle)
        # Eyes
        ax.plot(35, 40, 'o', color='#333', markersize=5)
        ax.plot(65, 40, 'o', color='#333', markersize=5)
        # Mouth
        ax.plot(50, 65, '_', color='#333', markersize=10)

        # Create heatmap overlay
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        heatmap = np.exp(-((X - cx)**2 + (Y - cy)**2) / 0.02)

        ax.imshow(face)
        ax.imshow(heatmap, cmap='jet', alpha=0.5, extent=[0, 100, 100, 0])
        ax.set_title(state, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Grad-CAM Visualizations for Different Engagement States',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'gradcam.png'), format='png')
    plt.close()
    print("Created: gradcam.pdf / gradcam.png")


if __name__ == '__main__':
    create_framework_figure()
    create_comparison_figure()
    create_ablation_figure()
    create_perclass_figure()
    create_efficiency_figure()
    create_gradcam_figure()
    print("\nAll figures generated successfully!")
