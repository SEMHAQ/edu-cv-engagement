#!/usr/bin/env python3
"""
Update paper figures with real experimental results.

Reads results from experiments/results/ and regenerates all figures
with actual data instead of simulated values.

Usage:
    python3 update_figures.py                          # Use default results path
    python3 update_figures.py --results-dir path/to/results
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Publication-quality defaults
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_results(results_dir):
    """Load experiment results from JSON files."""
    results = {}

    main_path = os.path.join(results_dir, "all_experiments.json")
    if os.path.exists(main_path):
        with open(main_path) as f:
            results["main"] = json.load(f)

    ablation_path = os.path.join(results_dir, "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            results["ablation"] = json.load(f)

    eff_path = os.path.join(results_dir, "efficiency.json")
    if os.path.exists(eff_path):
        with open(eff_path) as f:
            results["efficiency"] = json.load(f)

    return results


def parse_accuracy(s):
    """Parse '74.6 ± 0.7' string to float."""
    if isinstance(s, (int, float)):
        return float(s)
    return float(s.split("±")[0].strip())


def parse_std(s):
    """Parse '74.6 ± 0.7' to std float."""
    if isinstance(s, (int, float)):
        return 0.0
    parts = s.split("±")
    if len(parts) < 2:
        return 0.0
    return float(parts[1].strip())


def update_comparison_figure(results, output_dir):
    """Update Figure 2: Model comparison bar chart."""
    if "main" not in results:
        print("[SKIP] No main experiment results for comparison figure")
        return

    main = results["main"]
    # Order: baselines first, then proposed
    order = ["vgg16", "3d_cnn", "cnn_lstm", "resnet50", "mobilenetv3", "mobilenetv3_csam",
             "efficientnet_b0", "efficientnet_b0_csam"]
    order = [k for k in order if k in main]

    methods = []
    accuracy = []
    accuracy_std = []
    params = []

    for name in order:
        r = main[name]
        label = name.replace("_", " ").replace("csam", "+CSAM").replace("b0", "-B0")
        label = label.replace("mobilenetv3", "MobileNetV3").replace("efficientnet", "EfficientNet")
        label = label.replace("vgg16", "VGG-16").replace("resnet50", "ResNet-50")
        label = label.replace("cnn lstm", "CNN+LSTM").replace("3d cnn", "3D-CNN")
        methods.append(label)
        accuracy.append(parse_accuracy(r.get("test_accuracy", 0)))
        accuracy_std.append(parse_std(r.get("test_accuracy", 0)))

        # Params from efficiency data if available
        if "efficiency" in results and name in results["efficiency"]:
            params.append(results["efficiency"][name]["params_M"])
        else:
            params.append(0)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accuracy, width, yerr=accuracy_std,
                     label="Accuracy (%)", color=["#90CAF9"]*5 + ["#FF8A65"]*2 + ["#FFCC80"],
                     edgecolor="black", linewidth=0.8, capsize=3)
    ax1.set_ylabel("Accuracy (%)", color="#1565C0", fontsize=12)
    ax1.set_ylim(60, 80)
    ax1.tick_params(axis="y", labelcolor="#1565C0")

    if any(p > 0 for p in params):
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, params, width,
                         label="Parameters (M)", color=["#E0E0E0"]*5 + ["#A5D6A7"]*2 + ["#C8E6C9"],
                         edgecolor="black", linewidth=0.8)
        ax2.set_ylabel("Parameters (M)", color="#2E7D32", fontsize=12)
        ax2.set_yscale("log")
        ax2.set_ylim(1, 200)
        ax2.tick_params(axis="y", labelcolor="#2E7D32")

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left", fontsize=10)
    ax1.set_title("Figure 2: Model Comparison — Accuracy vs. Parameters", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_real.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "comparison_real.png"), format="png")
    plt.close()
    print("Created: comparison_real.pdf / comparison_real.png")


def update_ablation_figure(results, output_dir):
    """Update Figure 3: Ablation study results."""
    if "ablation" not in results:
        print("[SKIP] No ablation results")
        return

    ablation = results["ablation"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) Data Augmentation
    if "augmentation" in ablation:
        ax = axes[0]
        aug_data = ablation["augmentation"]
        presets = ["none", "flip_only", "flip_rotation", "full"]
        labels = ["None", "Flip\nonly", "Flip +\nRotation", "Full\nAug."]
        vals = [parse_accuracy(aug_data.get(p, {}).get("test_accuracy", 0)) * 100 for p in presets]
        colors = ["#E0E0E0", "#BDBDBD", "#90CAF9", "#42A5F5"]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_ylim(min(vals) - 2, max(vals) + 2)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("(a) Effect of Data Augmentation", fontsize=11, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # (b) Class Balancing
    if "loss" in ablation:
        ax = axes[1]
        loss_data = ablation["loss"]
        loss_types = ["none", "weighted_ce", "focal", "focal_weighted"]
        labels = ["No\nbal.", "Weighted\nCE", "Focal\nloss", "Focal\nweighted"]
        vals = [parse_accuracy(loss_data.get(l, {}).get("test_f1", 0)) * 100 for l in loss_types]
        colors = ["#E0E0E0", "#BDBDBD", "#BDBDBD", "#66BB6A"]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_ylim(min(vals) - 2, max(vals) + 2)
        ax.set_ylabel("F1-Score (%)")
        ax.set_title("(b) Effect of Class Balancing", fontsize=11, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # (c) Transfer Learning
    if "transfer" in ablation:
        ax = axes[2]
        tl_data = ablation["transfer"]
        labels = ["Random\nInit.", "ImageNet\nPre-trained"]
        vals = [parse_accuracy(tl_data.get(k, {}).get("test_accuracy", 0)) * 100
                for k in ["random", "imagenet"]]
        colors = ["#EF9A9A", "#66BB6A"]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_ylim(min(vals) - 2, max(vals) + 2)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("(c) Effect of Transfer Learning", fontsize=11, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Figure 3: Ablation Study Results (MobileNetV3-Small)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_real.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "ablation_real.png"), format="png")
    plt.close()
    print("Created: ablation_real.pdf / ablation_real.png")


def update_efficiency_figure(results, output_dir):
    """Update Figure 5: Efficiency bubble chart."""
    if "efficiency" not in results:
        print("[SKIP] No efficiency results")
        return

    eff = results["efficiency"]
    # Get accuracy from main results if available
    main = results.get("main", {})

    methods = []
    inference = []
    accuracy_list = []
    params = []

    for name, data in eff.items():
        label = name.replace("_", " ").replace("csam", "+CSAM").replace("b0", "-B0")
        label = label.replace("mobilenetv3", "MobileNetV3").replace("efficientnet", "EfficientNet")
        methods.append(label)
        inference.append(data["inference_ms"])
        params.append(data["params_M"])
        if name in main:
            accuracy_list.append(parse_accuracy(main[name].get("test_accuracy", 0)) * 100)
        else:
            accuracy_list.append(70)  # placeholder

    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = [p * 8 for p in params]
    colors = ["#90CAF9"] * max(0, len(methods) - 2) + ["#FF8A65", "#FFCC80"]
    if len(colors) < len(methods):
        colors = ["#90CAF9"] * len(methods)

    scatter = ax.scatter(inference, accuracy_list, s=sizes[:len(methods)],
                         c=colors[:len(methods)], edgecolors="black", linewidth=1.5, alpha=0.8, zorder=5)

    for i, method in enumerate(methods):
        offset_y = 0.8
        ax.annotate(method, (inference[i], accuracy_list[i]),
                    textcoords="offset points", xytext=(0, offset_y * 10),
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Inference Time (ms)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, max(inference) * 1.2)
    ax.set_ylim(min(accuracy_list) - 2, max(accuracy_list) + 2)
    ax.set_title("Figure 5: Efficiency vs. Accuracy (bubble size = parameters)", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency_real.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "efficiency_real.png"), format="png")
    plt.close()
    print("Created: efficiency_real.pdf / efficiency_real.png")


def main():
    parser = argparse.ArgumentParser(description="Update figures with real results")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = args.results_dir or os.path.join(project_root, "experiments", "results")
    output_dir = args.output_dir or os.path.join(project_root, "paper", "figures")

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        print("Run experiments first: python3 run_experiments.py")
        sys.exit(1)

    results = load_results(results_dir)
    if not results:
        print("No result files found in", results_dir)
        sys.exit(1)

    print(f"Loaded results: {list(results.keys())}")
    os.makedirs(output_dir, exist_ok=True)

    update_comparison_figure(results, output_dir)
    update_ablation_figure(results, output_dir)
    update_efficiency_figure(results, output_dir)

    print(f"\nFigures saved to {output_dir}")


if __name__ == "__main__":
    main()
