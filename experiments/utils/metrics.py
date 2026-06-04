"""Evaluation metrics for engagement detection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred, num_classes=4):
    """Compute all evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def format_metrics(metrics: dict, class_names=None):
    """Format metrics dict as a readable string."""
    if class_names is None:
        class_names = ["Engagement", "Boredom", "Confusion", "Frustration"]

    lines = [
        f"Accuracy:  {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision_macro']:.4f} (macro)",
        f"Recall:    {metrics['recall_macro']:.4f} (macro)",
        f"F1-Score:  {metrics['f1_macro']:.4f} (macro)",
        "",
        "Per-class F1:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {name}: P={metrics['precision_per_class'][i]:.4f} "
                      f"R={metrics['recall_per_class'][i]:.4f} "
                      f"F1={metrics['f1_per_class'][i]:.4f}")

    return "\n".join(lines)
