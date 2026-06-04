"""
Focal Weighted Loss

Combines class weighting (N / (C * n_i)) with focal modulation (1 - p)^gamma.
Jointly addresses class imbalance and hard-example mining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalWeightedLoss(nn.Module):
    """
    Focal Weighted Loss = -w_y * (1 - p_y)^gamma * log(p_y)

    Args:
        class_weights: Tensor of shape (C,) with weight per class.
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        class_weights: torch.Tensor = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw model output
            targets: (B,) integer class labels
        """
        probs = F.softmax(logits, dim=1)
        # Gather the probability of the true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        p_t = (probs * targets_one_hot).sum(dim=1)  # (B,)

        # Focal modulation
        focal_weight = (1 - p_t) ** self.gamma

        # Class weight
        if self.class_weights is not None:
            w_t = self.class_weights[targets]  # (B,)
        else:
            w_t = 1.0

        # Cross entropy per sample
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Combined loss
        loss = w_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Standard weighted cross-entropy for ablation comparison."""

    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.class_weights)


class FocalLoss(nn.Module):
    """Standard focal loss without class weighting, for ablation comparison."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        return (focal_weight * ce_loss).mean()


def build_loss(
    loss_type: str = "focal_weighted",
    class_weights: torch.Tensor = None,
    gamma: float = 2.0,
):
    """Factory to build loss functions for ablation studies."""
    if loss_type == "focal_weighted":
        return FocalWeightedLoss(class_weights=class_weights, gamma=gamma)
    elif loss_type == "weighted_ce":
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    elif loss_type == "focal":
        return FocalLoss(gamma=gamma)
    elif loss_type == "none":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
