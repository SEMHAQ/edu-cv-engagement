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

    Numerically stable: uses log_softmax internally to avoid double
    softmax computation.  Focal weight is detached from the computation
    graph of p_t to prevent gradient vanishing in early training.

    Args:
        class_weights: Tensor of shape (C,) with weight per class.
        gamma: Focusing parameter. Lower values (1.0) are more stable
               in early training than the original 2.0.
        label_smoothing: Label smoothing factor (0 = no smoothing).
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        class_weights: torch.Tensor = None,
        gamma: float = 1.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
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
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)          # (B, C)
        probs = log_probs.exp()                             # (B, C)

        # Gather log-prob of the true class
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        p_t = log_p_t.exp()                                              # (B,)

        # Focal modulation — detach so focal weight doesn't kill gradients
        focal_weight = ((1 - p_t) ** self.gamma).detach()   # (B,)

        # Class weight
        if self.class_weights is not None:
            w_t = self.class_weights[targets]                 # (B,)
        else:
            w_t = torch.ones_like(targets, dtype=torch.float)

        # Standard CE per sample (via log_softmax — numerically stable)
        ce_loss = -log_p_t                                    # (B,)

        # Optional label smoothing: blend uniform target with hard target
        if self.label_smoothing > 0:
            smooth_loss = -log_probs.mean(dim=1)              # (B,)
            ce_loss = (1 - self.label_smoothing) * ce_loss + self.label_smoothing * smooth_loss

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
        log_probs = F.log_softmax(logits, dim=1)
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        focal_weight = ((1 - p_t) ** self.gamma).detach()
        ce_loss = -log_p_t
        return (focal_weight * ce_loss).mean()


def build_loss(
    loss_type: str = "focal_weighted",
    class_weights: torch.Tensor = None,
    gamma: float = 1.0,
    label_smoothing: float = 0.1,
):
    """Factory to build loss functions for ablation studies."""
    if loss_type == "focal_weighted":
        return FocalWeightedLoss(
            class_weights=class_weights, gamma=gamma,
            label_smoothing=label_smoothing,
        )
    elif loss_type == "weighted_ce":
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    elif loss_type == "focal":
        return FocalLoss(gamma=gamma)
    elif loss_type == "none":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
