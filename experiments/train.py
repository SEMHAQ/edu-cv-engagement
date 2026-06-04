"""
Training loop with two-stage fine-tuning and 5-fold cross-validation.

Stage 1: Train classification head only (10 epochs, LR=1e-3)
Stage 2: Unfreeze last 3 backbone blocks, fine-tune all (30 epochs, LR=1e-4)
"""

import os
import sys
import time
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DataConfig, TrainConfig, ModelConfig
from data.dataset import get_dataloaders
from losses.focal_weighted import FocalWeightedLoss, build_loss
from utils.metrics import compute_metrics, format_metrics
from utils.logger import ExperimentLogger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Allow TF32 for better cuDNN compatibility
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    freeze_bn: bool = False, grad_clip_norm: float = 0.0):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    if freeze_bn:
        model.freeze_bn()  # Re-freeze BN after model.train() sets everything to train mode
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes=4):
    """Evaluate model, return loss, accuracy, and full metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="  Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = avg_loss

    return metrics


def train_fold(
    model,
    loaders: dict,
    cfg: TrainConfig,
    data_cfg: DataConfig,
    fold_idx: int,
    device: torch.device,
    loss_type: str = "weighted_ce",
    augmentation_preset: str = "full",
) -> dict:
    """Train model for one fold. Returns best validation metrics."""
    class_weights = loaders["class_weights"].to(device)
    criterion = build_loss(
        loss_type, class_weights=class_weights,
        gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing,
    )

    logger = ExperimentLogger(cfg.results_dir, f"fold_{fold_idx}")
    logger.log_config({
        "fold": fold_idx,
        "loss_type": loss_type,
        "augmentation": augmentation_preset,
    })

    best_val_acc = 0.0
    best_state = None
    best_metrics = None

    # ---- Stage 1: Train head only ----
    print(f"\n  [Fold {fold_idx}] Stage 1: Training head ({cfg.stage1_epochs} epochs, LR={cfg.stage1_lr})")
    model.freeze_backbone()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.stage1_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.stage1_epochs)

    for epoch in range(cfg.stage1_epochs):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        scheduler.step()

        val_metrics = evaluate(model, loaders["val"], criterion, device, data_cfg.num_classes)
        logger.log_epoch(epoch, {"train_loss": train_loss, "train_acc": train_acc}, phase="train")
        logger.log_epoch(epoch, val_metrics, phase="val")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics

        print(f"    Epoch {epoch+1}/{cfg.stage1_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}")

    # ---- Stage 2: Fine-tune backbone ----
    print(f"  [Fold {fold_idx}] Stage 2: Fine-tuning ({cfg.stage2_epochs} epochs, LR={cfg.stage2_lr})")
    model.unfreeze_last_n_blocks(cfg.unfreeze_blocks)
    model.freeze_bn()  # Keep BN running stats frozen to prevent collapse
    optimizer = Adam(
        [
            {"params": model.get_backbone_params(), "lr": cfg.stage2_lr},
            {"params": model.get_head_params(), "lr": cfg.stage2_lr * 10},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.stage2_epochs)

    for epoch in range(cfg.stage2_epochs):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
            freeze_bn=True, grad_clip_norm=cfg.grad_clip_norm,
        )
        scheduler.step()

        val_metrics = evaluate(model, loaders["val"], criterion, device, data_cfg.num_classes)
        logger.log_epoch(cfg.stage1_epochs + epoch, {"train_loss": train_loss, "train_acc": train_acc}, phase="train")
        logger.log_epoch(cfg.stage1_epochs + epoch, val_metrics, phase="val")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics

        print(f"    Epoch {epoch+1}/{cfg.stage2_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}")

    # Save best model
    if best_state is not None:
        save_path = os.path.join(cfg.checkpoint_dir, f"fold_{fold_idx}_best.pth")
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        torch.save(best_state, save_path)
        model.load_state_dict(best_state)

    # Evaluate on test set
    test_metrics = evaluate(model, loaders["test"], criterion, device, data_cfg.num_classes)
    logger.log_result({"val": best_metrics, "test": test_metrics})

    print(f"  [Fold {fold_idx}] Test Accuracy: {test_metrics['accuracy']:.4f} "
          f"F1: {test_metrics['f1_macro']:.4f}")

    return best_metrics, test_metrics


def cross_validate(
    model_fn,
    cfg: TrainConfig,
    data_cfg: DataConfig,
    device: torch.device,
    loss_type: str = "weighted_ce",
    augmentation_preset: str = "full",
    use_weighted_sampler: bool = True,
) -> dict:
    """Run k-fold cross-validation. Returns aggregated results."""
    all_val_metrics = []
    all_test_metrics = []

    for fold_idx in range(cfg.num_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{cfg.num_folds}")
        print(f"{'='*60}")

        # Get dataloaders for this fold
        loaders = get_dataloaders(
            data_cfg, cfg,
            augmentation_preset=augmentation_preset,
            fold_idx=fold_idx,
            use_weighted_sampler=use_weighted_sampler,
        )

        # Create fresh model for each fold
        model = model_fn()
        model = model.to(device)

        val_metrics, test_metrics = train_fold(
            model, loaders, cfg, data_cfg, fold_idx, device,
            loss_type=loss_type,
            augmentation_preset=augmentation_preset,
        )

        all_val_metrics.append(val_metrics)
        all_test_metrics.append(test_metrics)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Aggregate results
    result = aggregate_results(all_val_metrics, all_test_metrics, cfg.num_folds)
    return result


def aggregate_results(all_val_metrics, all_test_metrics, num_folds):
    """Aggregate cross-validation results: mean ± std."""
    def mean_std(key, metrics_list):
        vals = [m[key] for m in metrics_list]
        return float(np.mean(vals)), float(np.std(vals))

    result = {
        "num_folds": num_folds,
    }

    for split, metrics_list in [("val", all_val_metrics), ("test", all_test_metrics)]:
        acc_mean, acc_std = mean_std("accuracy", metrics_list)
        prec_mean, prec_std = mean_std("precision_macro", metrics_list)
        rec_mean, rec_std = mean_std("recall_macro", metrics_list)
        f1_mean, f1_std = mean_std("f1_macro", metrics_list)

        result[f"{split}_accuracy"] = f"{acc_mean*100:.1f} ± {acc_std*100:.1f}"
        result[f"{split}_precision"] = f"{prec_mean*100:.1f} ± {prec_std*100:.1f}"
        result[f"{split}_recall"] = f"{rec_mean*100:.1f} ± {rec_std*100:.1f}"
        result[f"{split}_f1"] = f"{f1_mean*100:.1f} ± {f1_std*100:.1f}"
        result[f"{split}_accuracy_raw"] = (acc_mean, acc_std)
        result[f"{split}_f1_raw"] = (f1_mean, f1_std)

        # Per-class F1
        per_class_f1 = []
        for i in range(len(metrics_list[0]["f1_per_class"])):
            vals = [m["f1_per_class"][i] for m in metrics_list]
            per_class_f1.append((float(np.mean(vals)), float(np.std(vals))))
        result[f"{split}_per_class_f1"] = per_class_f1

    return result
