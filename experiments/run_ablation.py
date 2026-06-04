#!/usr/bin/env python3
"""
Ablation study runner.

Runs all ablation experiments from the paper:
1. Data augmentation: {none, flip_only, flip_rotation, full}
2. Class balancing: {none, weighted_ce, focal, focal_weighted}
3. Transfer learning: {random_init, imagenet_pretrained}
4. CSAM effect: {with, without} × {MobileNetV3, EfficientNet-B0}

Usage:
    python3 run_ablation.py                          # Run all ablations
    python3 run_ablation.py --ablation augmentation  # Run specific ablation
    python3 run_ablation.py --dry-run                # Quick test
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DataConfig, TrainConfig, ModelConfig, AUGMENTATION_PRESETS
from train import cross_validate, set_seed
from models.mobilenetv3_csam import MobileNetV3CSAM
from models.efficientnet_csam import EfficientNetCSAM


def run_augmentation_ablation(data_cfg, train_cfg, model_cfg, device):
    """Ablation: effect of data augmentation."""
    print("\n" + "=" * 70)
    print("ABLATION: Data Augmentation")
    print("=" * 70)

    results = {}
    presets = ["none", "flip_only", "flip_rotation", "full"]

    for preset in presets:
        print(f"\n--- Augmentation: {preset} ---")

        def model_fn():
            return MobileNetV3CSAM(
                num_classes=data_cfg.num_classes,
                pretrained=True,
                use_csam=False,  # Base MobileNetV3 for this ablation
                csam_reduction=model_cfg.csam_reduction,
                head_hidden=model_cfg.head_hidden,
                head_dropout=model_cfg.head_dropout,
            )

        result = cross_validate(
            model_fn=model_fn,
            cfg=train_cfg,
            data_cfg=data_cfg,
            device=device,
            loss_type="focal_weighted",
            augmentation_preset=preset,
        )
        results[preset] = result
        print(f"  {preset}: Accuracy={result['test_accuracy']}, F1={result['test_f1']}")

    return results


def run_loss_ablation(data_cfg, train_cfg, model_cfg, device):
    """Ablation: effect of class balancing strategies."""
    print("\n" + "=" * 70)
    print("ABLATION: Class Balancing / Loss Function")
    print("=" * 70)

    results = {}
    loss_types = ["none", "weighted_ce", "focal", "focal_weighted"]

    for loss_type in loss_types:
        print(f"\n--- Loss: {loss_type} ---")

        def model_fn():
            return MobileNetV3CSAM(
                num_classes=data_cfg.num_classes,
                pretrained=True,
                use_csam=False,
                csam_reduction=model_cfg.csam_reduction,
                head_hidden=model_cfg.head_hidden,
                head_dropout=model_cfg.head_dropout,
            )

        result = cross_validate(
            model_fn=model_fn,
            cfg=train_cfg,
            data_cfg=data_cfg,
            device=device,
            loss_type=loss_type,
            augmentation_preset="full",
        )
        results[loss_type] = result
        print(f"  {loss_type}: Accuracy={result['test_accuracy']}, F1={result['test_f1']}")

    return results


def run_transfer_learning_ablation(data_cfg, train_cfg, model_cfg, device):
    """Ablation: effect of transfer learning."""
    print("\n" + "=" * 70)
    print("ABLATION: Transfer Learning")
    print("=" * 70)

    results = {}

    for pretrained in [False, True]:
        label = "imagenet" if pretrained else "random"
        print(f"\n--- Initialization: {label} ---")

        def model_fn(p=pretrained):
            return MobileNetV3CSAM(
                num_classes=data_cfg.num_classes,
                pretrained=p,
                use_csam=False,
                csam_reduction=model_cfg.csam_reduction,
                head_hidden=model_cfg.head_hidden,
                head_dropout=model_cfg.head_dropout,
            )

        result = cross_validate(
            model_fn=model_fn,
            cfg=train_cfg,
            data_cfg=data_cfg,
            device=device,
            loss_type="focal_weighted",
            augmentation_preset="full",
        )
        results[label] = result
        print(f"  {label}: Accuracy={result['test_accuracy']}, F1={result['test_f1']}")

    return results


def run_csam_ablation(data_cfg, train_cfg, model_cfg, device):
    """Ablation: effect of CSAM module."""
    print("\n" + "=" * 70)
    print("ABLATION: CSAM Effect")
    print("=" * 70)

    results = {}
    configs = [
        ("mobilenetv3", False),
        ("mobilenetv3_csam", True),
        ("efficientnet_b0", False),
        ("efficientnet_b0_csam", True),
    ]

    for name, use_csam in configs:
        print(f"\n--- {name} (CSAM={use_csam}) ---")

        def model_fn(uc=use_csam, n=name):
            if "mobilenetv3" in n:
                return MobileNetV3CSAM(
                    num_classes=data_cfg.num_classes,
                    pretrained=True,
                    use_csam=uc,
                    csam_reduction=model_cfg.csam_reduction,
                    csam_kernel=model_cfg.csam_kernel,
                    head_hidden=model_cfg.head_hidden,
                    head_dropout=model_cfg.head_dropout,
                )
            else:
                return EfficientNetCSAM(
                    num_classes=data_cfg.num_classes,
                    pretrained=True,
                    use_csam=uc,
                    csam_reduction=model_cfg.csam_reduction,
                    csam_kernel=model_cfg.csam_kernel,
                    head_hidden=model_cfg.head_hidden,
                    head_dropout=model_cfg.head_dropout,
                )

        result = cross_validate(
            model_fn=model_fn,
            cfg=train_cfg,
            data_cfg=data_cfg,
            device=device,
            loss_type="focal_weighted",
            augmentation_preset="full",
        )
        results[name] = result
        print(f"  {name}: Accuracy={result['test_accuracy']}, F1={result['test_f1']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["augmentation", "loss", "transfer", "csam"],
                        help="Run specific ablation study")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    data_cfg = DataConfig()
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()

    if args.seed:
        train_cfg.seed = args.seed
    if args.folds:
        train_cfg.num_folds = args.folds
    if args.dry_run:
        train_cfg.num_folds = 1
        train_cfg.stage1_epochs = 1
        train_cfg.stage2_epochs = 1

    set_seed(train_cfg.seed)
    if args.cpu:
        device = torch.device("cpu")
        torch.backends.cudnn.enabled = False
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}

    ablation_fns = {
        "augmentation": run_augmentation_ablation,
        "loss": run_loss_ablation,
        "transfer": run_transfer_learning_ablation,
        "csam": run_csam_ablation,
    }

    if args.ablation:
        fns = {args.ablation: ablation_fns[args.ablation]}
    else:
        fns = ablation_fns

    for name, fn in fns.items():
        start = time.time()
        result = fn(data_cfg, train_cfg, model_cfg, device)
        elapsed = time.time() - start
        all_results[name] = result
        print(f"\n{name} ablation completed in {elapsed/60:.1f} minutes")

        # Save incremental
        results_path = os.path.join(train_cfg.results_dir, "ablation_results.json")
        os.makedirs(train_cfg.results_dir, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll ablation results saved to {train_cfg.results_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
