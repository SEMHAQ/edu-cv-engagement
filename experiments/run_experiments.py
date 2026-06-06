#!/usr/bin/env python3
"""
Main experiment runner for multi-dataset engagement detection.

Runs experiments on FER2013, RAF-DB, and CK+ datasets.

Usage:
    python3 run_experiments.py                        # Run all datasets, all models
    python3 run_experiments.py --dataset fer2013      # Run specific dataset
    python3 run_experiments.py --model mobilenetv3_csam --dataset ckplus
    python3 run_experiments.py --dry-run --cpu        # Quick test
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DataConfig, TrainConfig, ModelConfig, DATASET_EMOTION_MAPS, PROJECT_ROOT
from train import cross_validate, set_seed
from models.mobilenetv3_csam import MobileNetV3CSAM
from models.efficientnet_csam import EfficientNetCSAM
from models.baselines import build_model
from efficiency import efficiency_report


# All models to compare
EXPERIMENT_CONFIGS = {
    # Baselines
    "vgg16": {
        "type": "baseline",
        "model_name": "vgg16",
        "description": "VGG-16 baseline",
    },
    "resnet50": {
        "type": "baseline",
        "model_name": "resnet50",
        "description": "ResNet-50 baseline",
    },
    "cnn_lstm": {
        "type": "baseline",
        "model_name": "cnn_lstm",
        "description": "CNN+LSTM baseline",
    },
    "3d_cnn": {
        "type": "baseline",
        "model_name": "3d_cnn",
        "description": "3D-CNN baseline",
    },
    # Proposed models
    "mobilenetv3": {
        "type": "proposed",
        "backbone": "mobilenetv3_small",
        "use_csam": False,
        "description": "MobileNetV3-Small (no CSAM)",
    },
    "mobilenetv3_csam": {
        "type": "proposed",
        "backbone": "mobilenetv3_small",
        "use_csam": True,
        "description": "MobileNetV3-Small + CSAM (proposed)",
    },
    "efficientnet_b0": {
        "type": "proposed",
        "backbone": "efficientnet_b0",
        "use_csam": False,
        "description": "EfficientNet-B0 (no CSAM)",
    },
    "efficientnet_b0_csam": {
        "type": "proposed",
        "backbone": "efficientnet_b0",
        "use_csam": True,
        "description": "EfficientNet-B0 + CSAM (proposed)",
    },
}


def build_model_from_config(exp_cfg: dict, model_cfg: ModelConfig, num_classes: int = 4):
    """Build model from experiment config dict."""
    if exp_cfg["type"] == "baseline":
        return build_model(exp_cfg["model_name"], num_classes=num_classes, pretrained=True)
    else:
        if "mobilenetv3" in exp_cfg["backbone"]:
            return MobileNetV3CSAM(
                num_classes=num_classes,
                pretrained=True,
                use_csam=exp_cfg["use_csam"],
                csam_reduction=model_cfg.csam_reduction,
                csam_kernel=model_cfg.csam_kernel,
                head_hidden=model_cfg.head_hidden,
                head_dropout=model_cfg.head_dropout,
            )
        elif "efficientnet" in exp_cfg["backbone"]:
            return EfficientNetCSAM(
                num_classes=num_classes,
                pretrained=True,
                use_csam=exp_cfg["use_csam"],
                csam_reduction=model_cfg.csam_reduction,
                csam_kernel=model_cfg.csam_kernel,
                head_hidden=model_cfg.head_hidden,
                head_dropout=model_cfg.head_dropout,
            )


def run_experiment(exp_name: str, exp_cfg: dict, data_cfg: DataConfig,
                   train_cfg: TrainConfig, model_cfg: ModelConfig,
                   device: torch.device) -> dict:
    """Run a single experiment with cross-validation."""
    print(f"\n{'#'*70}")
    print(f"# Experiment: {exp_name} on {data_cfg.dataset_name}")
    print(f"# {exp_cfg['description']}")
    print(f"{'#'*70}")

    def model_fn():
        return build_model_from_config(exp_cfg, model_cfg, data_cfg.num_classes)

    start_time = time.time()
    result = cross_validate(
        model_fn=model_fn,
        cfg=train_cfg,
        data_cfg=data_cfg,
        device=device,
        loss_type="focal_weighted",
        augmentation_preset="full",
    )
    elapsed = time.time() - start_time

    result["experiment"] = exp_name
    result["dataset"] = data_cfg.dataset_name
    result["description"] = exp_cfg["description"]
    result["training_time_minutes"] = elapsed / 60

    print(f"\n[RESULT] {exp_name} on {data_cfg.dataset_name}:")
    print(f"  Test Accuracy: {result['test_accuracy']}")
    print(f"  Test F1:       {result['test_f1']}")
    print(f"  Training time: {elapsed/60:.1f} minutes")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run engagement detection experiments")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="Dataset(s) to use: fer2013, rafdb, ckplus")
    parser.add_argument("--model", type=str, default=None,
                        help="Run single model (e.g., mobilenetv3_csam)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Test with 1 epoch, 1 fold")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    args = parser.parse_args()

    # Base configs
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()

    if args.seed:
        train_cfg.seed = args.seed
    if args.epochs:
        train_cfg.num_epochs = args.epochs
        train_cfg.stage1_epochs = max(1, args.epochs // 4)
        train_cfg.stage2_epochs = args.epochs - train_cfg.stage1_epochs
    if args.folds:
        train_cfg.num_folds = args.folds
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.dry_run:
        train_cfg.num_folds = 1
        train_cfg.stage1_epochs = 1
        train_cfg.stage2_epochs = 1
        train_cfg.num_epochs = 2

    set_seed(train_cfg.seed)
    if args.cpu:
        device = torch.device("cpu")
        torch.backends.cudnn.enabled = False
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {train_cfg.seed}")

    # Select datasets
    available_datasets = list(DATASET_EMOTION_MAPS.keys())
    if args.dataset:
        datasets = args.dataset
    else:
        datasets = available_datasets

    # Select experiments
    if args.model:
        if args.model not in EXPERIMENT_CONFIGS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(EXPERIMENT_CONFIGS.keys())}")
            sys.exit(1)
        experiments = {args.model: EXPERIMENT_CONFIGS[args.model]}
    elif args.skip_baselines:
        experiments = {k: v for k, v in EXPERIMENT_CONFIGS.items() if v["type"] == "proposed"}
    else:
        experiments = EXPERIMENT_CONFIGS

    # Run experiments per dataset
    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")

        data_cfg = DataConfig()
        data_cfg.set_dataset(dataset_name)

        # Dataset-specific results directory
        train_cfg.results_dir = os.path.join(PROJECT_ROOT, "experiments", "results", dataset_name)
        os.makedirs(train_cfg.results_dir, exist_ok=True)

        # Check if preprocessed data exists
        if not os.path.isdir(data_cfg.processed_dir):
            print(f"[SKIP] {dataset_name}: preprocessed data not found at {data_cfg.processed_dir}")
            print(f"  Run: python3 data/preprocess.py --dataset {dataset_name}")
            continue

        dataset_results = {}
        for exp_name, exp_cfg in experiments.items():
            result = run_experiment(exp_name, exp_cfg, data_cfg, train_cfg, model_cfg, device)
            dataset_results[exp_name] = result

            # Save incremental results
            all_results[dataset_name] = dataset_results
            results_path = os.path.join(train_cfg.results_dir, "all_experiments.json")
            os.makedirs(train_cfg.results_dir, exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        all_results[dataset_name] = dataset_results

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for dataset_name, dataset_results in all_results.items():
        print(f"\n--- {dataset_name} ---")
        print(f"{'Model':<25} {'Accuracy':>12} {'F1':>12} {'Time (min)':>12}")
        print("-" * 65)
        for name, result in dataset_results.items():
            print(f"{name:<25} {result['test_accuracy']:>12} {result['test_f1']:>12} "
                  f"{result['training_time_minutes']:>12.1f}")

    # Save summary
    summary_path = os.path.join(train_cfg.results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for dataset_name, dataset_results in all_results.items():
            f.write(f"\n{dataset_name}\n{'='*60}\n")
            for name, result in dataset_results.items():
                f.write(f"{name}: Acc={result['test_accuracy']}, F1={result['test_f1']}\n")

    print(f"\nResults saved to {train_cfg.results_dir}")

    # Efficiency analysis (run once, dataset-independent)
    print(f"\n{'='*70}")
    print("EFFICIENCY ANALYSIS")
    print(f"{'='*70}")
    eff_models = {}
    for name, exp_cfg in experiments.items():
        model = build_model_from_config(exp_cfg, model_cfg, num_classes=data_cfg.num_classes)
        eff_models[name] = model.to(device)

    eff_report_result = efficiency_report(eff_models, device)
    eff_path = os.path.join(train_cfg.results_dir, "efficiency.json")
    with open(eff_path, "w") as f:
        json.dump(eff_report_result, f, indent=2)

    print(f"\nAll experiments complete!")


if __name__ == "__main__":
    main()
