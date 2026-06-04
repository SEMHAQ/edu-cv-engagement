"""PyTorch Dataset and DataLoader for preprocessed DAiSEE face images."""

import os
import random
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig, TrainConfig, AUGMENTATION_PRESETS


class DAiSEEDataset(Dataset):
    """Dataset for preprocessed DAiSEE face images."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label


def build_transforms(cfg: DataConfig, augmentation_preset: str = "full", is_train: bool = True):
    """Build transform pipeline from augmentation preset."""
    if not is_train:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ])

    aug = AUGMENTATION_PRESETS.get(augmentation_preset, AUGMENTATION_PRESETS["full"])

    t = []
    t.append(transforms.Resize((cfg.image_size, cfg.image_size)))

    if aug["horizontal_flip"] > 0:
        t.append(transforms.RandomHorizontalFlip(p=aug["horizontal_flip"]))

    if aug["rotation"] > 0:
        t.append(transforms.RandomRotation(degrees=aug["rotation"]))

    if aug["crop_scale"][0] < 1.0:
        t.append(transforms.RandomResizedCrop(
            cfg.image_size, scale=aug["crop_scale"], ratio=(0.9, 1.1)
        ))

    if aug["brightness"] > 0 or aug["contrast"] > 0 or aug["hue"] > 0:
        t.append(transforms.ColorJitter(
            brightness=aug.get("brightness", 0),
            contrast=aug.get("contrast", 0),
            hue=aug.get("hue", 0),
        ))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=cfg.mean, std=cfg.std))

    return transforms.Compose(t)


def scan_processed_dir(processed_dir: str, split: str) -> Tuple[List[str], List[int]]:
    """Scan processed directory for images and labels."""
    split_dir = os.path.join(processed_dir, split)
    image_paths = []
    labels = []

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Processed split directory not found: {split_dir}")

    for class_id in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_id)
        if not os.path.isdir(class_dir):
            continue
        label = int(class_id)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label)

    return image_paths, labels


def get_class_weights(labels: List[int], num_classes: int = 4) -> torch.Tensor:
    """Compute class weights: N / (C * n_i)."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    N = len(labels)
    weights = N / (num_classes * counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(labels: List[int], num_classes: int = 4) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for class-balanced sampling."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    class_weights = 1.0 / (counts + 1e-8)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def get_dataloaders(
    cfg: DataConfig,
    train_cfg: TrainConfig,
    augmentation_preset: str = "full",
    fold_idx: int = 0,
    use_weighted_sampler: bool = False,
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders."""
    # Load all data
    train_paths, train_labels = scan_processed_dir(cfg.processed_dir, "train")
    val_paths, val_labels = scan_processed_dir(cfg.processed_dir, "validation")
    test_paths, test_labels = scan_processed_dir(cfg.processed_dir, "test")

    if train_cfg.num_folds <= 1:
        # No cross-validation: use the provided train/val split directly
        fold_train_paths = train_paths
        fold_train_labels = train_labels
        fold_val_paths = val_paths
        fold_val_labels = val_labels
    else:
        # For k-fold: merge train+val, then split
        all_paths = train_paths + val_paths
        all_labels = train_labels + val_labels

        n = len(all_paths)
        fold_size = n // train_cfg.num_folds
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < train_cfg.num_folds - 1 else n

        fold_val_paths = all_paths[val_start:val_end]
        fold_val_labels = all_labels[val_start:val_end]
        fold_train_paths = all_paths[:val_start] + all_paths[val_end:]
        fold_train_labels = all_labels[:val_start] + all_labels[val_end:]

    # Build transforms
    train_transform = build_transforms(cfg, augmentation_preset, is_train=True)
    val_transform = build_transforms(cfg, augmentation_preset, is_train=False)

    # Create datasets
    train_ds = DAiSEEDataset(fold_train_paths, fold_train_labels, train_transform)
    val_ds = DAiSEEDataset(fold_val_paths, fold_val_labels, val_transform)
    test_ds = DAiSEEDataset(test_paths, test_labels, val_transform)

    # Create dataloaders
    train_sampler = None
    shuffle = True
    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(fold_train_labels, cfg.num_classes)
        shuffle = False

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=train_cfg.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
        ),
    }

    # Also return class weights
    class_weights = get_class_weights(fold_train_labels, cfg.num_classes)
    loaders["class_weights"] = class_weights
    loaders["train_labels"] = fold_train_labels

    return loaders
