"""Central configuration for all experiments."""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class DataConfig:
    # Paths
    raw_dir: str = os.path.join(PROJECT_ROOT, "data", "DAiSEE")
    processed_dir: str = os.path.join(PROJECT_ROOT, "data", "processed")
    labels_dir: str = os.path.join(PROJECT_ROOT, "data", "DAiSEE", "Labels")

    # Frame extraction
    fps: float = 1.0  # frames per second to extract
    face_margin: float = 0.20  # 20% margin for face crop

    # Image
    image_size: int = 224
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: ["Engagement", "Boredom", "Confusion", "Frustration"])

    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ImageNet normalization
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    # General
    seed: int = 42
    num_workers: int = 4
    batch_size: int = 32
    num_epochs: int = 40
    num_folds: int = 5

    # Stage 1: train head only
    stage1_epochs: int = 10
    stage1_lr: float = 1e-3

    # Stage 2: fine-tune backbone
    stage2_epochs: int = 30
    stage2_lr: float = 1e-4
    unfreeze_blocks: int = 3  # last N blocks to unfreeze

    # Optimizer
    weight_decay: float = 1e-4

    # Focal weighted loss
    focal_gamma: float = 2.0

    # Paths
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "experiments", "checkpoints")
    results_dir: str = os.path.join(PROJECT_ROOT, "experiments", "results")


@dataclass
class ModelConfig:
    # CSAM
    csam_reduction: int = 4
    csam_kernel: int = 7

    # Classification head
    head_hidden: int = 256
    head_dropout: float = 0.5

    # Backbone choices
    backbone: str = "mobilenetv3_small"  # or "efficientnet_b0"
    use_csam: bool = True
    pretrained: bool = True


# Augmentation presets for ablation
AUGMENTATION_PRESETS = {
    "none": {
        "horizontal_flip": 0.0,
        "rotation": 0,
        "brightness": 0.0,
        "contrast": 0.0,
        "crop_scale": (1.0, 1.0),
        "hue": 0.0,
    },
    "flip_only": {
        "horizontal_flip": 0.5,
        "rotation": 0,
        "brightness": 0.0,
        "contrast": 0.0,
        "crop_scale": (1.0, 1.0),
        "hue": 0.0,
    },
    "flip_rotation": {
        "horizontal_flip": 0.5,
        "rotation": 15,
        "brightness": 0.0,
        "contrast": 0.0,
        "crop_scale": (1.0, 1.0),
        "hue": 0.0,
    },
    "full": {
        "horizontal_flip": 0.5,
        "rotation": 15,
        "brightness": 0.2,
        "contrast": 0.2,
        "crop_scale": (0.8, 1.0),
        "hue": 0.1,
    },
}
