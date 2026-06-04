"""Central configuration for all experiments (FER2013 / RAF-DB / CK+)."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Emotion → Engagement label mapping ────────────────────────────────────────
# Happy → 0:Engaged, Neutral → 1:Boredom, Sad/Anger → 2:Frustration, Surprise/Fear → 3:Confusion
# Disgust and Contempt are excluded (ambiguous, small sample count).

ENGAGEMENT_CLASSES = ["Engaged", "Boredom", "Frustration", "Confusion"]

# FER2013 original labels: {0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise, 6:Neutral}
FER2013_EMOTION_MAP = {
    3: 0,  # Happy   → Engaged
    6: 1,  # Neutral → Boredom
    4: 2,  # Sad     → Frustration
    0: 2,  # Angry   → Frustration
    5: 3,  # Surprise→ Confusion
    2: 3,  # Fear    → Confusion
    # 1:Disgust → excluded
}

# RAF-DB original labels (1-indexed): {1:Surprise, 2:Fear, 3:Disgust, 4:Happy, 5:Sad, 6:Anger, 7:Neutral}
RAFDB_EMOTION_MAP = {
    4: 0,  # Happy    → Engaged
    7: 1,  # Neutral  → Boredom
    5: 2,  # Sad      → Frustration
    6: 2,  # Anger    → Frustration
    1: 3,  # Surprise → Confusion
    2: 3,  # Fear     → Confusion
    # 3:Disgust → excluded
}

# CK+ original labels (0-indexed): {0:Neutral, 1:Anger, 2:Contempt, 3:Disgust, 4:Fear, 5:Happy, 6:Sad, 7:Surprise}
CKPLUS_EMOTION_MAP = {
    5: 0,  # Happy    → Engaged
    0: 1,  # Neutral  → Boredom
    2: 1,  # Contempt → Boredom (proxy: CK+ has no neutral in Kaggle version)
    6: 2,  # Sad      → Frustration
    1: 2,  # Anger    → Frustration
    7: 3,  # Surprise → Confusion
    4: 3,  # Fear     → Confusion
    # 3:Disgust → excluded
}

DATASET_EMOTION_MAPS = {
    "fer2013": FER2013_EMOTION_MAP,
    "ckplus": CKPLUS_EMOTION_MAP,
}


@dataclass
class DataConfig:
    # Dataset selection
    dataset_name: str = "fer2013"  # "fer2013", "rafdb", "ckplus"

    # Paths (set by set_dataset())
    raw_dir: str = ""
    processed_dir: str = ""

    # Image
    image_size: int = 224
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: ENGAGEMENT_CLASSES)

    # Split ratios (used when dataset has no official split)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ImageNet normalization
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    def set_dataset(self, name: str):
        """Update paths and settings for the given dataset."""
        self.dataset_name = name
        self.raw_dir = os.path.join(PROJECT_ROOT, "data", name)
        self.processed_dir = os.path.join(PROJECT_ROOT, "data", "processed", name)


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
    unfreeze_blocks: int = 3

    # Optimizer
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Focal weighted loss
    focal_gamma: float = 1.0
    label_smoothing: float = 0.1

    # Paths
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "experiments", "checkpoints")
    results_dir: str = os.path.join(PROJECT_ROOT, "experiments", "results", "default")


@dataclass
class ModelConfig:
    # CSAM
    csam_reduction: int = 4
    csam_kernel: int = 7

    # Classification head
    head_hidden: int = 256
    head_dropout: float = 0.5

    # Backbone choices
    backbone: str = "mobilenetv3_small"
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
        "crop_scale": (0.9, 1.0),  # Mild crop for small 48x48 images
        "hue": 0.1,
    },
}
