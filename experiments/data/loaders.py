"""
Unified dataset loaders for FER2013 and CK+.

Both datasets are in folder-per-emotion format:
  FER2013: data/fer2013/{train,test}/{angry,disgust,fear,happy,neutral,sad,surprise}/*.jpg
  CK+:     data/ckplus/CK+48/{anger,contempt,disgust,fear,happy,sadness,surprise}/*.png

Label mapping (emotion → engagement):
  Happy     → 0: Engaged
  Neutral   → 1: Boredom
  Sad/Anger → 2: Frustration
  Surprise/Fear → 3: Confusion
  Disgust/Contempt → excluded
"""

import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Folder name → engagement label maps ──────────────────────────────────────

FER2013_FOLDER_MAP = {
    "happy": 0,     # Engaged
    "neutral": 1,   # Boredom
    "sad": 2,       # Frustration
    "angry": 2,     # Frustration
    "surprise": 3,  # Confusion
    "fear": 3,      # Confusion
    # "disgust" → excluded
}

CKPLUS_FOLDER_MAP = {
    "happy": 0,      # Engaged
    "happiness": 0,  # Engaged
    "contempt": 1,   # Boredom (proxy: CK+ has no neutral, contempt is low-arousal negative)
    "neutral": 1,    # Boredom
    "sadness": 2,    # Frustration
    "sad": 2,        # Frustration
    "anger": 2,      # Frustration
    "angry": 2,      # Frustration
    "surprise": 3,   # Confusion
    "fear": 3,       # Confusion
    # "disgust" → excluded
}


def _scan_emotion_folder(folder_path: str, folder_map: dict) -> List[Tuple[str, int]]:
    """Scan a folder-per-emotion directory, return [(image_path, engagement_label), ...]."""
    items = []
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_dir):
            continue

        label_key = folder_name.lower().strip()
        if label_key not in folder_map:
            continue  # skip excluded emotions

        engagement = folder_map[label_key]

        for fname in sorted(os.listdir(folder_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_path = os.path.join(folder_dir, fname)
                items.append((img_path, engagement))

    return items


# ── FER2013 ──────────────────────────────────────────────────────────────────

def load_fer2013(raw_dir: str) -> dict:
    """
    Load FER2013 from folder-per-emotion structure.

    Expected:
        raw_dir/train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.jpg
        raw_dir/test/{...}/*.jpg
    """
    print(f"[FER2013] Loading from {raw_dir}")

    train_dir = os.path.join(raw_dir, "train")
    test_dir = os.path.join(raw_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"FER2013 train directory not found: {train_dir}")

    train_items = _scan_emotion_folder(train_dir, FER2013_FOLDER_MAP)
    test_items = _scan_emotion_folder(test_dir, FER2013_FOLDER_MAP) if os.path.isdir(test_dir) else []

    # Split train into train+val (85/15)
    np.random.seed(42)
    indices = np.random.permutation(len(train_items))
    val_size = max(1, int(len(train_items) * 0.15))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    splits = {
        "train": [train_items[i] for i in train_idx],
        "val": [train_items[i] for i in val_idx],
        "test": test_items,
    }

    for split_name, items in splits.items():
        counts = {}
        for _, label in items:
            counts[label] = counts.get(label, 0) + 1
        print(f"  {split_name}: {len(items)} images  {dict(sorted(counts.items()))}")

    return splits


# ── CK+ ──────────────────────────────────────────────────────────────────────

def load_ckplus(raw_dir: str) -> dict:
    """
    Load CK+ from folder-per-emotion structure.

    Expected:
        raw_dir/CK+48/{anger,contempt,disgust,fear,happy,sadness,surprise}/*.png
    OR:
        raw_dir/{anger,contempt,...}/*.png
    """
    print(f"[CK+] Loading from {raw_dir}")

    # Find the emotion folder
    data_dir = None
    for candidate in [
        os.path.join(raw_dir, "CK+48"),
        os.path.join(raw_dir, "ck"),
        raw_dir,
    ]:
        if os.path.isdir(candidate):
            # Check if it contains emotion subfolders
            subdirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
            if any(d.lower() in CKPLUS_FOLDER_MAP for d in subdirs):
                data_dir = candidate
                break

    if data_dir is None:
        raise FileNotFoundError(f"CK+ emotion folders not found in {raw_dir}")

    print(f"  Scanning: {data_dir}")
    all_items = _scan_emotion_folder(data_dir, CKPLUS_FOLDER_MAP)

    if len(all_items) == 0:
        raise ValueError(f"No images found in {data_dir}")

    # Subject-level split using subject ID from filename (e.g., S010_006_00000013.png)
    subject_ids = set()
    for img_path, _ in all_items:
        fname = os.path.basename(img_path)
        # Extract subject ID: S010_006_00000013 → S010
        parts = fname.split("_")
        if len(parts) >= 1 and parts[0].startswith("S"):
            subject_ids.add(parts[0])

    subject_ids = sorted(subject_ids)
    np.random.seed(42)
    np.random.shuffle(subject_ids)
    n = len(subject_ids)
    train_subj = set(subject_ids[:int(n * 0.7)])
    val_subj = set(subject_ids[int(n * 0.7):int(n * 0.85)])
    test_subj = set(subject_ids[int(n * 0.85):])

    splits = {"train": [], "val": [], "test": []}
    for img_path, label in all_items:
        fname = os.path.basename(img_path)
        parts = fname.split("_")
        subj_id = parts[0] if len(parts) >= 1 else ""

        if subj_id in train_subj:
            splits["train"].append((img_path, label))
        elif subj_id in val_subj:
            splits["val"].append((img_path, label))
        else:
            splits["test"].append((img_path, label))

    for split_name, items in splits.items():
        counts = {}
        for _, label in items:
            counts[label] = counts.get(label, 0) + 1
        print(f"  {split_name}: {len(items)} images  {dict(sorted(counts.items()))}")

    return splits


# ── Save to normalized folder structure ──────────────────────────────────────

def save_splits_to_folders(splits: dict, output_dir: str):
    """Save (image_path, label) pairs into normalized folder structure."""
    for split_name, items in splits.items():
        for idx, (src_path, label) in enumerate(items):
            out_dir = os.path.join(output_dir, split_name, str(label))
            os.makedirs(out_dir, exist_ok=True)

            img = Image.open(src_path)
            # Convert grayscale to 3-channel for pretrained models
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(os.path.join(out_dir, f"{idx:06d}.jpg"))


# ── Unified interface ────────────────────────────────────────────────────────

def load_dataset(name: str, raw_dir: str) -> dict:
    """Load any supported dataset by name."""
    loaders = {
        "fer2013": load_fer2013,
        "ckplus": load_ckplus,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")
    return loaders[name](raw_dir)


def preprocess_dataset(name: str, raw_dir: str, output_dir: str):
    """Full preprocessing: load raw → save to normalized folder structure."""
    print(f"\n{'='*60}")
    print(f"Preprocessing {name}")
    print(f"  Raw:    {raw_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    splits = load_dataset(name, raw_dir)
    save_splits_to_folders(splits, output_dir)

    total = sum(len(items) for items in splits.values())
    print(f"  Total: {total} images saved to {output_dir}")
