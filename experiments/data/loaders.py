"""
Unified dataset loaders for FER2013, RAF-DB, and CK+.

Each loader:
1. Reads the raw dataset from data/{name}/
2. Applies emotion → engagement label mapping
3. Returns lists of (image_path_or_array, engagement_label)
4. The preprocess.py script uses these to create the normalized folder structure

Label mapping (emotion → engagement):
  Happy     → 0: Engaged
  Neutral   → 1: Boredom
  Sad/Anger → 2: Frustration
  Surprise/Fear → 3: Confusion
"""

import os
import glob
import csv
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FER2013_EMOTION_MAP, RAFDB_EMOTION_MAP, CKPLUS_EMOTION_MAP


# ── FER2013 ──────────────────────────────────────────────────────────────────

def load_fer2013(raw_dir: str) -> dict:
    """
    Load FER2013 from CSV file.

    Expected structure:
        raw_dir/
            fer2013.csv  (or icml_face_data.csv)

    Returns: {split: [(pixel_array_48x48, engagement_label), ...]}
    """
    # Find the CSV file
    csv_path = None
    for name in ["fer2013.csv", "icml_face_data.csv", "fer2013.zip"]:
        p = os.path.join(raw_dir, name)
        if os.path.isfile(p):
            csv_path = p
            break

    if csv_path is None:
        # Check for extracted CSV in subdirectory
        for root, dirs, files in os.walk(raw_dir):
            for f in files:
                if f.endswith(".csv") and "fer" in f.lower():
                    csv_path = os.path.join(root, f)
                    break
            if csv_path:
                break

    if csv_path is None:
        raise FileNotFoundError(
            f"FER2013 CSV not found in {raw_dir}. "
            f"Download from: https://www.kaggle.com/datasets/msambare/fer2013"
        )

    print(f"[FER2013] Loading from {csv_path}")
    df = pd.read_csv(csv_path)

    # Handle different CSV formats
    if "pixels" in df.columns:
        pixel_col = "pixels"
        emotion_col = "emotion"
        usage_col = "Usage"
    elif " Usage" in df.columns:
        pixel_col = " pixels"
        emotion_col = "emotion"
        usage_col = " Usage"
    else:
        # Try to infer
        cols = df.columns.tolist()
        pixel_col = [c for c in cols if "pixel" in c.lower()][0]
        emotion_col = [c for c in cols if "emotion" in c.lower()][0]
        usage_col = [c for c in cols if "usage" in c.lower()][0]

    # Filter out excluded emotions (Disgust=1)
    exclude_emotions = {k for k, v in FER2013_EMOTION_MAP.items() if k not in FER2013_EMOTION_MAP}
    # Actually, just filter to only mapped emotions
    df = df[df[emotion_col].isin(FER2013_EMOTION_MAP.keys())]

    # Map to engagement labels
    df["engagement"] = df[emotion_col].map(FER2013_EMOTION_MAP)

    splits = {}
    for split_name, usage_val in [("train", "Training"), ("val", "PublicTest"), ("test", "PrivateTest")]:
        split_df = df[df[usage_col].str.strip() == usage_val]
        items = []
        for _, row in split_df.iterrows():
            pixels = np.array(row[pixel_col].split(), dtype=np.uint8).reshape(48, 48)
            label = int(row["engagement"])
            items.append((pixels, label))
        splits[split_name] = items
        print(f"  {split_name}: {len(items)} images")

    return splits


def save_fer2013_images(splits: dict, output_dir: str):
    """Save FER2013 pixel arrays as JPEG images in normalized folder structure."""
    for split_name, items in splits.items():
        for idx, (pixels, label) in enumerate(items):
            out_dir = os.path.join(output_dir, split_name, str(label))
            os.makedirs(out_dir, exist_ok=True)
            img = Image.fromarray(pixels, mode="L")
            img.save(os.path.join(out_dir, f"{idx:06d}.jpg"))


# ── RAF-DB ───────────────────────────────────────────────────────────────────

def load_rafdb(raw_dir: str) -> dict:
    """
    Load RAF-DB from aligned images + label file.

    Expected structure:
        raw_dir/
            basic/
                EmoLabel/
                    list_patition_label.txt
                aligned/
                    train_00001_aligned.jpg
                    test_00001_aligned.jpg
                    ...
    OR:
        raw_dir/
            EmoLabel/
                list_patition_label.txt
            aligned/
                ...

    Returns: {split: [(image_path, engagement_label), ...]}
    """
    # Find label file and image directory
    label_path = None
    img_dir = None

    for candidate in [
        os.path.join(raw_dir, "basic", "EmoLabel", "list_patition_label.txt"),
        os.path.join(raw_dir, "EmoLabel", "list_patition_label.txt"),
        os.path.join(raw_dir, "list_patition_label.txt"),
    ]:
        if os.path.isfile(candidate):
            label_path = candidate
            break

    if label_path is None:
        raise FileNotFoundError(
            f"RAF-DB label file not found in {raw_dir}. "
            f"Download from: http://www.whdeng.cn/RAF/model.html"
        )

    # Find aligned image directory
    for candidate in [
        os.path.join(raw_dir, "basic", "aligned"),
        os.path.join(raw_dir, "aligned"),
        os.path.join(raw_dir, "basic", "Image", "aligned"),
    ]:
        if os.path.isdir(candidate):
            img_dir = candidate
            break

    if img_dir is None:
        raise FileNotFoundError(f"RAF-DB aligned image directory not found in {raw_dir}")

    print(f"[RAF-DB] Loading labels from {label_path}")
    print(f"[RAF-DB] Images from {img_dir}")

    # Parse label file
    labels_df = pd.read_csv(label_path, sep=r"\s+", header=None, names=["filename", "emotion_raw"])

    splits = {"train": [], "val": [], "test": []}

    for _, row in labels_df.iterrows():
        emo = int(row["emotion_raw"])
        if emo not in RAFDB_EMOTION_MAP:
            continue  # skip excluded emotions (Disgust)

        engagement = RAFDB_EMOTION_MAP[emo]

        # filename in label: train_00001.jpg → actual file: train_00001_aligned.jpg
        fname = row["filename"]
        img_name = fname.replace(".jpg", "_aligned.jpg")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.isfile(img_path):
            # Try without _aligned suffix
            img_path = os.path.join(img_dir, fname)
            if not os.path.isfile(img_path):
                continue

        # Determine split from filename prefix
        if fname.startswith("train"):
            splits["train"].append((img_path, engagement))
        elif fname.startswith("test"):
            splits["test"].append((img_path, engagement))

    # Split train into train+val (80/20)
    train_items = splits["train"]
    np.random.seed(42)
    indices = np.random.permutation(len(train_items))
    val_size = int(len(train_items) * 0.2)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    splits["val"] = [train_items[i] for i in val_idx]
    splits["train"] = [train_items[i] for i in train_idx]

    for split_name, items in splits.items():
        print(f"  {split_name}: {len(items)} images")

    return splits


def save_rafdb_images(splits: dict, output_dir: str):
    """Copy RAF-DB images into normalized folder structure."""
    for split_name, items in splits.items():
        for idx, (src_path, label) in enumerate(items):
            out_dir = os.path.join(output_dir, split_name, str(label))
            os.makedirs(out_dir, exist_ok=True)
            # Copy or convert image
            img = Image.open(src_path).convert("RGB")
            img.save(os.path.join(out_dir, f"{idx:06d}.jpg"))


# ── CK+ ──────────────────────────────────────────────────────────────────────

def load_ckplus(raw_dir: str) -> dict:
    """
    Load CK+ from raw CMU format or Kaggle pre-processed format.

    Raw format:
        raw_dir/
            cohn-kanade-images/{Subject}/{Sequence}/*.png
            Emotion/{Subject}/{Sequence}/*.txt

    Kaggle format:
        raw_dir/
            {EmotionName}/*.png  (e.g., Angry/, Happy/, etc.)

    Returns: {split: [(image_path_or_array, engagement_label), ...]}
    """
    # Detect format
    emotion_dir = os.path.join(raw_dir, "Emotion")
    image_dir = os.path.join(raw_dir, "cohn-kanade-images")

    if os.path.isdir(emotion_dir) and os.path.isdir(image_dir):
        return _load_ckplus_raw(raw_dir, emotion_dir, image_dir)
    else:
        return _load_ckplus_kaggle(raw_dir)


def _load_ckplus_raw(raw_dir, emotion_dir, image_dir):
    """Load CK+ from raw CMU format."""
    print(f"[CK+] Loading raw format from {raw_dir}")

    # CK+ emotion names (0-indexed)
    ck_emotions = {
        0: "Neutral", 1: "Anger", 2: "Contempt", 3: "Disgust",
        4: "Fear", 5: "Happy", 6: "Sadness", 7: "Surprise"
    }

    all_items = []

    for subject in sorted(os.listdir(emotion_dir)):
        subj_emo_dir = os.path.join(emotion_dir, subject)
        if not os.path.isdir(subj_emo_dir):
            continue

        for seq in sorted(os.listdir(subj_emo_dir)):
            seq_emo_dir = os.path.join(subj_emo_dir, seq)
            if not os.path.isdir(seq_emo_dir):
                continue

            # Read emotion label
            txt_files = glob.glob(os.path.join(seq_emo_dir, "*.txt"))
            if not txt_files:
                continue

            with open(txt_files[0]) as f:
                content = f.read().strip()
                if not content:
                    continue
                emo_label = int(float(content))

            # Map to engagement
            from config import CKPLUS_EMOTION_MAP
            if emo_label not in CKPLUS_EMOTION_MAP:
                continue  # skip Contempt, Disgust

            engagement = CKPLUS_EMOTION_MAP[emo_label]

            # Get peak frame (last frame in sequence)
            seq_img_dir = os.path.join(image_dir, subject, seq)
            if not os.path.isdir(seq_img_dir):
                continue

            frame_files = sorted(glob.glob(os.path.join(seq_img_dir, "*.png")))
            if not frame_files:
                continue

            peak_frame = frame_files[-1]
            all_items.append((peak_frame, engagement))

    print(f"  Total labeled frames: {len(all_items)}")

    # Subject-level split: 70% train, 15% val, 15% test
    subjects = sorted(set(os.path.basename(os.path.dirname(os.path.dirname(p))) for p, _ in all_items))
    np.random.seed(42)
    np.random.shuffle(subjects)
    n = len(subjects)
    train_subj = set(subjects[:int(n * 0.7)])
    val_subj = set(subjects[int(n * 0.7):int(n * 0.85)])
    test_subj = set(subjects[int(n * 0.85):])

    splits = {"train": [], "val": [], "test": []}

    for img_path, label in all_items:
        # Determine subject from path: .../cohn-kanade-images/S005/001/frame.png
        parts = img_path.split(os.sep)
        # Find the subject ID (S005 format)
        subj_id = None
        for p in parts:
            if p.startswith("S") and len(p) == 4 and p[1:].isdigit():
                subj_id = p
                break
        if subj_id is None:
            continue

        if subj_id in train_subj:
            splits["train"].append((img_path, label))
        elif subj_id in val_subj:
            splits["val"].append((img_path, label))
        else:
            splits["test"].append((img_path, label))

    for split_name, items in splits.items():
        print(f"  {split_name}: {len(items)} images")

    return splits


def _load_ckplus_kaggle(raw_dir):
    """Load CK+ from Kaggle pre-processed format (folder per emotion)."""
    print(f"[CK+] Loading Kaggle format from {raw_dir}")

    # Map folder names to original CK+ labels
    folder_to_emo = {
        "Anger": 1, "Angry": 1,
        "Contempt": 2,
        "Disgust": 3,
        "Fear": 4,
        "Happy": 5, "Happiness": 5,
        "Sadness": 6, "Sad": 6,
        "Surprise": 7,
        "Neutral": 0,
    }

    from config import CKPLUS_EMOTION_MAP
    all_items = []

    for folder_name in os.listdir(raw_dir):
        folder_path = os.path.join(raw_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Match folder name to emotion
        emo_key = None
        for key in folder_to_emo:
            if key.lower() == folder_name.lower():
                emo_key = folder_to_emo[key]
                break

        if emo_key is None or emo_key not in CKPLUS_EMOTION_MAP:
            continue

        engagement = CKPLUS_EMOTION_MAP[emo_key]

        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder_path, fname)
                all_items.append((img_path, engagement))

    print(f"  Total images: {len(all_items)}")

    # Subject-level split if filenames contain subject IDs, else random
    np.random.seed(42)
    indices = np.random.permutation(len(all_items))
    n = len(all_items)
    train_idx = indices[:int(n * 0.7)]
    val_idx = indices[int(n * 0.7):int(n * 0.85)]
    test_idx = indices[int(n * 0.85):]

    splits = {
        "train": [all_items[i] for i in train_idx],
        "val": [all_items[i] for i in val_idx],
        "test": [all_items[i] for i in test_idx],
    }

    for split_name, items in splits.items():
        print(f"  {split_name}: {len(items)} images")

    return splits


def save_ckplus_images(splits: dict, output_dir: str):
    """Save CK+ images into normalized folder structure."""
    for split_name, items in splits.items():
        for idx, (src_path, label) in enumerate(items):
            out_dir = os.path.join(output_dir, split_name, str(label))
            os.makedirs(out_dir, exist_ok=True)
            img = Image.open(src_path)
            # Convert grayscale to RGB for consistency with pretrained models
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(os.path.join(out_dir, f"{idx:06d}.jpg"))


# ── Unified interface ────────────────────────────────────────────────────────

def load_dataset(name: str, raw_dir: str) -> dict:
    """
    Load any of the 3 datasets by name.

    Args:
        name: "fer2013", "rafdb", or "ckplus"
        raw_dir: path to raw dataset directory

    Returns: {split: [(image_or_path, engagement_label), ...]}
    """
    loaders = {
        "fer2013": load_fer2013,
        "rafdb": load_rafdb,
        "ckplus": load_ckplus,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")
    return loaders[name](raw_dir)


def preprocess_dataset(name: str, raw_dir: str, output_dir: str):
    """
    Full preprocessing pipeline: load raw dataset, save to normalized folder structure.
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing {name}")
    print(f"  Raw:      {raw_dir}")
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}")

    splits = load_dataset(name, raw_dir)

    savers = {
        "fer2013": save_fer2013_images,
        "rafdb": save_rafdb_images,
        "ckplus": save_ckplus_images,
    }

    savers[name](splits, output_dir)

    # Print summary
    total = 0
    for split_name, items in splits.items():
        count = len(items)
        total += count
        print(f"  {split_name}: {count} images")
    print(f"  Total: {total} images")
    print(f"  Saved to: {output_dir}")
