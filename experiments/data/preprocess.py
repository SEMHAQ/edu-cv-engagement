#!/usr/bin/env python3
"""
Preprocess DAiSEE dataset:
1. Extract frames from videos at 1 FPS
2. Detect and crop faces using MTCNN
3. Resize to 224x224, apply histogram equalization
4. Save processed face images

Expected DAiSEE directory structure:
    data/DAiSEE/
        Videos/
            Train/   (*.mp4 files)
            Test/    (*.mp4 files)
            Validation/ (*.mp4 files)
        Labels/
            TrainLabels.csv
            TestLabels.csv
            ValidationLabels.csv

CSV format: columns named like "Engagement", "Boredom", "Confusion", "Frustration"
with values 0-3, and a column with clip IDs.
"""

import os
import sys
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig


def load_labels(labels_csv: str) -> dict:
    """Load labels CSV, return dict: clip_id -> (engagement, boredom, confusion, frustration)."""
    labels = {}
    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # DAiSEE CSV has column "ClipID" or similar
            clip_id = None
            for key in row:
                if "clip" in key.lower() or "id" in key.lower():
                    clip_id = row[key].strip()
                    break
            if clip_id is None:
                # Fallback: use first column
                clip_id = list(row.values())[0].strip()

            # Extract label columns
            try:
                engagement = int(row.get("Engagement", row.get(" engagement", 0)))
                boredom = int(row.get("Boredom", row.get(" boredom", 0)))
                confusion = int(row.get("Confusion", row.get(" confusion", 0)))
                frustration = int(row.get("Frustration", row.get(" frustration", 0)))
            except (ValueError, KeyError):
                continue

            # Binarize: >=2 means positive for that state
            # Paper uses 4-class: engagement, boredom, confusion, frustration
            # Use the dominant label (argmax of the 4 scores)
            scores = [engagement, boredom, confusion, frustration]
            label = int(np.argmax(scores))
            labels[clip_id] = label

    return labels


def extract_faces_mtcnn(frame_rgb, mtcnn_detector, margin=0.20):
    """Detect face in frame using MTCNN, crop with margin, return face image or None."""
    import torch
    boxes, probs = mtcnn_detector.detect(frame_rgb)
    if boxes is None or len(boxes) == 0:
        return None

    # Take the face with highest confidence
    best_idx = np.argmax(probs)
    box = boxes[best_idx]
    x1, y1, x2, y2 = box

    # Add margin
    w = x2 - x1
    h = y2 - y1
    margin_x = w * margin
    margin_y = h * margin
    x1 = max(0, int(x1 - margin_x))
    y1 = max(0, int(y1 - margin_y))
    x2 = min(frame_rgb.shape[1], int(x2 + margin_x))
    y2 = min(frame_rgb.shape[0], int(y2 + margin_y))

    face = frame_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return face


def preprocess_dataset(cfg: DataConfig):
    """Main preprocessing pipeline."""
    from facenet_pytorch import MTCNN
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=False, device=device, select_largest=True)

    os.makedirs(cfg.processed_dir, exist_ok=True)

    splits = ["Train", "Validation", "Test"]
    total_frames = 0

    for split in splits:
        video_dir = os.path.join(cfg.raw_dir, "Videos", split)
        label_file = os.path.join(cfg.labels_dir, f"{split}Labels.csv")

        if not os.path.isdir(video_dir):
            print(f"[WARN] Video directory not found: {video_dir}, skipping {split}")
            continue
        if not os.path.isfile(label_file):
            print(f"[WARN] Label file not found: {label_file}, skipping {split}")
            continue

        labels = load_labels(label_file)
        output_dir = os.path.join(cfg.processed_dir, split.lower())
        os.makedirs(output_dir, exist_ok=True)
        for c in range(cfg.num_classes):
            os.makedirs(os.path.join(output_dir, str(c)), exist_ok=True)

        # Find all video files
        video_files = sorted(
            [f for f in os.listdir(video_dir) if f.endswith((".mp4", ".avi", ".mkv"))]
        )

        print(f"\n[{split}] Found {len(video_files)} videos, {len(labels)} labels")

        frame_count = 0
        skipped = 0

        for vf in tqdm(video_files, desc=f"Processing {split}"):
            # Extract clip ID from filename (remove extension)
            clip_id = os.path.splitext(vf)[0]

            if clip_id not in labels:
                skipped += 1
                continue

            label = labels[clip_id]
            video_path = os.path.join(video_dir, vf)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                skipped += 1
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30.0

            # Frame interval for 1 FPS extraction
            frame_interval = int(video_fps / cfg.fps)
            if frame_interval < 1:
                frame_interval = 1

            frame_idx = 0
            saved_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect and crop face
                    face = extract_faces_mtcnn(frame_rgb, mtcnn, cfg.face_margin)

                    if face is not None:
                        # Resize
                        face_resized = cv2.resize(
                            face, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR
                        )

                        # Histogram equalization on luminance channel
                        face_lab = cv2.cvtColor(face_resized, cv2.COLOR_RGB2LAB)
                        face_lab[:, :, 0] = cv2.equalizeHist(face_lab[:, :, 0])
                        face_eq = cv2.cvtColor(face_lab, cv2.COLOR_LAB2RGB)

                        # Save
                        out_path = os.path.join(
                            output_dir, str(label), f"{clip_id}_f{saved_idx:04d}.jpg"
                        )
                        cv2.imwrite(out_path, cv2.cvtColor(face_eq, cv2.COLOR_RGB2BGR))
                        frame_count += 1
                        saved_idx += 1

                frame_idx += 1

            cap.release()

        total_frames += frame_count
        print(f"[{split}] Saved {frame_count} face images, skipped {skipped} videos")

    print(f"\n[DONE] Total frames saved: {total_frames}")
    print(f"Output directory: {cfg.processed_dir}")


if __name__ == "__main__":
    cfg = DataConfig()
    preprocess_dataset(cfg)
