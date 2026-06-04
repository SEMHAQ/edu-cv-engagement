#!/usr/bin/env python3
"""
DAiSEE Dataset Processing Pipeline
-----------------------------------
DAiSEE is a video-based engagement detection dataset.
This script:
1. Extracts frames from DAiSEE videos (1 frame per second)
2. Maps DAiSEE labels (Engagement, Boredom, Confusion, Frustration) to our 4-class scheme
3. Splits into train/val/test
4. Saves processed images in the same format as FER2013

DAiSEE label mapping (from the dataset's README):
- Engagement: 0 (very low), 1 (low), 2 (high), 3 (very high)
- Boredom: 0-3
- Confusion: 0-3  
- Frustration: 0-3

We map to 4 engagement classes based on the dominant emotion label.
"""
import os
import sys
import json
import shutil
import random
import numpy as np
from pathlib import Path
from collections import Counter

# Configuration
RAW_DIR = "/mnt/e/Project/MDPI/edu-cv-engagement/data/raw/daisee"
PROCESSED_DIR = "/mnt/e/Project/MDPI/edu-cv-engagement/data/processed/daisee"
FRAME_RATE = 1  # Extract 1 frame per second
IMG_SIZE = 224   # Resize to 224x224

def find_daisee_root():
    """Find the DAiSEE dataset root directory."""
    # Check common locations
    candidates = [
        RAW_DIR,
        "/mnt/e/360Downloads/DAiSEE",
        "/mnt/e/360Downloads/daisee",
        "/mnt/e/360Downloads/DAISEE",
    ]
    
    for c in candidates:
        if os.path.exists(c):
            # Check if it has the expected structure
            if os.path.exists(os.path.join(c, "Labels")) or os.path.exists(os.path.join(c, "Videos")):
                return c
            # Check subdirectories
            for sub in os.listdir(c):
                sub_path = os.path.join(c, sub)
                if os.path.isdir(sub_path):
                    if os.path.exists(os.path.join(sub_path, "Labels")) or os.path.exists(os.path.join(sub_path, "Videos")):
                        return sub_path
    
    # Search in 360Downloads
    downloads = "/mnt/e/360Downloads"
    if os.path.exists(downloads):
        for f in os.listdir(downloads):
            if "daisee" in f.lower() or "DAiSEE" in f:
                full_path = os.path.join(downloads, f)
                if os.path.isdir(full_path):
                    return full_path
                elif f.endswith('.zip'):
                    print(f"Found zip: {full_path}")
                    print("Please extract manually or wait for auto-extraction")
                    return None
    
    return None

def load_labels(labels_dir):
    """Load DAiSEE labels from CSV files."""
    labels = {}
    
    for split in ["Train", "Validation", "Test"]:
        label_file = os.path.join(labels_dir, f"{split}_Labels.csv")
        if not os.path.exists(label_file):
            # Try alternative names
            for alt in [f"{split.lower()}_labels.csv", f"{split}.csv"]:
                alt_path = os.path.join(labels_dir, alt)
                if os.path.exists(alt_path):
                    label_file = alt_path
                    break
        
        if not os.path.exists(label_file):
            print(f"  Warning: {label_file} not found")
            continue
        
        print(f"  Loading {label_file}...")
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 5:
                video_id = parts[0]
                engagement = int(parts[1])
                boredom = int(parts[2])
                confusion = int(parts[3])
                frustration = int(parts[4])
                labels[video_id] = {
                    'engagement': engagement,
                    'boredom': boredom,
                    'confusion': confusion,
                    'frustration': frustration,
                    'split': split
                }
    
    return labels

def map_to_engagement_class(label):
    """
    Map DAiSEE multi-dimensional labels to 4 engagement classes.
    
    DAiSEE has 4 dimensions (engagement, boredom, confusion, frustration) each 0-3.
    We use the dominant dimension to determine the class:
    - Class 0 (Engaged): High engagement (>=2), low boredom (<2)
    - Class 1 (Boredom): High boredom (>=2), low engagement (<2)
    - Class 2 (Frustration): High frustration (>=2)
    - Class 3 (Confusion): High confusion (>=2)
    
    If multiple are high, use the highest score.
    """
    engagement = label['engagement']
    boredom = label['boredom']
    confusion = label['confusion']
    frustration = label['frustration']
    
    # Simple mapping based on dominant emotion
    scores = {
        0: engagement,   # Engaged
        1: boredom,      # Boredom
        2: frustration,  # Frustration
        3: confusion     # Confusion
    }
    
    # If engagement is high and boredom is low, classify as Engaged
    if engagement >= 2 and boredom < 2:
        return 0
    
    # Find the dominant non-engagement emotion
    emotions = {1: boredom, 2: frustration, 3: confusion}
    dominant_emotion = max(emotions, key=emotions.get)
    dominant_score = emotions[dominant_emotion]
    
    # If dominant emotion is strong enough, use it
    if dominant_score >= 2:
        return dominant_emotion
    
    # Default to engagement level
    if engagement >= 2:
        return 0
    else:
        return 1  # Low engagement = Boredom

def extract_frames(video_path, output_dir, frame_rate=FRAME_RATE):
    """Extract frames from a video file."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frame_rate) if fps > 0 else 30
        
        count = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Resize
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                # Save
                output_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
                cv2.imwrite(output_path, frame)
                count += 1
            
            frame_idx += 1
        
        cap.release()
        return count
    except Exception as e:
        print(f"  Error extracting {video_path}: {e}")
        return 0

def process_daisee(root_dir):
    """Process the entire DAiSEE dataset."""
    print(f"Processing DAiSEE from: {root_dir}")
    
    # Find labels
    labels_dir = os.path.join(root_dir, "Labels")
    if not os.path.exists(labels_dir):
        # Try alternative
        for d in os.listdir(root_dir):
            if "label" in d.lower():
                labels_dir = os.path.join(root_dir, d)
                break
    
    labels = load_labels(labels_dir)
    print(f"  Loaded {len(labels)} video labels")
    
    if len(labels) == 0:
        print("  ERROR: No labels found!")
        return
    
    # Find videos directory
    videos_dir = os.path.join(root_dir, "Videos")
    if not os.path.exists(videos_dir):
        for d in os.listdir(root_dir):
            if "video" in d.lower():
                videos_dir = os.path.join(root_dir, d)
                break
    
    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in range(4):
            os.makedirs(os.path.join(PROCESSED_DIR, split, str(cls)), exist_ok=True)
    
    # Process each video
    stats = Counter()
    for video_id, label in labels.items():
        # Map to engagement class
        cls = map_to_engagement_class(label)
        
        # Determine split
        split = label['split'].lower()
        if split == "validation":
            split = "val"
        
        # Find video file
        video_path = None
        for ext in [".avi", ".mp4"]:
            candidate = os.path.join(videos_dir, video_id + ext)
            if os.path.exists(candidate):
                video_path = candidate
                break
        
        if video_path is None:
            # Try searching
            for root, dirs, files in os.walk(videos_dir):
                for f in files:
                    if video_id in f and (f.endswith('.avi') or f.endswith('.mp4')):
                        video_path = os.path.join(root, f)
                        break
                if video_path:
                    break
        
        if video_path is None:
            continue
        
        # Extract frames
        output_dir = os.path.join(PROCESSED_DIR, split, str(cls))
        n_frames = extract_frames(video_path, output_dir)
        stats[(split, cls)] += n_frames
    
    # Print statistics
    print("\n=== DAiSEE Processing Complete ===")
    print(f"Total videos processed: {sum(stats.values())}")
    for split in ["train", "val", "test"]:
        for cls in range(4):
            n = stats.get((split, cls), 0)
            if n > 0:
                class_names = ["Engaged", "Boredom", "Frustration", "Confusion"]
                print(f"  {split}/{class_names[cls]}: {n} frames")

if __name__ == "__main__":
    root = find_daisee_root()
    if root is None:
        print("DAiSEE dataset not found!")
        print("Please download to /mnt/e/360Downloads/ or /mnt/e/Project/MDPI/edu-cv-engagement/data/raw/daisee")
        sys.exit(1)
    
    process_daisee(root)
