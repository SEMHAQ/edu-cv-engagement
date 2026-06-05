#!/usr/bin/env python3
"""Process DAiSEE: extract frames using Engagement label directly (0-3)."""
import os, csv, sys, cv2
from collections import Counter

RAW_DIR = "/mnt/e/Project/MDPI/edu-cv-engagement/data/raw/daisee_kaggle"
PROCESSED_DIR = "/mnt/e/Project/MDPI/edu-cv-engagement/data/processed/daisee"
IMG_SIZE = 224
FRAME_INTERVAL = 30

def build_video_index(videos_dir):
    index = {}
    for subject in os.listdir(videos_dir):
        sp = os.path.join(videos_dir, subject)
        if not os.path.isdir(sp): continue
        for clip in os.listdir(sp):
            cp = os.path.join(sp, clip)
            if not os.path.isdir(cp): continue
            for f in os.listdir(cp):
                if f.endswith('.avi'):
                    index[f] = os.path.join(cp, f)
    return index

def extract_frames(video_path, output_dir, clip_prefix):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    count = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % FRAME_INTERVAL == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(output_dir, f"{clip_prefix}_{count:04d}.jpg"), frame)
            count += 1
        idx += 1
    cap.release()
    return count

def process_split(split_name, labels_file, videos_dir):
    print(f"\n=== {split_name} ===", flush=True)
    print("  Building video index...", flush=True)
    video_index = build_video_index(videos_dir)
    print(f"  Indexed {len(video_index)} videos", flush=True)
    
    labels = {}
    with open(labels_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row['ClipID'].strip()
            labels[clip_id] = row
    print(f"  Labels: {len(labels)}", flush=True)
    
    stats = Counter()
    processed = 0
    skipped = 0
    for clip_id, row in labels.items():
        # Directly use Engagement label (0-3)
        engagement = int(row.get('Engagement', row.get(' Engagement', '0')).strip())
        
        video_path = video_index.get(clip_id)
        if not video_path:
            skipped += 1
            continue
        
        output_dir = os.path.join(PROCESSED_DIR, split_name.lower(), str(engagement))
        os.makedirs(output_dir, exist_ok=True)
        clip_prefix = clip_id.replace('.avi', '')
        n = extract_frames(video_path, output_dir, clip_prefix)
        stats[engagement] += n
        processed += 1
        if processed % 50 == 0:
            print(f"  {processed}/{len(labels)} videos, {sum(stats.values())} frames", flush=True)
            sys.stdout.flush()
    
    names = ["0_VeryLow", "1_Low", "2_High", "3_VeryHigh"]
    total = sum(stats.values())
    print(f"  Done: {total} frames from {processed} videos ({skipped} skipped)", flush=True)
    for c in range(4):
        print(f"    {names[c]}: {stats.get(c,0)}", flush=True)

def main():
    for split in ["train", "test", "validation"]:
        for cls in range(4):
            os.makedirs(os.path.join(PROCESSED_DIR, split, str(cls)), exist_ok=True)
    
    splits = [
        ("Train", os.path.join(RAW_DIR, "Labels", "TrainLabels.csv"), os.path.join(RAW_DIR, "DataSet", "Train")),
        ("Test", os.path.join(RAW_DIR, "Labels", "TestLabels.csv"), os.path.join(RAW_DIR, "DataSet", "Test")),
        ("Validation", os.path.join(RAW_DIR, "Labels", "ValidationLabels.csv"), os.path.join(RAW_DIR, "DataSet", "Validation")),
    ]
    for name, lf, vd in splits:
        if os.path.exists(lf) and os.path.exists(vd):
            process_split(name, lf, vd)
    
    print("\n=== Summary ===", flush=True)
    for split in ["train", "test", "validation"]:
        total = 0
        for c in range(4):
            d = os.path.join(PROCESSED_DIR, split, str(c))
            if os.path.exists(d):
                total += len([f for f in os.listdir(d) if f.endswith('.jpg')])
        print(f"  {split}: {total} frames", flush=True)

if __name__ == "__main__":
    main()
