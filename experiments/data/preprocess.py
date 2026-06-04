#!/usr/bin/env python3
"""
Preprocess datasets into normalized folder structure.

Usage:
    python3 data/preprocess.py                    # Process all available datasets
    python3 data/preprocess.py --dataset fer2013  # Process specific dataset

Expected raw data locations:
    data/fer2013/fer2013.csv
    data/rafdb/basic/EmoLabel/list_patition_label.txt + basic/aligned/*.jpg
    data/ckplus/Emotion/ + cohn-kanade-images/  (or Kaggle folder format)

Output:
    data/processed/{dataset}/{train,val,test}/{0,1,2,3}/*.jpg
    where 0=Engaged, 1=Boredom, 2=Frustration, 3=Confusion
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_ROOT
from data.loaders import preprocess_dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess facial expression datasets")
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["fer2013", "rafdb", "ckplus"],
        help="Dataset to preprocess (default: all available)",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["fer2013", "rafdb", "ckplus"]

    for name in datasets:
        raw_dir = os.path.join(PROJECT_ROOT, "data", name)
        output_dir = os.path.join(PROJECT_ROOT, "data", "processed", name)

        if not os.path.isdir(raw_dir):
            print(f"\n[SKIP] {name}: raw directory not found at {raw_dir}")
            continue

        if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"\n[SKIP] {name}: already preprocessed at {output_dir}")
            continue

        try:
            preprocess_dataset(name, raw_dir, output_dir)
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")


if __name__ == "__main__":
    main()
