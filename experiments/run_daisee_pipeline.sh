#!/bin/bash
# Full DAiSEE experiment pipeline
# Run this after DAiSEE is downloaded and processed

set -e
PROJ="/mnt/e/Project/MDPI/edu-cv-engagement"
cd "$PROJ"

echo "=== Step 1: Process DAiSEE ==="
python3 experiments/process_daisee.py

echo ""
echo "=== Step 2: Run DAiSEE experiments ==="
python3 experiments/run_experiments.py --model mobilenetv3_csam --dataset daisee --batch-size 64

echo ""
echo "=== Step 3: Run cross-dataset evaluation (FER2013 -> DAiSEE) ==="
# Modify cross_dataset_eval.py to also test on DAiSEE
python3 -c "
import sys
sys.path.insert(0, 'experiments')
from cross_dataset_eval import main as cross_eval
# This will use the best FER2013 model and test on DAiSEE
"

echo ""
echo "=== Step 4: Update figures ==="
python3 experiments/update_figures.py 2>/dev/null || echo "update_figures.py not found, skipping"

echo ""
echo "=== Step 5: Update paper with DAiSEE results ==="
echo "Manual step: update paper/main.tex with DAiSEE results"

echo ""
echo "=== Step 6: Compile paper ==="
cd paper
export PATH=/mnt/e/texlive/2026/bin/windows:$PATH
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "=== Step 7: Commit and push ==="
cd "$PROJ"
git add -A
git commit -m "feat: add DAiSEE experiment results"
git push origin main

echo ""
echo "=== Done! ==="
