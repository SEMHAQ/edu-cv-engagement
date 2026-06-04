#!/usr/bin/env python3
"""Minimal FER2013 training test - no focal loss, no tricks."""
import os, sys, torch, torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mobilenetv3_csam import MobileNetV3CSAM

DATA_DIR = "/mnt/e/Project/MDPI/edu-cv-engagement/data/processed/fer2013"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple transforms
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Datasets
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)

print(f"Train: {len(train_ds)} samples, classes: {train_ds.classes}")
print(f"Val: {len(val_ds)} samples, classes: {val_ds.classes}")

# Class weights for loss
targets = [s[1] for s in train_ds.samples]
class_counts = np.bincount(targets, minlength=4)
class_weights = torch.FloatTensor(len(targets) / (len(class_counts) * class_counts)).to(DEVICE)
print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights}")

# Weighted sampler
sample_weights = [class_weights[t].item() for t in targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

# Model
model = MobileNetV3CSAM(num_classes=4, pretrained=True).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# Simple CrossEntropyLoss with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Train 20 epochs
for epoch in range(20):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    scheduler.step()
    
    train_acc = correct / total
    
    # Evaluate
    model.eval()
    val_correct, val_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_acc = val_correct / val_total
    pred_dist = np.bincount(all_preds, minlength=4)
    
    print(f"Epoch {epoch+1:2d}/20 | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Preds: {pred_dist}")
    
    # Early check - if all classes predicted, good sign
    if np.sum(pred_dist > 0) == 4:
        print(f"  -> All 4 classes predicted!")

print("\nDone!")
