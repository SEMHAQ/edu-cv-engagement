#!/usr/bin/env python3
"""Cross-dataset evaluation: train on FER2013, test on CK+ without fine-tuning."""
import os, sys, json, torch, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mobilenetv3_csam import MobileNetV3CSAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FER2013 emotion -> engagement mapping
FER_MAP = {3: 0, 6: 1, 4: 2, 0: 2, 5: 3, 2: 3}  # happy=0, neutral=1, sad+angry=2, surprise+fear+disgust=3

# CK+ emotion -> engagement mapping (folder names)
CK_MAP = {
    "happy": 0, "happiness": 0,
    "neutral": 1, "contempt": 1,
    "sadness": 2, "sad": 2, "anger": 2, "angry": 2,
    "surprise": 3, "fear": 3, "disgust": 3,
}

def load_ckplus_test(data_dir, transform):
    """Load CK+ test set with engagement mapping."""
    test_dir = os.path.join(data_dir, "test")
    images, labels = [], []
    for cls_name in os.listdir(test_dir):
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        label = CK_MAP.get(cls_name.lower())
        if label is None:
            print(f"  Skipping unknown class: {cls_name}")
            continue
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            images.append(img_path)
            labels.append(label)
    return images, labels

class CrossDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

def evaluate(model, loader, num_classes=4):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    report = classification_report(all_labels, all_preds, target_names=["Engaged","Boredom","Frustration","Confusion"], zero_division=0)
    return acc, f1, f1_per, cm, report

def main():
    # Find best model checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    best_ckpt = None
    best_acc = 0
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pth"):
            path = os.path.join(ckpt_dir, f)
            state = torch.load(path, map_location="cpu", weights_only=True)
            # Try to extract accuracy from filename or state
            if isinstance(state, dict) and "accuracy" in state:
                if state["accuracy"] > best_acc:
                    best_acc = state["accuracy"]
                    best_ckpt = path
            else:
                best_ckpt = path  # Use latest if no accuracy info
    
    if best_ckpt is None:
        print("No checkpoint found!")
        return
    
    print(f"Loading model from: {best_ckpt}")
    model = MobileNetV3CSAM(num_classes=4, pretrained=False)
    state = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model = model.to(DEVICE)
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load CK+ test set
    ckplus_dir = "/mnt/e/Project/MDPI/edu-cv-engagement/data/processed/ckplus"
    images, labels = load_ckplus_test(ckplus_dir, transform)
    print(f"CK+ test set: {len(images)} images")
    print(f"  Class distribution: {np.bincount(labels, minlength=4)}")
    
    dataset = CrossDataset(images, labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Evaluate
    acc, f1, f1_per, cm, report = evaluate(model, loader)
    print(f"\nCross-dataset evaluation (FER2013 -> CK+):")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1 (macro): {f1:.3f}")
    print(f"  F1 per class: {[f'{x:.3f}' for x in f1_per]}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Save results
    results = {
        "experiment": "cross_dataset_fer2013_to_ckplus",
        "source_dataset": "FER2013",
        "target_dataset": "CK+",
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "f1_per_class": [float(x) for x in f1_per],
        "confusion_matrix": cm.tolist(),
        "checkpoint": best_ckpt,
        "num_test_samples": len(images),
    }
    os.makedirs("experiments/results/cross_dataset", exist_ok=True)
    with open("experiments/results/cross_dataset/fer2013_to_ckplus.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to experiments/results/cross_dataset/fer2013_to_ckplus.json")

if __name__ == "__main__":
    main()
