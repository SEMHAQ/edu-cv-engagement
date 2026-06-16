#!/usr/bin/env python3
"""
Video-level DAiSEE engagement detection using TemporalAggregator (LSTM).
1. Extract frame features from trained MobileNetV3+CSAM
2. Group by video clip
3. Train LSTM aggregator
4. Evaluate video-level accuracy
"""
import os, sys, json, csv, torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

sys.path.insert(0, os.path.dirname(__file__))
from models.mobilenetv3_csam import MobileNetV3CSAM
from models.temporal import TemporalAggregator, VideoDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'daisee')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'daisee_temporal')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

BATCH_SIZE = 32
MAX_FRAMES = 16
HIDDEN_DIM = 128
NUM_EPOCHS = 30
LR = 1e-3


def extract_features(split, model):
    """Extract frame features from MobileNetV3+CSAM."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    features = []
    labels = []
    filenames = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, lbls) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            # Extract features from backbone (before classifier)
            # Extract features from features layers
            x = imgs
            for block, csam in zip(model.features, model.csam_modules):
                x = block(x)
                x = csam(x)
            feats = x
            if feats.dim() == 4:
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
            features.append(feats.cpu())
            # Map to binary: 0,1 -> 0, 2,3 -> 1
            binary_labels = [1 if l >= 2 else 0 for l in lbls.tolist()]
            labels.extend(binary_labels)
            
            # Get filenames for grouping
            start_idx = batch_idx * loader.batch_size
            for i in range(len(imgs)):
                idx = start_idx + i
                if idx < len(dataset.samples):
                    path = dataset.samples[idx][0]
                    # Extract video ID from filename: e.g., "1100011002_0003.jpg" -> "1100011002"
                    fname = os.path.basename(path)
                    vid = fname.rsplit('_', 1)[0] if '_' in fname else fname.rsplit('.', 1)[0]
                    filenames.append(vid)
    
    features = torch.cat(features, dim=0)
    return features, labels, filenames


def train_temporal(train_features, train_labels, train_vids, 
                   val_features, val_labels, val_vids, num_classes):
    """Train the temporal aggregator."""
    train_ds = VideoDataset(train_features, train_labels, train_vids, 
                           max_frames=MAX_FRAMES, num_classes=num_classes)
    val_ds = VideoDataset(val_features, val_labels, val_vids,
                         max_frames=MAX_FRAMES, num_classes=num_classes)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = train_features[0].shape[0]
    model = TemporalAggregator(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM,
        num_layers=2, num_classes=num_classes, dropout=0.3
    ).to(DEVICE)
    
    # Class weights for imbalanced data
    label_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    weights = torch.tensor(len(train_labels) / (num_classes * label_counts + 1e-8), 
                          dtype=torch.float32).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_val_f1 = 0
    history = []
    
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for feats, lbls, masks in train_loader:
            feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(feats)
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(lbls)
            train_correct += (logits.argmax(1) == lbls).sum().item()
            train_total += len(lbls)
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for feats, lbls, masks in val_loader:
                feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
                logits, _ = model(feats)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * len(lbls)
                val_correct += (logits.argmax(1) == lbls).sum().item()
                val_total += len(lbls)
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_true.extend(lbls.cpu().tolist())
        
        # Compute F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
        
        history.append({
            'epoch': epoch,
            'phase': 'val',
            'accuracy': val_correct / val_total,
            'f1_macro': val_f1,
            'loss': val_loss / val_total,
        })
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Acc: {train_correct/train_total:.3f} | "
              f"Val Acc: {val_correct/val_total:.3f} | Val F1: {val_f1:.3f}",
              flush=True)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'temporal_best.pth'))
    
    return model, history


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    # Load trained frame-level model
    print("Loading MobileNetV3+CSAM model...", flush=True)
    frame_model = MobileNetV3CSAM(num_classes=2, pretrained=False).to(DEVICE)
    
    # Load best checkpoint
    ckpt_path = os.path.join(CKPT_DIR, 'fold_0_best.pth')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        if 'model_state_dict' in state:
            frame_model.load_state_dict(state['model_state_dict'], strict=False)
        else:
            frame_model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}", flush=True)
    else:
        print("WARNING: No checkpoint found, using random weights", flush=True)
    
    # Extract features
    print("\nExtracting train features...", flush=True)
    train_feats, train_labels, train_vids = extract_features('train', frame_model)
    print(f"Train: {len(train_feats)} frames, {len(set(train_vids))} videos", flush=True)
    
    print("Extracting validation features...", flush=True)
    val_feats, val_labels, val_vids = extract_features('validation', frame_model)
    print(f"Val: {len(val_feats)} frames, {len(set(val_vids))} videos", flush=True)
    
    print("Extracting test features...", flush=True)
    test_feats, test_labels, test_vids = extract_features('test', frame_model)
    print(f"Test: {len(test_feats)} frames, {len(set(test_vids))} videos", flush=True)
    
    # Train temporal aggregator
    print("\nTraining temporal aggregator...", flush=True)
    temporal_model, history = train_temporal(
        train_feats, train_labels, train_vids,
        val_feats, val_labels, val_vids,
        num_classes=2
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...", flush=True)
    test_ds = VideoDataset(test_feats, test_labels, test_vids,
                          max_frames=MAX_FRAMES, num_classes=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    temporal_model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for feats, lbls, masks in test_loader:
            feats = feats.to(DEVICE)
            logits, _ = temporal_model(feats)
            test_preds.extend(logits.argmax(1).cpu().tolist())
            test_true.extend(lbls.tolist())
    
    from sklearn.metrics import f1_score, accuracy_score
    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
    
    print(f"\n=== Video-Level Results (DAiSEE) ===", flush=True)
    print(f"Test Accuracy: {test_acc:.1%}", flush=True)
    print(f"Test F1 (macro): {test_f1:.1%}", flush=True)
    
    # Save results
    results = {
        'model': 'MobileNetV3+CSAM + TemporalAggregator (BiLSTM)',
        'dataset': 'DAiSEE',
        'task': 'binary_video_level',
        'test_accuracy': test_acc,
        'test_f1_macro': test_f1,
        'test_preds': test_preds,
        'test_true': test_true,
        'history': history,
    }
    
    with open(os.path.join(RESULTS_DIR, 'temporal_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}/temporal_results.json", flush=True)


if __name__ == '__main__':
    main()
