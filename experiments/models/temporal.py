"""
Temporal Aggregation Module: LSTM-based video-level engagement prediction.
Aggregates frame-level features from MobileNetV3+CSAM into video-level predictions.
"""
import torch
import torch.nn as nn


class TemporalAggregator(nn.Module):
    """
    Bidirectional LSTM for aggregating frame-level features into video-level predictions.
    
    Pipeline: frame features -> BiLSTM -> attention pooling -> classifier
    
    This module takes a sequence of frame-level feature vectors (extracted by 
    MobileNetV3+CSAM) and produces a single video-level prediction by:
    1. Processing the sequence through a bidirectional LSTM
    2. Using attention-weighted pooling to aggregate LSTM outputs
    3. Classifying the aggregated representation
    """
    
    def __init__(self, input_dim=576, hidden_dim=128, num_layers=2, 
                 num_classes=2, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, frame_features):
        """
        Args:
            frame_features: (batch, seq_len, input_dim) - frame-level features
            
        Returns:
            logits: (batch, num_classes) - video-level predictions
            attention_weights: (batch, seq_len) - temporal attention weights
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(frame_features)  # (batch, seq_len, hidden*2)
        
        # Attention-weighted pooling
        attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        
        # Classification
        logits = self.classifier(context)  # (batch, num_classes)
        
        return logits, attn_weights.squeeze(-1)


class VideoDataset(torch.utils.data.Dataset):
    """
    Dataset for video-level engagement prediction.
    Loads pre-extracted frame features grouped by video clip.
    """
    
    def __init__(self, frame_features, frame_labels, video_ids, 
                 max_frames=16, num_classes=2):
        """
        Args:
            frame_features: list of (feature_dim,) tensors
            frame_labels: list of int labels
            video_ids: list of video identifiers (same ID = same video)
            max_frames: max frames per video (pad/truncate)
            num_classes: number of classes
        """
        self.max_frames = max_frames
        self.num_classes = num_classes
        
        # Group frames by video
        self.videos = {}
        for feat, label, vid in zip(frame_features, frame_labels, video_ids):
            if vid not in self.videos:
                self.videos[vid] = {'features': [], 'label': label}
            self.videos[vid]['features'].append(feat)
        
        self.video_ids = list(self.videos.keys())
        
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        data = self.videos[vid]
        
        features = data['features']
        label = data['label']
        
        # Pad or truncate to max_frames
        if len(features) >= self.max_frames:
            # Uniform sampling
            indices = torch.linspace(0, len(features)-1, self.max_frames).long()
            features = [features[i] for i in indices]
        else:
            # Pad with zeros
            pad_len = self.max_frames - len(features)
            features = features + [torch.zeros_like(features[0])] * pad_len
        
        features = torch.stack(features)  # (max_frames, feature_dim)
        mask = torch.zeros(self.max_frames)
        mask[:min(len(data['features']), self.max_frames)] = 1.0
        
        return features, label, mask
