"""
Baseline models for comparison:
- VGG-16: Pretrained, replace classifier
- ResNet-50: Pretrained, replace final FC
- CNN+LSTM: Simple CNN features fed into LSTM
- 3D-CNN: Simple 3D convolutional network (operates on stacked frames)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGG16Baseline(nn.Module):
    """VGG-16 with modified classifier for engagement detection."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vgg16(weights=weights)
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3):
        self.freeze_backbone()
        features = list(self.backbone.features.children())
        total = len(features)
        for i in range(max(0, total - n * 4), total):
            for p in features[i].parameters():
                p.requires_grad = True

    def get_backbone_params(self):
        return list(self.backbone.features.parameters())

    def get_head_params(self):
        return list(self.backbone.classifier.parameters())


class ResNet50Baseline(nn.Module):
    """ResNet-50 with modified final FC."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            if "fc" not in name:
                p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3):
        self.freeze_backbone()
        # Unfreeze layer4 and possibly layer3
        children = list(self.backbone.children())
        for child in children[-(n + 1):-1]:  # skip fc
            for p in child.parameters():
                p.requires_grad = True

    def get_backbone_params(self):
        return [p for n, p in self.backbone.named_parameters() if "fc" not in n]

    def get_head_params(self):
        return list(self.backbone.fc.parameters())


class CNNLSTMBaseline(nn.Module):
    """CNN feature extractor followed by LSTM for temporal modeling.

    Since we process single frames, the LSTM operates on a sequence of
    spatial feature patches. For simplicity, we reshape the CNN feature
    map into a sequence.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        # Use a lightweight CNN backbone
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(
            input_size=576,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        feat = self.cnn(x)  # (B, 576, 7, 7)
        B, C, H, W = feat.shape
        # Reshape to sequence: (B, H*W, C)
        feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        lstm_out, _ = self.lstm(feat)  # (B, H*W, 256)
        # Take last hidden state
        lstm_feat = lstm_out[:, -1, :]  # (B, 256)
        return self.classifier(lstm_feat)

    def freeze_backbone(self):
        for p in self.cnn.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3):
        self.freeze_backbone()
        children = list(self.cnn.children())
        total = len(children)
        for i in range(max(0, total - n), total):
            for p in children[i].parameters():
                p.requires_grad = True

    def get_backbone_params(self):
        return list(self.cnn.parameters())

    def get_head_params(self):
        params = list(self.lstm.parameters()) + list(self.classifier.parameters())
        return params


class ThreeDCNNBaseline(nn.Module):
    """Simple 3D-CNN for engagement detection.

    Operates on single frames by replicating across a pseudo-temporal dimension,
    or can accept multi-frame input.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True, num_frames: int = 8):
        super().__init__()
        self.num_frames = num_frames

        self.features = nn.Sequential(
            # Conv3d block 1
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Conv3d block 2
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Conv3d block 3
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Conv3d block 4
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 3, 224, 224) - single frame
        # Replicate across temporal dimension
        B = x.shape[0]
        x = x.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)  # (B, 3, T, 224, 224)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3):
        self.freeze_backbone()
        children = list(self.features.children())
        total = len(children)
        for i in range(max(0, total - n * 3), total):
            if hasattr(children[i], 'parameters'):
                for p in children[i].parameters():
                    p.requires_grad = True

    def get_backbone_params(self):
        return list(self.features.parameters())

    def get_head_params(self):
        return list(self.classifier.parameters())


def build_model(model_name: str, num_classes: int = 4, pretrained: bool = True, **kwargs):
    """Factory function to build models by name."""
    models_map = {
        "vgg16": VGG16Baseline,
        "resnet50": ResNet50Baseline,
        "cnn_lstm": CNNLSTMBaseline,
        "3d_cnn": ThreeDCNNBaseline,
    }
    if model_name not in models_map:
        raise ValueError(f"Unknown baseline model: {model_name}. Choose from {list(models_map.keys())}")
    return models_map[model_name](num_classes=num_classes, pretrained=pretrained, **kwargs)
