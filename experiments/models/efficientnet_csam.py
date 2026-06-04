"""
EfficientNet-B0 with CSAM attention module.

Same pattern as MobileNetV3+CSAM but with EfficientNet-B0 backbone.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from .csam import CSAM


class EfficientNetCSAM(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        use_csam: bool = True,
        csam_reduction: int = 4,
        csam_kernel: int = 7,
        head_hidden: int = 256,
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.use_csam = use_csam

        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features  # Sequential of MBConv blocks

        # Insert CSAM after each block
        if use_csam:
            self._build_csam_modules(csam_reduction, csam_kernel)
        else:
            self.csam_modules = None

        # EfficientNet-B0 final feature dimension: 1280
        self._feature_dim = 1280

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._feature_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def _build_csam_modules(self, reduction, kernel_size):
        """Build CSAM modules with correct channel dimensions."""
        self.csam_modules = nn.ModuleList()
        dummy = torch.randn(1, 3, 224, 224)
        x = dummy
        for block in self.features:
            x = block(x)
            channels = x.shape[1]
            self.csam_modules.append(CSAM(channels, reduction, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_csam and self.csam_modules is not None:
            for block, csam in zip(self.features, self.csam_modules):
                x = block(x)
                x = csam(x)
        else:
            x = self.features(x)

        x = self.classifier(x)
        return x

    def get_backbone_params(self):
        return list(self.features.parameters())

    def get_head_params(self):
        return list(self.classifier.parameters())

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def freeze_bn(self):
        """Set all BatchNorm layers to eval mode (freeze running stats)."""
        for m in self.features.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    def unfreeze_last_n_blocks(self, n: int = 3):
        self.freeze_backbone()
        total = len(self.features)
        for i in range(max(0, total - n), total):
            for p in self.features[i].parameters():
                p.requires_grad = True
            if self.use_csam and self.csam_modules is not None:
                for p in self.csam_modules[i].parameters():
                    p.requires_grad = True
