"""
MobileNetV3-Small with CSAM attention module.

Architecture:
- MobileNetV3-Small backbone (pretrained on ImageNet)
- CSAM inserted after selected inverted residual blocks
- Classification head: GAP -> FC(256, ReLU) -> Dropout(0.5) -> FC(num_classes)
"""

import torch
import torch.nn as nn
import torchvision.models as models

from .csam import CSAM


class MobileNetV3CSAM(nn.Module):
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

        # Load pretrained MobileNetV3-Small
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # Sequential of inverted residual blocks

        # Insert CSAM after each block
        if use_csam:
            self.csam_modules = nn.ModuleList()
            for block in self.features:
                # Get output channels of each block
                # MobileNetV3-Small blocks output varying channels
                # We'll determine channels dynamically
                self.csam_modules.append(None)  # placeholder

            # Build CSAM for each block by checking output channels
            self._build_csam_modules(csam_reduction, csam_kernel)
        else:
            self.csam_modules = None

        # Determine final feature dimension
        self._feature_dim = self._get_feature_dim()

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._feature_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def _get_feature_dim(self):
        """Get the output channel dimension of the last feature block."""
        # MobileNetV3-Small last block output channels
        # The last inverted residual block in mobilenet_v3_small outputs 576 channels
        for i in reversed(range(len(self.features))):
            block = self.features[i]
            if hasattr(block, 'block'):
                # InvertedResidual has a 'block' attribute (Sequential)
                for layer in reversed(list(block.block)):
                    if isinstance(layer, nn.Conv2d):
                        return layer.out_channels
            elif hasattr(block, 'conv'):
                for layer in reversed(list(block.conv)):
                    if isinstance(layer, nn.Conv2d):
                        return layer.out_channels
        return 576  # default for MobileNetV3-Small

    def _build_csam_modules(self, reduction, kernel_size):
        """Build CSAM modules with correct channel dimensions."""
        self.csam_modules = nn.ModuleList()
        dummy = torch.randn(1, 3, 224, 224)
        x = dummy
        for i, block in enumerate(self.features):
            x = block(x)
            channels = x.shape[1]
            self.csam_modules.append(CSAM(channels, reduction))

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
        """Return backbone parameters (for stage 2 unfreezing)."""
        return list(self.features.parameters())

    def get_head_params(self):
        """Return classification head parameters."""
        return list(self.classifier.parameters())

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for p in self.features.parameters():
            p.requires_grad = False

    def freeze_bn(self):
        """Set all BatchNorm layers to eval mode (freeze running stats).
        Critical for fine-tuning: prevents small-batch noise from destabilizing
        pretrained BN statistics."""
        for m in self.features.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    def unfreeze_last_n_blocks(self, n: int = 3):
        """Unfreeze the last n blocks of the backbone."""
        # Freeze all first
        self.freeze_backbone()
        # Unfreeze last n
        total = len(self.features)
        for i in range(max(0, total - n), total):
            for p in self.features[i].parameters():
                p.requires_grad = True
            if self.use_csam and self.csam_modules is not None:
                for p in self.csam_modules[i].parameters():
                    p.requires_grad = True
