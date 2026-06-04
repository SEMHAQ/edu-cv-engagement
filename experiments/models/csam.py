"""
Channel-Spatial Attention Module (CSAM)

As described in the paper:
- Channel attention: GAP -> FC -> ReLU -> FC -> sigmoid (reduction ratio r=4)
- Spatial attention: AvgPool+MaxPool -> 7x7 Conv -> sigmoid
- Output: F' = F * M_c * M_s
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention via squeeze-and-excitation style."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial attention via avg+max pool concatenation."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(combined))
        return x * attn


class CSAM(nn.Module):
    """
    Channel-Spatial Attention Module.
    Applies channel attention then spatial attention sequentially.
    """

    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
