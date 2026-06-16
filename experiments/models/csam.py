"""
Channel-Spatial Attention Module (CSAM) with Multi-Scale Spatial Attention

Enhanced version with:
- Channel attention: GAP -> FC -> ReLU -> FC -> sigmoid (reduction ratio r=4)
- Multi-scale spatial attention: parallel 3x3, 5x5, 7x7 convolutions
- Adaptive fusion: learnable weights for multi-scale spatial features
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


class MultiScaleSpatialAttention(nn.Module):
    """
    Multi-scale spatial attention with parallel convolutions at different scales.
    Captures both local (3x3) and global (7x7) spatial patterns.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Parallel multi-scale convolutions
        self.conv3x3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(2, 1, 5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
        # Adaptive fusion weights
        self.alpha = nn.Parameter(torch.ones(3) / 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        
        # Multi-scale attention
        attn3 = self.conv3x3(combined)
        attn5 = self.conv5x5(combined)
        attn7 = self.conv7x7(combined)
        
        # Adaptive fusion
        weights = torch.softmax(self.alpha, dim=0)
        attn = weights[0] * attn3 + weights[1] * attn5 + weights[2] * attn7
        attn = self.sigmoid(attn)
        
        return x * attn


class CSAM(nn.Module):
    """
    Channel-Spatial Attention Module with Multi-Scale Spatial Attention.
    Applies channel attention then multi-scale spatial attention sequentially.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = MultiScaleSpatialAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
