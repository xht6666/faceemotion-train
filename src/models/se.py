"""Squeeze-and-Excitation 通道注意力模块（SENet, Hu et al. 2018）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SEModule(nn.Module):
    """SE: 全局平均池化 -> 两层瓶颈 FC -> 逐通道缩放。

    ``F_se = x * sigmoid(W2 · σ(W1 · GAP(x)))``
    """

    def __init__(self, channels: int, reduction: int = 16, bias: bool = False):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
