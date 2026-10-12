"""架构一：从零实现的轻量 CNN（纯卷积堆叠，无残差连接）。

结构：4 个 [Conv3x3 - BN - ReLU - MaxPool] 阶段 + 全局平均池化 + 两层 MLP。
"""

from __future__ import annotations

import torch.nn as nn

from .se import SEModule


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """轻量 CNN 基线（默认通道 32/64/128/256）。"""

    def __init__(self, num_classes: int = 5, widths=(32, 64, 128, 256),
                 in_channels: int = 3, se: bool = False, dropout: float = 0.3):
        super().__init__()
        w = list(widths)
        layers = [_conv_block(in_channels, w[0])]
        for i in range(1, len(w)):
            layers.append(_conv_block(w[i - 1], w[i]))
        self.features = nn.Sequential(*layers)
        self.se = SEModule(w[-1], reduction=8) if se else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(w[-1], max(w[-1] // 2, 32)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(w[-1] // 2, 32), num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        if self.se is not None:
            x = self.se(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
