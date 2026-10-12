"""架构二：VGGNet（Simonyan & Zisserman, 2015）从零实现。

针对 48x48 输入的适配：
* 5 个 MaxPool2d 将特征图 48 -> 24 -> 12 -> 6 -> 3 -> 1；
* 分类头用全局平均池化代替 VGG 原版 7x7 展平的全连接，避免小图上的参数爆炸。

支持 ``cfg`` 选择深度：vgg11（8 层卷积）/ vgg16（13 层卷积）。
"""

from __future__ import annotations

import torch.nn as nn

from .se import SEModule

# 'M' 表示 MaxPool，其余为卷积通道数
VGG_CFGS = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, cfg: str = "vgg16", num_classes: int = 5,
                 in_channels: int = 3, se: bool = False, dropout: float = 0.5):
        super().__init__()
        if cfg not in VGG_CFGS:
            raise ValueError(f"unknown vgg cfg: {cfg}")
        self.cfg_name = cfg
        features, cin = [], in_channels
        for c in VGG_CFGS[cfg]:
            if c == "M":
                features.append(nn.MaxPool2d(2, 2))
            else:
                features += [
                    nn.Conv2d(cin, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                ]
                cin = c
        self.features = nn.Sequential(*features)
        self.se = SEModule(cin, reduction=16) if se else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(cin, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        if self.se is not None:
            x = self.se(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
