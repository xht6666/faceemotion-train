"""架构三：ResNet（He et al., 2016）从零实现（不使用任何预训练权重）。

针对 48x48 小图的适配：
* stem 用 3x3/s1 卷积且不接最大池化（保留空间分辨率）；
* 4 个 stage 分辨率 48 -> 24 -> 12 -> 6，末端全局平均池化。

额外能力（用于消融分析）：
* ``residual=False``  -> 关闭恒等短路，得到层数/参数量完全一致的"纯深度网络"，
   用于对照残差连接对梯度消失与退化问题的缓解作用；
* ``se=True``         -> 在每个残差块末端插入 SE 通道注意力。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .se import SEModule


def conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


def conv1x1(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int = 1,
                 residual: bool = True, se: bool = False, reduction: int = 16):
        super().__init__()
        self.residual = residual
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SEModule(cout, reduction=reduction) if se else None
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(conv1x1(cin, cout, stride),
                                            nn.BatchNorm2d(cout))
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        if self.residual and self.downsample is not None:
            identity = self.downsample(x)
        if self.residual:          # y = F(x) + x   （关键的恒等映射）
            out = out + identity
        # else: 纯网络 y = F(x)，仅用于对照实验（层数/参数量与残差版完全一致）
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, layers: Sequence[int], base_width: int = 64,
                 num_classes: int = 5, in_channels: int = 3,
                 residual: bool = True, se: bool = False, reduction: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.residual = residual
        self.in_ch = base_width
        self.stem = nn.Sequential(
            conv3x3(in_channels, base_width),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_width, layers[0], stride=1, se=se, reduction=reduction)
        self.layer2 = self._make_layer(base_width * 2, layers[1], stride=2, se=se, reduction=reduction)
        self.layer3 = self._make_layer(base_width * 4, layers[2], stride=2, se=se, reduction=reduction)
        self.layer4 = self._make_layer(base_width * 8, layers[3], stride=2, se=se, reduction=reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(base_width * 8, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, cout: int, blocks: int, stride: int,
                    se: bool, reduction: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_ch, cout, stride,
                             residual=self.residual, se=se, reduction=reduction)]
        self.in_ch = cout
        for _ in range(1, blocks):
            layers.append(BasicBlock(cout, cout, residual=self.residual,
                                     se=se, reduction=reduction))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


def resnet18(num_classes: int = 5, **kw) -> ResNet:
    return ResNet([2, 2, 2, 2], base_width=64, num_classes=num_classes, **kw)


def resnet10_lite(num_classes: int = 5, **kw) -> ResNet:
    """轻量化 ResNet：1/4 通道宽度 + 每 stage 1 个块（约 2.9M -> 1.5M 参数）。"""
    return ResNet([1, 1, 1, 1], base_width=48, num_classes=num_classes, **kw)


def resnet32(num_classes: int = 5, **kw) -> ResNet:
    """深层 ResNet（32 层卷积），用于退化/梯度消失对照实验。"""
    return ResNet([4, 4, 4, 4], base_width=64, num_classes=num_classes, **kw)
