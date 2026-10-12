"""模型工厂：按名称构建模型，并输出参数量统计。"""

from __future__ import annotations

import torch.nn as nn

from .cnn import SimpleCNN
from .resnet import ResNet, resnet10_lite, resnet18, resnet32
from .vgg import VGG

try:
    from ..task import NUM_CLASSES as _TASK_NUM_CLASSES
except Exception:                       # 独立使用时退化为 5 类
    _TASK_NUM_CLASSES = 5

__all__ = ["build_model", "count_params", "MODEL_NAMES"]


def build_model(name: str, num_classes: int | None = None, in_channels: int = 3,
                **kw) -> nn.Module:
    if num_classes is None:
        num_classes = _TASK_NUM_CLASSES
    name = name.lower()
    if name == "cnn":                       # 轻量纯 CNN（无残差）
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels, **kw)
    if name in ("vgg", "vgg16"):
        return VGG("vgg16", num_classes=num_classes, in_channels=in_channels, **kw)
    if name == "vgg11":
        return VGG("vgg11", num_classes=num_classes, in_channels=in_channels, **kw)
    if name == "resnet18":
        return resnet18(num_classes=num_classes, in_channels=in_channels, **kw)
    if name == "resnet18_plain":            # 去掉恒等短路的对照网络
        return resnet18(num_classes=num_classes, in_channels=in_channels,
                        residual=False, **kw)
    if name == "resnet_lite":               # 轻量化 ResNet（1.5M）
        return resnet10_lite(num_classes=num_classes, in_channels=in_channels, **kw)
    if name == "resnet_lite_se":            # 轻量化 + SE 注意力
        return resnet10_lite(num_classes=num_classes, in_channels=in_channels, se=True, **kw)
    if name == "resnet18_se":               # 标准 ResNet + SE
        return resnet18(num_classes=num_classes, in_channels=in_channels, se=True, **kw)
    if name == "resnet32":
        return resnet32(num_classes=num_classes, in_channels=in_channels, **kw)
    if name == "resnet32_plain":
        return resnet32(num_classes=num_classes, in_channels=in_channels,
                        residual=False, **kw)
    raise ValueError(f"unknown model: {name}")


MODEL_NAMES = ["cnn", "vgg", "resnet18", "resnet18_plain", "resnet32", "resnet32_plain",
               "resnet_lite", "resnet_lite_se", "resnet18_se"]


def count_params(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
