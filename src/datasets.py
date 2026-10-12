"""FER+ 6 分类数据集读取与数据增强（类别/路径/归一化统计见 src/task.py）。

数据组织::

    FER2013Train/  label.csv  fer0000001.png ...
    FER2013Valid/  label.csv  ...
    FER2013Test/   label.csv  ...

label.csv 每行::

    图像名, 使用区域, neutral, happiness, surprise, sadness, anger, fear, unknown

标签 = 前 6 个情绪列的 argmax（已逐行校验 unknown 永远不会胜出）。
"""

from __future__ import annotations

import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .task import (CLASS_NAMES, CLASS_NAMES_ZH, DATA_ROOT, NUM_CLASSES,
                   NUM_VOTE_COLS, PIXEL_MEAN, PIXEL_STD)

IMAGE_SIZE = 48

__all__ = ["CLASS_NAMES", "CLASS_NAMES_ZH", "NUM_CLASSES", "DATA_ROOT",
           "PIXEL_MEAN", "PIXEL_STD", "IMAGE_SIZE", "read_labels",
           "FERDataset", "FER5Dataset", "train_transforms", "eval_transforms",
           "make_loaders", "class_counts", "seed_everything"]


def read_labels(split_csv: str) -> list[tuple[str, int]]:
    """读取 label.csv，返回 [(文件名, 类别id)], 类别id ∈ [0, NUM_CLASSES)。"""
    samples: list[tuple[str, int]] = []
    with open(split_csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            votes = [int(x) for x in row[2:2 + NUM_VOTE_COLS]]  # 情绪列 + unknown
            y = int(np.argmax(votes[:NUM_CLASSES]))
            samples.append((row[0], y))
    return samples


class FERDataset(Dataset):
    """单个 split 的 FER+ 数据集（6 类）。"""

    def __init__(self, root: str, split: str, transform=None, color: str = "RGB"):
        self.split_dir = os.path.join(root, split)
        self.transform = transform
        self.mode = color.upper()  # RGB / L
        self.samples = read_labels(os.path.join(self.split_dir, "label.csv"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, y = self.samples[idx]
        img = Image.open(os.path.join(self.split_dir, name)).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        return img, y


# 兼容旧代码的别名
FER5Dataset = FERDataset


def train_transforms(jitter: bool = True, erasing: bool = False) -> transforms.Compose:
    """训练集增强：随机裁剪 + 翻转 + 旋转 + 色彩抖动。"""
    ops: list = [
        transforms.RandomCrop(IMAGE_SIZE, padding=4),      # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),            # 水平翻转
        transforms.RandomRotation(degrees=10, fill=0),      # 随机旋转
    ]
    if jitter:
        # 色彩抖动（亮度/对比度/饱和度/色相），人脸灰度图转换为 RGB 后可完整生效
        ops.append(
            transforms.ColorJitter(brightness=0.35, contrast=0.35,
                                   saturation=0.25, hue=0.05)
        )
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
    ]
    if erasing:
        ops.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.5, 2.0)))
    return transforms.Compose(ops)


def eval_transforms() -> transforms.Compose:
    """验证/测试：仅归一化。"""
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(PIXEL_MEAN, PIXEL_STD)])


def make_loaders(root: str = DATA_ROOT, batch_size: int = 128, num_workers: int = 4,
                  jitter: bool = True, pin_memory: bool = True
                  ) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set = FER5Dataset(root, "FER2013Train", train_transforms(jitter))
    valid_set = FER5Dataset(root, "FER2013Valid", eval_transforms())
    test_set = FER5Dataset(root, "FER2013Test", eval_transforms())

    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=False, generator=g, persistent_workers=num_workers > 0)
    valid_loader = DataLoader(valid_set, batch_size=batch_size * 2, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=num_workers > 0)
    return train_loader, valid_loader, test_loader


def class_counts(root: str) -> np.ndarray:
    """训练集各类样本数，用于分析类别不平衡。"""
    cnt = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, y in read_labels(os.path.join(root, "FER2013Train", "label.csv")):
        cnt[y] += 1
    return cnt


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
