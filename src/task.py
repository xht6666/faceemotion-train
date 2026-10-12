"""本工程的固定配置：**FER+ 6 分类任务**（中性/快乐/惊讶/悲伤/愤怒/恐惧）。

数据组织（直接位于项目根目录）::

    FER2013Train/  label.csv + 21685 张 48x48 PNG
    FER2013Valid/  label.csv +  2841 张
    FER2013Test/   label.csv +  2713 张

label.csv 每行的投票列为::

    neutral, happiness, surprise, sadness, anger, fear, unknown

标签 = 前 6 个情绪列的 argmax（已逐行校验 unknown 永不胜出）。

所有输出（结果 JSON / 曲线 / 混淆矩阵 / 权重 / 梯度数据）都直接写入
``reports/`` 与 ``checkpoints/``，不带后缀。
"""

from __future__ import annotations

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK = "6class"
DATA_ROOT = PROJECT_ROOT

CLASS_NAMES = ["neutral", "happiness", "surprise", "sadness", "anger", "fear"]
CLASS_NAMES_ZH = ["中性", "快乐", "惊讶", "悲伤", "愤怒", "恐惧"]

# 训练集 RGB 归一化统计量（21685 张训练图片实测）
PIXEL_MEAN = (0.5070, 0.5070, 0.5070)
PIXEL_STD = (0.2539, 0.2539, 0.2539)

NUM_CLASSES = len(CLASS_NAMES)
# label.csv 中投票列数 = 情绪列 + unknown 列
NUM_VOTE_COLS = NUM_CLASSES + 1

# ---------------------------------------------------------------- 输出目录
SUFFIX = ""                              # 独立 6 分类工程，无需后缀
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
RESULT_DIR = os.path.join(REPORT_DIR, "results")
CURVE_DIR = os.path.join(REPORT_DIR, "curves")
GRAD_DIR = os.path.join(REPORT_DIR, "grads")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def task_title() -> str:
    """报告中使用的任务名。"""
    return "6 分类（含 fear）"
