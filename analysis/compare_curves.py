"""对比曲线生成：架构 / 优化器 / 学习率调度 / 数据增强 / SE / 残差消融。

用法::

    python analysis/compare_curves.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.task import CURVE_DIR, RESULT_DIR, SUFFIX  # noqa: E402

for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def load(name: str) -> dict | None:
    p = os.path.join(RESULT_DIR, f"{name}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def overlay(specs: list[tuple[str, str]], field: str, ylabel: str, title: str,
            fname: str, best_marker: bool = True) -> str | None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for name, label in specs:
        r = load(name)
        if r is None:
            continue
        plotted = True
        ys = [h[field] for h in r["history"]]
        ax.plot([h["epoch"] for h in r["history"]], ys, label=label, lw=1.8)
        if best_marker and field == "val_acc":
            be = r["convergence"]["best_epoch"]
            ax.scatter([be], [max(ys)], s=28, zorder=5)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(CURVE_DIR, fname)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def residual_analysis() -> str | None:
    """残差消融：训练/验证准确率与泛化差距（过拟合）双图。"""
    a, b = load("resnet18"), load("resnet18_plain")
    if a is None or b is None:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for r, label in ((a, "ResNet-18（残差连接）"),
                     (b, "ResNet-18（关闭残差，同构）")):
        ep = [h["epoch"] for h in r["history"]]
        axes[0].plot(ep, [h["train_acc"] for h in r["history"]],
                     ls="--", alpha=0.8, label=f"{label} train")
        axes[0].plot(ep, [h["val_acc"] for h in r["history"]],
                     lw=1.8, label=f"{label} val")
        axes[2].plot(ep, [h["generalization_gap"] for h in r["history"]],
                     lw=1.8, label=label)
        # 逐层梯度（由 grad_analysis 提供）
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("训练/验证准确率：残差 vs 无残差"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot([h["epoch"] for h in a["history"]],
                 [h["train_loss"] for h in a["history"]], label="残差 train")
    axes[1].plot([h["epoch"] for h in a["history"]],
                 [h["val_loss"] for h in a["history"]], label="残差 val")
    axes[1].plot([h["epoch"] for h in b["history"]],
                 [h["train_loss"] for h in b["history"]], ls="--", label="无残差 train")
    axes[1].plot([h["epoch"] for h in b["history"]],
                 [h["val_loss"] for h in b["history"]], ls="--", label="无残差 val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    axes[1].set_title("损失曲线"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("train - val 准确率")
    axes[2].set_title("泛化差距（过拟合程度）"); axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(CURVE_DIR, f"residual_ablation{SUFFIX}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(CURVE_DIR, exist_ok=True)
    jobs = [
        ([( "cnn", "CNN 0.42M"), ("vgg", "VGG-16 14.85M"),
          ("resnet18", "ResNet-18 11.17M")],
         "val_acc", "验证集准确率", "架构对比：收敛速度与最终精度", "cmp_arch.png"),
        ([("resnet32", "ResNet-32 残差 23.71M"),
          ("resnet32_plain", "ResNet-32 关闭残差 23.71M")],
         "val_acc", "验证集准确率", "深层网络退化实验：32 层残差 vs 纯网络", "cmp_deep.png"),
        ([("resnet18", "Adam lr=1e-3"), ("resnet18_sgd", "SGD momentum=0.9 lr=0.1")],
         "val_acc", "验证集准确率", "优化器对比：Adam vs SGD", "cmp_optimizer.png"),
        ([("resnet18", "CosineAnnealing"), ("resnet18_steplr", "StepLR(step=20, γ=0.1)")],
         "val_acc", "验证集准确率", "学习率调度对比：Cosine vs StepLR", "cmp_scheduler.png"),
        ([("resnet_lite", "轻量化 ResNet 2.76M"), ("resnet_lite_se", "轻量化 + SE 2.78M"),
          ("resnet18", "ResNet-18 11.17M（参照）")],
         "val_acc", "验证集准确率", "SE 注意力：轻量化网络逼近大模型精度", "cmp_se.png"),
        ([("resnet18", "使用数据增强"), ("resnet18_noaug", "关闭数据增强")],
         "val_acc", "验证集准确率", "数据增强消融：随机裁剪/翻转/旋转/色彩抖动", "cmp_aug.png"),
        ([("resnet18", "train"), ],
         "train_acc", "训练准确率", "训练准确率（收敛速度）", "cmp_train_acc"),
    ]
    for specs, field, ylabel, title, fname in jobs:
        if not fname.endswith(".png"):
            continue
        out = overlay(specs, field, ylabel, title, fname[:-4] + SUFFIX + ".png")
        print("->", out)

    # 残差消融
    print("->", residual_analysis())
    # 泛化差距（架构对比）
    out = overlay([("cnn", "CNN"), ("vgg", "VGG-16"),
                   ("resnet18", "ResNet-18"), ("resnet18_plain", "ResNet-18 无残差")],
                  "generalization_gap", "train - val 准确率",
                  "泛化差距（过拟合程度）对比", f"cmp_gap{SUFFIX}.png")
    print("->", out)


if __name__ == "__main__":
    main()
