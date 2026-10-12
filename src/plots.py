"""可视化：训练曲线、混淆矩阵、模型对比图。"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

from .task import CURVE_DIR, REPORT_DIR, SUFFIX


def plot_curves(result: dict) -> str:
    h = result["history"]
    ep = [r["epoch"] for r in h]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    ax = axes[0]
    ax.plot(ep, [r["train_loss"] for r in h], label="train loss")
    ax.plot(ep, [r["val_loss"] for r in h], label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"{result['exp_name']}  loss")

    ax = axes[1]
    ax.plot(ep, [r["train_acc"] for r in h], label="train acc")
    ax.plot(ep, [r["val_acc"] for r in h], label="val acc")
    be = result["convergence"]["best_epoch"]
    bv = result["convergence"]["best_val_acc"]
    ax.scatter([be], [bv], color="red", zorder=5)
    ax.annotate(f"best {bv*100:.2f}% @ep{be}", (be, bv),
                textcoords="offset points", xytext=(-90, -18), color="red")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("accuracy (泛化差距 = train - val)")

    ax = axes[2]
    ax.plot(ep, [r["lr"] for r in h], color="tab:green")
    ax.set_xlabel("epoch"); ax.set_ylabel("lr"); ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.set_title(f"lr schedule: {result['config']['scheduler']} / "
                 f"{result['config']['optimizer']}")

    fig.tight_layout()
    os.makedirs(CURVE_DIR, exist_ok=True)
    path = os.path.join(CURVE_DIR, f"{result['exp_name']}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_confusion(result: dict) -> str | None:
    cm = result.get("confusion_matrix")
    if not cm:
        return None
    cm = np.array(cm, dtype=float)
    names = result["class_names_zh"]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    color="white" if v > cm.max() * 0.55 else "black", fontsize=9)
    ax.set_title(f"{result['exp_name']}  测试集混淆矩阵  "
                 f"acc={result['test_accuracy']*100:.2f}%")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, f"confusion{SUFFIX}_{result['exp_name']}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_accuracy_params(results: list[dict]) -> str:
    """精度 vs 参数量散点（柱状按准确率排序）。"""
    results = sorted(results, key=lambda r: (r.get("test_accuracy") or 0))
    names = [f"{r['model']}" for r in results]
    accs = [(r.get("test_accuracy") or 0) * 100 for r in results]
    pars = [r["params_M"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, accs, color="tab:blue", alpha=0.75)
    ax.set_xlabel("测试集准确率 (%)")
    ax.set_xlim(0, max(accs) + 8)
    for b, a, p in zip(bars, accs, pars):
        ax.text(a + 0.4, b.get_y() + b.get_height() / 2,
                f"{a:.2f}%  ({p:.2f}M)", va="center", fontsize=9)
    ax.set_title("各架构：准确率 vs 参数量")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, f"accuracy_vs_params{SUFFIX}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_gradient_norms(grad_rows: dict[str, list]) -> str:
    """各层梯度 L2 范数对比（纯网络 vs 残差网络）。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, rows in grad_rows.items():
        xs = list(range(1, len(rows) + 1))
        ax.semilogy(xs, rows, marker="o", ms=4, label=name)
    ax.set_xlabel("卷积层序号（从浅到深）")
    ax.set_ylabel("梯度 L2 范数（log）")
    ax.set_title("反向传播梯度范数随深度的变化：残差连接缓解梯度消失")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, f"grad_norms{SUFFIX}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
