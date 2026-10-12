"""实验编排：架构对比 / 消融（残差、优化器、学习率调度、SE 注意力、类别权重）。

用法::

    python experiments.py --list
    python experiments.py --only resnet18
    python experiments.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.plots import plot_confusion, plot_curves
from src.task import DATA_ROOT, task_title
from src.trainer import RESULT_DIR, TrainConfig, Trainer

EPOCHS = 60
BATCH = 128

# 名称 -> 配置（未写出的字段取 TrainConfig 默认值：adam/cosine/增强开/AMP 开）
EXPERIMENTS: dict[str, dict] = {
    # ---------- 架构对比（同一套训练配方，公平比较） ----------
    "cnn":            dict(model="cnn", epochs=EPOCHS, batch_size=BATCH, notes="轻量纯 CNN 基线"),
    "vgg":            dict(model="vgg", epochs=EPOCHS, batch_size=BATCH, notes="VGGNet-16（纯卷积堆叠）"),
    "resnet18":       dict(model="resnet18", epochs=EPOCHS, batch_size=BATCH, notes="ResNet-18（主模型）"),
    # ---------- 数据增强消融 ----------
    "resnet18_noaug": dict(model="resnet18", epochs=EPOCHS, batch_size=BATCH,
                           jitter=False, notes="关闭数据增强（对照组）"),
    # ---------- 残差连接消融（层数/参数量完全一致） ----------
    "resnet18_plain": dict(model="resnet18_plain", epochs=EPOCHS, batch_size=BATCH,
                           notes="关闭恒等短路的对照组"),
    # ---------- 深层网络退化实验（32 层卷积） ----------
    "resnet32": dict(model="resnet32", epochs=EPOCHS, batch_size=BATCH,
                     notes="深层 ResNet-32（残差）"),
    "resnet32_plain": dict(model="resnet32_plain", epochs=EPOCHS, batch_size=BATCH,
                           notes="深层 ResNet-32（关闭残差，退化对照）"),
    # ---------- 优化器对比 ----------
    "resnet18_sgd":   dict(model="resnet18", epochs=EPOCHS, batch_size=BATCH,
                           optimizer="sgd", lr=0.1, weight_decay=5e-4,
                           notes="SGD+momentum vs Adam"),
    # ---------- 学习率调度对比 ----------
    "resnet18_steplr": dict(model="resnet18", epochs=EPOCHS, batch_size=BATCH,
                            scheduler="steplr", step_size=20, gamma=0.1,
                            notes="StepLR vs CosineAnnealing"),
    # ---------- 轻量化 + SE 注意力 ----------
    "resnet_lite":    dict(model="resnet_lite", epochs=EPOCHS, batch_size=BATCH,
                           notes="轻量化 ResNet（2.76M）"),
    "resnet_lite_se": dict(model="resnet_lite_se", epochs=EPOCHS, batch_size=BATCH,
                           notes="轻量化 + SE 注意力"),
    # ---------- 多随机种子鲁棒性验证（与上方同配置，仅 seed 不同） ----------
    "resnet18_seed123": dict(model="resnet18", epochs=EPOCHS, batch_size=BATCH,
                             seed=123, notes="ResNet-18 主模型 seed=123"),
    "resnet18_sgd_seed123": dict(model="resnet18", epochs=EPOCHS, batch_size=BATCH,
                                 optimizer="sgd", lr=0.1, weight_decay=5e-4, seed=123,
                                 notes="SGD 组 seed=123"),
    "resnet18_steplr_seed123": dict(model="resnet18", epochs=EPOCHS, batch_size=BATCH,
                                    scheduler="steplr", step_size=20, gamma=0.1,
                                    seed=123, notes="StepLR 组 seed=123"),
    "resnet_lite_seed123": dict(model="resnet_lite", epochs=EPOCHS, batch_size=BATCH,
                                seed=123, notes="轻量化组 seed=123"),
    "resnet_lite_se_seed123": dict(model="resnet_lite_se", epochs=EPOCHS, batch_size=BATCH,
                                   seed=123, notes="轻量化+SE 组 seed=123"),
    # ---------- 类别不平衡处理（fear 仅占 1.58%） ----------
    "resnet18_cw": dict(model="resnet18", epochs=EPOCHS, batch_size=BATCH,
                        class_weight=True, notes="类别加权 CE（fear 1.58% 不平衡）"),
}


def run_one(name: str, epochs_override: int | None = None) -> dict:
    kw = dict(EXPERIMENTS[name])
    if epochs_override:
        kw["epochs"] = epochs_override
    cfg = TrainConfig(exp_name=name, **kw)
    trainer = Trainer(cfg, root=DATA_ROOT)
    print(f"\n=== [{task_title()}] {name} | model={cfg.model} params="
          f"{trainer.total_params/1e6:.3f}M epochs={cfg.epochs} "
          f"opt={cfg.optimizer} sched={cfg.scheduler} ===", flush=True)
    result = trainer.fit()
    for p in (plot_curves(result), plot_confusion(result)):
        if p:
            print("  plot ->", p, flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--epochs", type=int, default=0, help="覆盖 epoch 数（冒烟用）")
    args = ap.parse_args()

    if args.list:
        for k, v in EXPERIMENTS.items():
            print(f"  {k:18s} {v}")
        return
    print(f"任务: {task_title()}  数据根: {DATA_ROOT}", flush=True)

    if args.only:
        names = [s.strip() for s in args.only.split(",") if s.strip()]
    elif args.all:
        names = list(EXPERIMENTS)
    else:
        ap.error("需要 --all 或 --only")

    os.makedirs(RESULT_DIR, exist_ok=True)
    summary = {}
    for n in names:
        if n not in EXPERIMENTS:
            raise SystemExit(f"未知实验: {n}")
        r = run_one(n, args.epochs or None)
        summary[n] = {
            "test_acc": r["test_accuracy"],
            "best_val": r["best_val_acc"],
            "params_M": r["params_M"],
            "best_epoch": r["convergence"]["best_epoch"],
        }
        with open(os.path.join(RESULT_DIR, "_progress.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 完成 ===")
    for k, v in summary.items():
        print(f"  {k:18s} test={v['test_acc']*100:.2f}%  "
              f"val={v['best_val']*100:.2f}%  {v['params_M']:.3f}M  "
              f"best_ep={v['best_epoch']}")


if __name__ == "__main__":
    main()
