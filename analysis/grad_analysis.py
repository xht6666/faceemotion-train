"""残差连接的梯度流分析：纯网络 vs 残差网络的逐层梯度范数。

原理：反向传播时

    纯网络：  dx_l = W_l·W_{l-1}···W_1 · dL/dy   （连乘，逐层衰减/爆炸）
    残差网络：dx_l = (I + ∂F_l/∂x)···(I + ∂F_1/∂x) · dL/dy  （含 1 的加性项，
              Jacobian 行列式 ≥ 1，梯度可无损回传到浅层）

脚本对每个卷积层的**输出激活**取梯度 L2 范数，横向对比不同架构，
同时给出首尾层衰减比 / 每层几何衰减率，用于报告的定量结论。

用法::

    python analysis/grad_analysis.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.datasets import (CLASS_NAMES, DATA_ROOT, FER5Dataset, NUM_CLASSES,
                          eval_transforms, seed_everything)
from src.models import build_model
from src.plots import plot_gradient_norms
from src.task import CKPT_DIR, GRAD_DIR as OUT_DIR

BATCH = 128


def collect_conv_grad_norms(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                            use_amp: bool = False) -> dict[str, float]:
    """前向挂 hook 捕获每个 Conv2d 输出激活的梯度 L2 范数。"""
    grads: dict[str, tuple[float, int]] = {}   # name -> (L2 范数, 元素数)
    handles = []

    def make_hook(name: str):
        def hook(_m, _inp, out: torch.Tensor):
            if isinstance(out, torch.Tensor) and out.requires_grad:
                out.register_hook(
                    lambda g: grads.__setitem__(name, (float(g.norm()), g.numel())))
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(make_hook(name)))

    model.train()
    model.zero_grad(set_to_none=True)
    crit = nn.CrossEntropyLoss()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        logits = model(x)
        loss = crit(logits, y)
    loss.backward()
    for h in handles:
        h.remove()
    # hook 按前向执行顺序插入，dict 保留该顺序（即层序）
    ordered = dict(grads)
    model.zero_grad(set_to_none=True)
    return ordered


def decay_stats(rows: list[tuple[float, int]]) -> dict:
    """rows: [(梯度L2范数, 激活元素数)]; 用 RMS=范数/sqrt(N) 消除张量尺寸影响。"""
    a = np.array([n for n, _ in rows], dtype=np.float64)
    rms = np.array([n / np.sqrt(max(k, 1)) for n, k in rows], dtype=np.float64)
    ratios = rms[1:] / np.maximum(rms[:-1], 1e-12)
    return {
        "first_layer_rms": float(rms[0]),
        "last_layer_rms": float(rms[-1]),
        "head_tail_rms_ratio": float(rms[-1] / max(rms[0], 1e-12)),  # >1: 浅层梯度被削弱
        "geomean_ratio_per_layer": float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-12))))),
        "dynamic_range": float(rms.max() / max(rms.min(), 1e-12)),
        "head_tail_norm_ratio": float(a[-1] / max(a[0], 1e-12)),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_everything(123)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = FER5Dataset(DATA_ROOT, "FER2013Test", eval_transforms())
    g = torch.Generator().manual_seed(7)
    idx = torch.randperm(len(ds), generator=g)[:BATCH]
    xs = torch.stack([ds[i][0] for i in idx.tolist()])
    ys = torch.tensor([ds[i][1] for i in idx.tolist()], dtype=torch.long)
    xs, ys = xs.to(dev), ys.to(dev)

    targets = [("vgg", "VGG-16（纯卷积堆叠，13 层卷积）"),
               ("resnet18_plain", "ResNet-18 关闭残差（同构对照）"),
               ("resnet18", "ResNet-18（残差连接）"),
               ("resnet32_plain", "ResNet-32 关闭残差（深层纯网络）"),
               ("resnet32", "ResNet-32（深层残差网络）")]

    rms_rows: dict[str, list[float]] = {}
    norm_rows: dict[str, list[float]] = {}
    numel_rows: dict[str, list[int]] = {}
    summary: dict[str, dict] = {}

    for tag, label in targets:
        for phase in ("init", "trained"):
            ck = os.path.join(CKPT_DIR, f"{tag}.pt")
            if phase == "trained" and not os.path.exists(ck):
                continue
            seed_everything(123)
            try:
                model = build_model(tag).to(dev)
            except Exception as e:          # 该模型尚未实现则跳过
                print(f"skip {tag}: {e}")
                break
            if phase == "trained":
                state = torch.load(ck, map_location=dev, weights_only=False)
                model.load_state_dict(state["model"])

            grads = collect_conv_grad_norms(model, xs, ys, use_amp=False)
            items = list(grads.values())                    # [(norm, numel)]
            rms = [n / np.sqrt(max(k, 1)) for n, k in items]
            key = f"{tag}_{phase}"
            norm_rows[key] = [n for n, _ in items]
            numel_rows[key] = [k for _, k in items]
            rms_rows[key] = rms
            summary[key] = {"label": label, "phase": phase,
                            "n_conv": len(items), **decay_stats(items)}
            s = summary[key]
            print(f"{key:26s} n_conv={len(items):2d} "
                  f"RMS_first={s['first_layer_rms']:.3e} "
                  f"RMS_last={s['last_layer_rms']:.3e} "
                  f"last/first={s['head_tail_rms_ratio']:.1f} "
                  f"per-layer x{s['geomean_ratio_per_layer']:.3f} "
                  f"range={s['dynamic_range']:.1f}")
            del model
            torch.cuda.empty_cache()

    # 长格式 CSV：model_phase, layer, norm, numel, rms
    csv_path = os.path.join(OUT_DIR, "grad_norms.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model_phase", "label", "phase", "layer_idx",
                    "grad_l2_norm", "activation_numel", "grad_rms"])
        for key in rms_rows:
            for i, (n, k, r) in enumerate(zip(norm_rows[key], numel_rows[key],
                                              rms_rows[key]), start=1):
                w.writerow([key, summary[key]["label"], summary[key]["phase"],
                            i, f"{n:.6e}", k, f"{r:.6e}"])
    with open(os.path.join(OUT_DIR, "grad_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 画训练后的 RMS 曲线
    trained = {summary[k]["label"]: v for k, v in rms_rows.items()
               if k.endswith("_trained")}
    out = plot_gradient_norms(trained) if trained else None
    print("csv ->", csv_path)
    print("plot ->", out)
    for k, s in summary.items():
        print(f"  {k:26s} last/first={s['head_tail_rms_ratio']:8.1f}  "
              f"per-layer x{s['geomean_ratio_per_layer']:.4f}  "
              f"range={s['dynamic_range']:7.1f}")


if __name__ == "__main__":
    main()
