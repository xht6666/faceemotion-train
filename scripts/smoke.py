"""冒烟测试：模型可构建、前向可运行，并测量单个 epoch 训练耗时。"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.datasets import make_loaders, seed_everything
from src.models import MODEL_NAMES, build_model, count_params


def main() -> None:
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    from src.task import NUM_CLASSES, task_title
    print(f"任务: {task_title()}  类别数: {NUM_CLASSES}")

    print("\n== 参数量 ==")
    for name in MODEL_NAMES:
        m = build_model(name).to(device)
        total, train = count_params(m)
        y = m(torch.randn(2, 3, 48, 48).to(device))
        print(f"  {name:18s} {total/1e6:8.3f}M  out={tuple(y.shape)}")

    print("\n== 单 epoch 计时（train+valid）==")
    train_l, valid_l, test_l = make_loaders(batch_size=128, num_workers=4)
    print("  batches:", len(train_l), "valid:", len(valid_l))
    for name in ("cnn", "vgg", "resnet18", "resnet_lite_se"):
        model = build_model(name).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        t0 = time.time()
        model.train()
        for i, (x, y) in enumerate(train_l):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        t1 = time.time()
        model.eval()
        with torch.no_grad():
            for x, y in valid_l:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    model(x)
        t2 = time.time()
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {name:18s} train {t1-t0:6.1f}s  valid {t2-t1:5.1f}s  "
              f"peak_mem {mem:.2f}GB")
        torch.cuda.reset_peak_memory_stats()
        del model, opt


if __name__ == "__main__":
    main()
