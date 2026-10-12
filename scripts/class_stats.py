"""数据集分类统计：逐 split、逐类样本数与占比（结果打印为 Markdown 表）。

用法::

    python scripts/class_stats.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.task import CLASS_NAMES, CLASS_NAMES_ZH, DATA_ROOT  # noqa: E402

SPLITS = ["FER2013Train", "FER2013Valid", "FER2013Test"]
ZH = dict(zip(CLASS_NAMES, CLASS_NAMES_ZH))


def count(split: str) -> np.ndarray:
    c = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    path = os.path.join(DATA_ROOT, split, "label.csv")
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            votes = [int(x) for x in row[2:2 + len(CLASS_NAMES) + 1]]
            c[int(np.argmax(votes[:len(CLASS_NAMES)]))] += 1
    return c


def main() -> None:
    stats = {s: count(s) for s in SPLITS}
    total = sum(stats.values())

    print("| 类别 | 中文 | 训练集 | 占比 | 验证集 | 占比 | 测试集 | 占比 | 全部 | 占比 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, name in enumerate(CLASS_NAMES):
        row = [f"`{name}`", ZH[name]]
        for s in SPLITS:
            v = int(stats[s][i])
            row += [str(v), f"{100*v/stats[s].sum():.2f}%"]
        row += [str(int(total[i])), f"{100*total[i]/total.sum():.2f}%"]
        print("| " + " | ".join(row) + " |")
    row = ["**合计**", ""]
    for s in SPLITS:
        row += [f"**{int(stats[s].sum())}**", "100%"]
    row += [f"**{int(total.sum())}**", "100%"]
    print("| " + " | ".join(row) + " |")

    # 类别不平衡度：最多类 / 最少类
    print(f"\n最多类/最少类 = {total.max()/total.min():.1f}×"
          f"（{CLASS_NAMES[int(total.argmax())]} vs {CLASS_NAMES[int(total.argmin())]}）")
    # 反频率权重（类别加权 CE 用）
    w = total.sum() / (len(CLASS_NAMES) * total.astype(np.float64))
    print("逆频率权重 w_c = " + ", ".join(f"{n}={v:.2f}" for n, v in zip(CLASS_NAMES, w)))


if __name__ == "__main__":
    main()
