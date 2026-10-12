"""汇总全部实验结果，生成 reports/comparison.{csv,md} 与类别级对比表。"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.task import REPORT_DIR, RESULT_DIR, SUFFIX, task_title  # noqa: E402

EXCLUDE = {"timing_test"}


def to_markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """不依赖 tabulate 的 Markdown 表格生成。"""

    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def converge_epoch(history: list[dict], target: float) -> int | None:
    for h in history:
        if h["val_acc"] >= target:
            return h["epoch"]
    return None


def load_results() -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(RESULT_DIR, "*.json"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name in EXCLUDE or name.startswith("_"):
            continue
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def summary_rows(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        hist = r["history"]
        best = r["convergence"]["best_val_acc"]
        rows.append({
            "实验": r["exp_name"],
            "模型": r["model"],
            "说明": r["config"].get("notes", ""),
            "优化器": r["config"]["optimizer"],
            "调度": r["config"]["scheduler"],
            "参数量(M)": r["params_M"],
            "测试准确率": r["test_accuracy"],
            "测试MacroF1": r["test_macro_f1"],
            "平衡准确率": r["test_balanced_accuracy"],
            "最佳验证准确率": best,
            "最佳轮次": r["convergence"]["best_epoch"],
            "收敛轮次(best-1pp)": converge_epoch(hist, best - 0.01),
            "达75%验证轮次": r["convergence"]["epochs_to_val75"],
            "最佳轮泛化差距": round(r["generalization_gap_at_best"], 4),
            "训练总时长(s)": r["total_train_time_s"],
        })
    return rows


def per_class_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"实验": r["exp_name"]}
        for c in r["class_names"]:
            rec = r.get("test_per_class_recall", {}).get(c)
            f1 = r.get("test_per_class_f1", {}).get(c)
            row[f"{c} 召回"] = round(rec, 4) if rec is not None else None
            row[f"{c} F1"] = round(f1, 4) if f1 is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_summary(results: list[dict]) -> pd.DataFrame:
    """按（模型, 优化器, 调度, 是否增强, 类别权重）分组，对多种子结果做 mean/std/min/max。"""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        c = r["config"]
        key = (c["model"], c["optimizer"], c["scheduler"], c["jitter"],
               c.get("class_weight", False))
        groups.setdefault(key, []).append(r)

    rows = []
    for (model, opt, sched, jitter, cw), rs in groups.items():
        accs = [r["test_accuracy"] for r in rs]
        vals = [r["best_val_acc"] for r in rs]
        n = len(accs)
        note = next((r["config"].get("notes", "") for r in rs
                     if "seed=" not in r["config"].get("notes", "")),
                    rs[0]["config"].get("notes", ""))
        rows.append({
            "模型": model,
            "说明": note,
            "优化器": opt,
            "调度": sched,
            "增强": "是" if jitter else "否",
            "类别权重": "是" if cw else "否",
            "参数量(M)": rs[0]["params_M"],
            "种子数": n,
            "测试准确率mean": float(np.mean(accs)),
            "测试准确率std": float(np.std(accs, ddof=1)) if n > 1 else 0.0,
            "min": float(np.min(accs)),
            "max": float(np.max(accs)),
            "最佳验证mean": float(np.mean(vals)),
        })
    df = pd.DataFrame(rows).sort_values("测试准确率mean", ascending=False)
    return df


def main() -> None:
    results = load_results()
    if not results:
        raise SystemExit("没有找到结果 JSON")

    df = pd.DataFrame(summary_rows(results)).sort_values("测试准确率", ascending=False)
    csv_path = os.path.join(REPORT_DIR, f"comparison{SUFFIX}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")

    pc = per_class_table(results)
    pc_path = os.path.join(REPORT_DIR, f"per_class{SUFFIX}.csv")
    pc.to_csv(pc_path, index=False, encoding="utf-8-sig", float_format="%.4f")

    md = [f"# 实验对比总表（{task_title()}）\n",
          to_markdown_table(df),
          "\n## 各类别召回率 / F1（测试集）\n",
          to_markdown_table(pc)]
    md_path = os.path.join(REPORT_DIR, f"comparison{SUFFIX}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # 分组（多种子）汇总
    gdf = grouped_summary(results)
    gcsv = os.path.join(REPORT_DIR, f"summary_by_config{SUFFIX}.csv")
    gdf.to_csv(gcsv, index=False, encoding="utf-8-sig", float_format="%.4f")
    with open(os.path.join(REPORT_DIR, f"summary_by_config{SUFFIX}.md"),
              "w", encoding="utf-8") as f:
        f.write(f"# 按配置分组（多种子）汇总 · {task_title()}\n\n" + to_markdown_table(gdf))
    print("\n--- 分组汇总（多种子）---")
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(gdf.to_string(index=False))
    print("csv ->", gcsv)

    try:
        from src.plots import plot_accuracy_params
        print("plot ->", plot_accuracy_params(results))
    except Exception as e:          # 绘图失败不影响表格输出
        print("plot skipped:", e)

    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(df.to_string(index=False))
    print("\ncsv ->", csv_path)
    print("csv ->", pc_path)
    print("md  ->", md_path)


if __name__ == "__main__":
    main()
