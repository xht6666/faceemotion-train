"""分类指标（不依赖 sklearn）：混淆矩阵、每类召回率、Macro-F1、平衡准确率。"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(tp), where=(prec + rec) > 0)
    return f1


def stats(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1 = per_class_f1(cm)
    support = cm.sum(axis=1)
    acc = float(np.trace(cm)) / max(cm.sum(), 1)
    balanced = float(np.mean(np.diag(cm) / np.maximum(support, 1)))
    return {
        "accuracy": acc,
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": balanced,
        "per_class_f1": f1.tolist(),
        "per_class_recall": (np.diag(cm) / np.maximum(support, 1)).tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }
