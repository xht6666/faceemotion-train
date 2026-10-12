"""统一训练器：优化器/学习率调度可配置，记录收敛过程与评估指标。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import metrics as M
from .datasets import (CLASS_NAMES, CLASS_NAMES_ZH, NUM_CLASSES, class_counts,
                       make_loaders, seed_everything)
from .models import build_model, count_params
from .task import (CKPT_DIR, CURVE_DIR, DATA_ROOT, PROJECT_ROOT, RESULT_DIR,
                   TASK)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")


@dataclass
class TrainConfig:
    exp_name: str
    model: str = "resnet18"
    epochs: int = 60
    batch_size: int = 128
    # 优化器
    optimizer: str = "adam"          # adam | sgd
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    # 学习率调度
    scheduler: str = "cosine"        # cosine | steplr | none
    step_size: int = 20
    gamma: float = 0.1
    min_lr: float = 1e-5
    # 其他
    label_smoothing: float = 0.0
    class_weight: bool = False
    seed: int = 42
    amp: bool = True
    num_workers: int = 8
    jitter: bool = True              # 是否使用数据增强
    notes: str = ""
    extra: dict = field(default_factory=dict)


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                               momentum=cfg.momentum, nesterov=True,
                               weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")


def build_scheduler(cfg: TrainConfig, opt: torch.optim.Optimizer):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=cfg.min_lr)
    if cfg.scheduler == "steplr":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size,
                                               gamma=cfg.gamma)
    if cfg.scheduler == "none":
        return None
    raise ValueError(f"unknown scheduler: {cfg.scheduler}")


class Trainer:
    def __init__(self, cfg: TrainConfig, root: str = DATA_ROOT):
        self.cfg = cfg
        self.root = root
        seed_everything(cfg.seed)

        self.train_loader, self.valid_loader, self.test_loader = make_loaders(
            root, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            jitter=cfg.jitter)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(cfg.model, num_classes=NUM_CLASSES).to(self.device)
        self.total_params, self.trainable_params = count_params(self.model)

        if cfg.class_weight:
            counts = class_counts(root)
            w = counts.sum() / (len(counts) * counts.astype(np.float64))
            class_weights = torch.tensor(w, dtype=torch.float32,
                                         device=self.device)
        else:
            class_weights = None
        self.class_weights = class_weights
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=cfg.label_smoothing,
            reduction="mean")

        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.history: list[dict] = []
        self.best = {"val_acc": -1.0, "epoch": -1}
        self.test_stats: dict | None = None
        os.makedirs(RESULT_DIR, exist_ok=True)
        os.makedirs(CKPT_DIR, exist_ok=True)
        os.makedirs(CURVE_DIR, exist_ok=True)

    # ---------------- 训练 ----------------
    def train_one_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        return total_loss / max(seen, 1), correct / max(seen, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> dict:
        loader = loader or self.valid_loader
        self.model.eval()
        ys, ps = [], []
        total_loss, seen = 0.0, 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            seen += y.size(0)
            ys.append(y.cpu().numpy())
            ps.append(logits.float().cpu().numpy().argmax(1))
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        out = M.stats(y_true, y_pred, NUM_CLASSES)
        out["loss"] = total_loss / max(seen, 1)
        return out

    def fit(self) -> dict:
        cfg = self.cfg
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self.train_one_epoch()
            val = self.evaluate(self.valid_loader)
            if self.scheduler is not None:
                self.scheduler.step()
            lr_now = self.optimizer.param_groups[0]["lr"]
            rec = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val["loss"],
                "val_acc": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
                "val_balanced_acc": val["balanced_accuracy"],
                "lr": lr_now,
                "time_s": round(time.time() - t0, 2),
                "generalization_gap": tr_acc - val["accuracy"],
            }
            self.history.append(rec)
            if val["accuracy"] > self.best["val_acc"]:
                self.best = {"val_acc": val["accuracy"], "epoch": epoch}
                torch.save({"model": self.model.state_dict(), "cfg": asdict(cfg),
                            "epoch": epoch, "val_acc": val["accuracy"]},
                           os.path.join(CKPT_DIR, f"{cfg.exp_name}.pt"))
            if epoch % 5 == 0 or epoch == 1 or epoch == cfg.epochs:
                print(f"[{cfg.exp_name}] ep{epoch:03d} "
                      f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                      f"val_loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
                      f"f1 {val['macro_f1']:.4f} | lr {lr_now:.2e} "
                      f"| {rec['time_s']:.1f}s", flush=True)
        self.test_stats = self._final_test()
        result = self.build_result()
        self._save(result)
        return result

    # ---------------- 结果汇总 ----------------
    @torch.no_grad()
    def _final_test(self) -> dict:
        ckpt = os.path.join(CKPT_DIR, f"{self.cfg.exp_name}.pt")
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        return self.evaluate(self.test_loader)

    def _convergence(self) -> dict:
        accs = [h["val_acc"] for h in self.history]
        best = max(accs)

        def first_reach(target: float):
            for i, a in enumerate(accs, start=1):
                if a >= target:
                    return i
            return None

        return {
            "best_epoch": int(np.argmax(accs)) + 1,
            "best_val_acc": best,
            "final_val_acc": accs[-1],
            "epochs_to_val70": first_reach(0.70),
            "epochs_to_val75": first_reach(0.75),
            "epochs_to_95pct_of_best": first_reach(best - (1 - 0.95) * best),
            "mean_last5_val_acc": float(np.mean(accs[-5:])),
        }

    def build_result(self) -> dict:
        cfg = self.cfg
        conv = self._convergence()
        best_ep = conv["best_epoch"]
        gap_at_best = self.history[best_ep - 1]["generalization_gap"]
        test = self.test_stats or {}
        return {
            "exp_name": cfg.exp_name,
            "model": cfg.model,
            "task": TASK,
            "config": asdict(cfg),
            "params_total": self.total_params,
            "params_trainable": self.trainable_params,
            "params_M": round(self.total_params / 1e6, 3),
            "best_val_acc": conv["best_val_acc"],
            "test_accuracy": test.get("accuracy"),
            "test_macro_f1": test.get("macro_f1"),
            "test_balanced_accuracy": test.get("balanced_accuracy"),
            "test_per_class_recall": dict(zip(CLASS_NAMES,
                                              test.get("per_class_recall", []))),
            "test_per_class_f1": dict(zip(CLASS_NAMES, test.get("per_class_f1", []))),
            "confusion_matrix": test.get("confusion_matrix"),
            "convergence": conv,
            "generalization_gap_at_best": gap_at_best,
            "train_acc_at_best": self.history[best_ep - 1]["train_acc"],
            "total_train_time_s": round(sum(h["time_s"] for h in self.history), 1),
            "history": self.history,
            "class_names": CLASS_NAMES,
            "class_names_zh": CLASS_NAMES_ZH,
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        }

    def _save(self, result: dict) -> None:
        path = os.path.join(RESULT_DIR, f"{self.cfg.exp_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[{self.cfg.exp_name}] saved -> {path}", flush=True)


def load_result(exp_name: str) -> dict:
    with open(os.path.join(RESULT_DIR, f"{exp_name}.json"), encoding="utf-8") as f:
        return json.load(f)
