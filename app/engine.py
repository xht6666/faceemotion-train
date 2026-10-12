"""推理引擎：模型发现、懒加载缓存、统一预处理与单图预测。

设计要点
--------
1. 模型清单来自 ``checkpoints/*.pt``，并用 ``reports/results/*.json`` 补充
   测试准确率、参数量、说明等元信息（缺 JSON 也能跑，只是没有指标）。
2. 模型**懒加载**：首次使用才读权重，最多同时保留 4 个在显存/内存，
   用全局锁串行化推理，避免 demo 场景下的并发显存抖动。
3. 预处理与训练/评估完全一致：``Resize(48x48) -> ToTensor -> Normalize(mean,std)``。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict

import torch
from PIL import Image
from torchvision import transforms

from src.models import build_model
from src.task import (CLASS_NAMES, CLASS_NAMES_ZH, NUM_CLASSES, PIXEL_MEAN,
                      PIXEL_STD, PROJECT_ROOT)

CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULT_DIR = os.path.join(PROJECT_ROOT, "reports", "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

MAX_CACHE = 4                       # 最多同时驻留的模型数
MAX_UPLOAD = 12 * 1024 * 1024       # 上传大小上限 12MB

PREPROCESS = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
])

# 展示名（用 reports 里的 notes 兜底）
DISPLAY_PREFIX = {
    "cnn": "轻量 CNN",
    "vgg": "VGGNet-16",
    "resnet18": "ResNet-18",
    "resnet32": "ResNet-32",
    "resnet_lite": "轻量化 ResNet",
    "resnet_lite_se": "轻量化 ResNet + SE",
}

_lock = threading.Lock()                       # 串行化 GPU 推理
_cache: "OrderedDict[str, torch.nn.Module]" = OrderedDict()


def _result_meta(name: str) -> dict:
    """读取训练结果 JSON，取准确率/参数量/说明。"""
    p = os.path.join(RESULT_DIR, f"{name}.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
    except Exception:
        return {}
    return {
        "test_accuracy": r.get("test_accuracy"),
        "test_macro_f1": r.get("test_macro_f1"),
        "params_M": r.get("params_M"),
        "notes": r.get("config", {}).get("notes", ""),
        "arch": r.get("model", ""),
        "seed": r.get("config", {}).get("seed"),
        "class_weight": r.get("config", {}).get("class_weight", False),
    }


def _display_name(name: str, meta: dict) -> str:
    arch = meta.get("arch") or name.rsplit("_seed", 1)[0]
    base = DISPLAY_PREFIX.get(arch, arch)
    if meta.get("notes"):
        return f"{base} · {meta['notes']}"
    if name.endswith("_seed123"):
        return f"{base} · seed=123"
    return base


def list_models() -> list[dict]:
    """扫描 checkpoints，返回可用于接口的模型清单（按准确率降序）。"""
    if not os.path.isdir(CKPT_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(CKPT_DIR)):
        if not fn.endswith(".pt"):
            continue
        name = fn[:-3]
        meta = _result_meta(name)
        out.append({
            "id": name,
            "display": _display_name(name, meta),
            "arch": meta.get("arch", name),
            "params_M": meta.get("params_M"),
            "test_accuracy": meta.get("test_accuracy"),
            "test_macro_f1": meta.get("test_macro_f1"),
            "notes": meta.get("notes", ""),
            "seed": meta.get("seed"),
        })
    out.sort(key=lambda x: -(x["test_accuracy"] or 0))
    return out


def list_classes() -> list[dict]:
    return [{"index": i, "name": n, "zh": z}
            for i, (n, z) in enumerate(zip(CLASS_NAMES, CLASS_NAMES_ZH))]


def _arch_of(name: str) -> str:
    meta = _result_meta(name)
    if meta.get("arch"):
        return meta["arch"]
    ck = os.path.join(CKPT_DIR, f"{name}.pt")
    try:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        return state.get("cfg", {}).get("model", name)
    except Exception:
        return name


def get_model(name: str) -> torch.nn.Module:
    """按名取模型（懒加载 + LRU，最多 MAX_CACHE 个）。"""
    with _lock:
        if name in _cache:
            _cache.move_to_end(name)
            return _cache[name]
        ck = os.path.join(CKPT_DIR, f"{name}.pt")
        if not os.path.exists(ck):
            raise FileNotFoundError(f"checkpoint 不存在: {name}")
        arch = _arch_of(name)
        model = build_model(arch, num_classes=NUM_CLASSES)
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.to(DEVICE).eval()
        _cache[name] = model
        while len(_cache) > MAX_CACHE:
            _cache.popitem(last=False)
        return model


def _fix_exif(img: Image.Image) -> Image.Image:
    from PIL import ImageOps

    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def load_image(data: bytes) -> Image.Image:
    """bytes -> RGB PIL 图（带 EXIF 方向校正）。"""
    import io

    img = Image.open(io.BytesIO(data))
    return _fix_exif(img).convert("RGB")


def preprocess(img: Image.Image) -> torch.Tensor:
    """PIL RGB 图 -> (1,3,48,48) 归一化张量。"""
    return PREPROCESS(img).unsqueeze(0)


def _predict_tensor(x: torch.Tensor, model_name: str) -> tuple[list[float], float]:
    model = get_model(model_name)
    with _lock, torch.inference_mode():
        t0 = time.perf_counter()
        x = x.to(DEVICE)
        logits = model(x)
        prob = torch.softmax(logits.float(), dim=1)[0]
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
    return prob.cpu().tolist(), dt


def predict_bytes(data: bytes, model_name: str = "resnet18") -> dict:
    """上传的图片字节 -> 预测结果字典。"""
    t_all = time.perf_counter()
    img = load_image(data)
    w, h = img.size
    probs, dt = _predict_tensor(preprocess(img), model_name)
    return _pack(probs, dt, model_name, (time.perf_counter() - t_all) * 1000,
                 {"width": w, "height": h})


def predict_pil(img: Image.Image, model_name: str = "resnet18") -> dict:
    probs, dt = _predict_tensor(preprocess(img.convert("RGB")), model_name)
    return _pack(probs, dt, model_name, None, {"width": img.size[0],
                                               "height": img.size[1]})


def _pack(probs: list[float], infer_ms: float, model_name: str,
          total_ms: float | None, extra: dict) -> dict:
    top = int(max(range(len(probs)), key=lambda i: probs[i]))
    items = [{"index": i, "name": CLASS_NAMES[i], "zh": CLASS_NAMES_ZH[i],
              "prob": round(float(p), 6)}
             for i, p in enumerate(probs)]
    out = {
        "ok": True,
        "model": model_name,
        "num_classes": NUM_CLASSES,
        "top": {"index": top, "name": CLASS_NAMES[top],
                "zh": CLASS_NAMES_ZH[top], "prob": round(float(probs[top]), 6)},
        "probs": items,
        "infer_ms": round(infer_ms, 2),
        "image": extra,
    }
    if total_ms is not None:
        out["total_ms"] = round(total_ms, 2)
    return out


def device_info() -> dict:
    info = {"device": str(DEVICE), "torch": torch.__version__}
    if DEVICE.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
