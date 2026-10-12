"""FER+ 6 分类情绪识别服务（FastAPI）。

启动（在项目根目录）::

    uvicorn app.main:app --host 127.0.0.1 --port 8000

然后浏览器打开 http://127.0.0.1:8000/ 使用演示页面，
接口文档见 http://127.0.0.1:8000/docs 与《接口说明.md》。
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import (FastAPI, File, Form, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app import engine

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SPLITS = ("FER2013Train", "FER2013Valid", "FER2013Test")

app = FastAPI(
    title="FER+ 6 分类表情识别接口",
    version="1.0.0",
    description=(
        "PyTorch 从零训练的 6 类表情识别服务：中性 / 快乐 / 惊讶 / 悲伤 / 愤怒 / 恐惧。\n\n"
        "- `POST /api/recognize` 单图识别（可选模型）\n"
        "- `WS /ws/video` 视频帧实时识别，逐帧返回情绪概率\n"
        "- `GET /api/models` 可选模型清单（含测试准确率）\n"
        "- 演示页面：`/`"
    ),
)


# ------------------------------------------------------------------ 静态页面
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/static/{path:path}", include_in_schema=False)
def static(path: str) -> FileResponse:
    f = os.path.join(STATIC_DIR, path)
    if not os.path.isfile(f):
        raise HTTPException(404, "not found")
    return FileResponse(f)


# ------------------------------------------------------------------ 元信息
@app.get("/api/health")
def health() -> dict:
    models = engine.list_models()
    return {"status": "ok", "models": len(models),
            "default_model": "resnet18" if any(m["id"] == "resnet18"
                                                for m in models)
            else (models[0]["id"] if models else None),
            **engine.device_info()}


@app.get("/api/classes")
def classes() -> dict:
    return {"num_classes": engine.NUM_CLASSES, "classes": engine.list_classes()}


@app.get("/api/models")
def models() -> dict:
    return {"default": "resnet18", "models": engine.list_models()}


# --------------------------------------------------------------- 随机样例图
@app.get("/api/sample/{split}")
def sample(split: str, seed: int | None = None) -> FileResponse:
    """从指定 split 里随机取一张测试图片，供演示页面"一键识别"。"""
    if split not in SPLITS:
        raise HTTPException(400, f"split 必须是 {SPLITS}")
    d = os.path.join(ROOT, split)
    files = [f for f in os.listdir(d) if f.endswith(".png")]
    if not files:
        raise HTTPException(404, "该 split 没有图片")
    rnd = random.Random(seed) if seed is not None else random
    f = rnd.choice(files)
    return FileResponse(os.path.join(d, f), media_type="image/png",
                        headers={"X-Sample-Name": f})


# --------------------------------------------------------------- 单图识别
@app.post("/api/recognize")
async def recognize(file: UploadFile = File(..., description="人脸/表情图片"),
                    model: str = Form("resnet18", description="模型 id，见 /api/models")):
    data = await file.read()
    if not data:
        raise HTTPException(400, "空文件")
    if len(data) > engine.MAX_UPLOAD:
        raise HTTPException(413, f"图片超过 {engine.MAX_UPLOAD // 1024 // 1024}MB")
    if not os.path.exists(os.path.join(engine.CKPT_DIR, f"{model}.pt")):
        raise HTTPException(404, f"模型不存在: {model}（可用清单见 /api/models）")
    try:
        result = await run_in_threadpool(engine.predict_bytes, data, model)
    except Exception as e:                       # 解码失败等
        raise HTTPException(400, f"图片无法解析: {e}")
    result["file"] = file.filename
    return JSONResponse(result)


# ------------------------------------------------------- WebSocket 视频实时识别
@app.websocket("/ws/video")
async def ws_video(ws: WebSocket) -> None:
    """协议：
    - 客户端发 **二进制** = 一帧图片（JPEG/PNG 字节）
    - 客户端发 **文本**   = 控制指令 JSON：``{"model": "resnet32"}``
    - 服务端回 **文本**   = 逐帧结果 JSON（见《接口说明.md》）
    """
    await ws.accept()
    model_name = "resnet18" if os.path.exists(
        os.path.join(engine.CKPT_DIR, "resnet18.pt")) else None
    seq = 0
    n_err = 0
    lat_sum = 0.0
    started = time.time()
    await ws.send_text(json.dumps({
        "ok": True, "type": "hello", "model": model_name,
        "device": str(engine.DEVICE), "classes": engine.list_classes(),
    }, ensure_ascii=False))

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            # ---- 控制指令（切换模型等） ----
            if msg.get("text"):
                try:
                    ctrl = json.loads(msg["text"])
                except Exception:
                    await ws.send_text(json.dumps(
                        {"ok": False, "type": "error", "error": "非 JSON 文本"}))
                    continue
                if "model" in ctrl:
                    mid = str(ctrl["model"])
                    if not os.path.exists(os.path.join(engine.CKPT_DIR, f"{mid}.pt")):
                        await ws.send_text(json.dumps(
                            {"ok": False, "type": "error",
                             "error": f"模型不存在: {mid}"}, ensure_ascii=False))
                    else:
                        model_name = mid
                        await ws.send_text(json.dumps(
                            {"ok": True, "type": "model", "model": model_name},
                            ensure_ascii=False))
                elif ctrl.get("type") == "ping":
                    await ws.send_text(json.dumps({"ok": True, "type": "pong"}))
                continue

            # ---- 图像帧 ----
            data = msg.get("bytes")
            if not data:
                continue
            seq += 1
            if model_name is None:
                await ws.send_text(json.dumps(
                    {"ok": False, "type": "error", "error": "没有可用模型",
                     "seq": seq}))
                continue
            try:
                result = await run_in_threadpool(engine.predict_bytes, data,
                                                 model_name)
            except Exception as e:
                n_err += 1
                await ws.send_text(json.dumps(
                    {"ok": False, "type": "error", "seq": seq,
                     "error": f"帧解析失败: {e}"}, ensure_ascii=False))
                continue
            lat_sum += result["infer_ms"]
            result.update({
                "type": "frame",
                "seq": seq,
                "model": model_name,
                "avg_infer_ms": round(lat_sum / max(seq - n_err, 1), 2),
                "errors": n_err,
                "uptime_s": round(time.time() - started, 1),
            })
            await ws.send_text(json.dumps(result, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
