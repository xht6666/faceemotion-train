"""接口自检：用 starlette TestClient 跑一遍 REST + WebSocket（不启端口）。

用法::

    python scripts/check_api.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

c = TestClient(app)

ok = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global ok
    print(f"[{'OK ' if cond else 'FAIL'}] {name} {extra}")
    ok = ok and cond


# ---------------- REST ----------------
r = c.get("/api/health")
j = r.json()
check("health", r.status_code == 200 and j["status"] == "ok",
      f"device={j.get('device')} models={j.get('models')}")

r = c.get("/api/models")
models = r.json()["models"]
check("models", r.status_code == 200 and len(models) > 0, f"n={len(models)}")

r = c.get("/api/classes")
check("classes", r.json()["num_classes"] == 6)

r = c.get("/")
check("index page", r.status_code == 200 and "情绪识别" in r.text,
      f"{len(r.text)} bytes")

# 随机样例 -> 单图识别（每个模型各测一次，取前 3 个 + 默认模型）
sample = c.get("/api/sample/FER2013Test", params={"seed": 7})
check("sample", sample.status_code == 200 and len(sample.content) > 100,
      f"{sample.headers.get('x-sample-name')} {len(sample.content)}B")

for mid in [m["id"] for m in models[:3]] + ["resnet18_cw"]:
    r = c.post("/api/recognize", files={"file": ("t.png", sample.content, "image/png")},
               data={"model": mid})
    j = r.json() if r.status_code == 200 else {}
    check(f"recognize/{mid}", r.status_code == 200 and j.get("ok")
          and len(j.get("probs", [])) == 6,
          f"top={j.get('top',{}).get('name')} "
          f"p={j.get('top',{}).get('prob')} {j.get('infer_ms')}ms")

# 非法输入
r = c.post("/api/recognize", files={"file": ("a.txt", b"hello", "text/plain")},
           data={"model": "resnet18"})
check("bad image -> 400", r.status_code == 400, str(r.status_code))
r = c.post("/api/recognize", files={"file": ("a.png", sample.content, "image/png")},
           data={"model": "nope"})
check("bad model -> 404", r.status_code == 404, str(r.status_code))

# ---------------- WebSocket ----------------
with c.websocket_connect("/ws/video") as ws:
    hello = json.loads(ws.receive_text())
    check("ws hello", hello.get("ok") and hello.get("type") == "hello",
          f"model={hello.get('model')} device={hello.get('device')}")

    # 控制：切模型
    ws.send_text(json.dumps({"model": "resnet32"}))
    ans = json.loads(ws.receive_text())
    check("ws switch model", ans.get("ok") and ans.get("model") == "resnet32")

    ws.send_text(json.dumps({"type": "ping"}))
    check("ws ping", json.loads(ws.receive_text()).get("type") == "pong")

    # 连发 3 帧（同一张图）
    for i in range(3):
        ws.send_bytes(sample.content)
        j = json.loads(ws.receive_text())
        check(f"ws frame#{i}", j.get("ok") and j.get("type") == "frame"
              and len(j.get("probs", [])) == 6,
              f"seq={j.get('seq')} top={j.get('top',{}).get('name')} "
              f"{j.get('infer_ms')}ms")

    # 非法帧
    ws.send_bytes(b"not-an-image")
    j = json.loads(ws.receive_text())
    check("ws bad frame -> ok=false", j.get("ok") is False, str(j.get("error"))[:60])

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
