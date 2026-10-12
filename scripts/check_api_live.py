"""临时：对已启动的线上服务做一次真实 HTTP + WebSocket 验证。"""
import asyncio
import json
import pathlib
import sys

import httpx
import websockets

BASE = "http://127.0.0.1:8000"
IMG = pathlib.Path(r"D:\facefinal\FER2013Test\fer0033970.png").read_bytes()


def main() -> None:
    r = httpx.get(f"{BASE}/api/sample/FER2013Test", params={"seed": 7}, timeout=30)
    img = r.content
    print("sample:", r.status_code, len(img), "bytes",
          r.headers.get("x-sample-name"))

    for mid in ("resnet18", "vgg", "resnet32_plain", "resnet18_cw"):
        resp = httpx.post(f"{BASE}/api/recognize",
                          files={"file": ("t.png", img, "image/png")},
                          data={"model": mid}, timeout=60)
        j = resp.json()
        top3 = sorted(j["probs"], key=lambda p: -p["prob"])[:3]
        print(f"recognize {mid:20s} {resp.status_code} top={j['top']['zh']} "
              f"{j['top']['prob']:.3f} infer={j['infer_ms']}ms "
              f"total={j.get('total_ms')}ms "
              + " | ".join(f"{p['zh']}:{p['prob']:.2f}" for p in top3))

    r = httpx.get(f"{BASE}/", timeout=30)
    print("page:", r.status_code, len(r.text), "bytes",
          "情绪识别" in r.text)


async def ws_test() -> None:
    async with websockets.connect("ws://127.0.0.1:8000/ws/video") as ws:
        hello = json.loads(await ws.recv())
        print("hello:", hello["model"], hello["device"],
              len(hello["classes"]), "classes")
        await ws.send(json.dumps({"model": "resnet_lite_se"}))
        print("switch:", await ws.recv())
        for i in range(5):
            await ws.send(IMG)
            j = json.loads(await ws.recv())
            print(f"frame seq={j['seq']} top={j['top']['zh']} "
                  f"{j['top']['prob']:.3f} infer={j['infer_ms']}ms "
                  f"avg={j['avg_infer_ms']}ms model={j['model']}")
        await ws.send(json.dumps({"type": "ping"}))
        print("ping:", await ws.recv())
        await ws.send(b"\x89PNG not a real image")
        print("bad :", json.loads(await ws.recv())["error"][:50])


if __name__ == "__main__":
    main()
    asyncio.run(ws_test())
    print("LIVE_OK")
    sys.exit(0)
