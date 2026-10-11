"""FastAPI backend for the localhost dashboard.

- Serves `static/` (dashboard UI).
- MJPEG video at /video (server-side overlays, latest frame only).
- WebSocket /ws pushing state snapshots ~10 Hz with reconnect support.
- Validated REST API; allowlisted settings/actions only — no arbitrary
  execution endpoint exists by design.

Binds 127.0.0.1 only (see __main__). Run:
    ./myenv/bin/python webapp.py   (or: ./myenv/bin/uvicorn webapp:app)
"""

import asyncio
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from webcam import CameraManager, SETTINGS_FIELDS

app = FastAPI(title="finger-pos AI camera")
manager = CameraManager()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


@app.get("/api/status")
async def api_status():
    snap = manager.snapshot()
    snap["backend"] = {"ok": True, "time": round(time.time(), 3)}
    return snap


class CameraStart(BaseModel):
    camera: int = Field(default=0, ge=0, le=64)


@app.post("/api/camera/start")
async def camera_start(body: CameraStart):
    ok, msg = manager.start(camera=body.camera)
    if not manager.running and manager.last_error and not ok:
        raise HTTPException(500, manager.last_error or msg)
    # The loop may still be spinning up or failing on first read; report
    # honestly — the status endpoint carries running/error afterwards.
    return {"ok": manager.running or ok, "msg": msg,
            "error": manager.last_error}


@app.post("/api/camera/stop")
async def camera_stop():
    ok, msg = manager.stop()
    return {"ok": ok, "msg": msg}


@app.post("/api/gestures/enable")
async def gestures_enable():
    manager.engine.enabled = True
    manager.log("CONTROL ON (web)")
    return {"ok": True, "enabled": True}


@app.post("/api/gestures/disable")
async def gestures_disable():
    manager.engine.enabled = False
    manager.engine.volume_mode = False
    manager.log("CONTROL OFF (web)")
    return {"ok": True, "enabled": False}


class SettingsPatch(BaseModel):
    model_config = {"extra": "forbid"}
    smoothing: float | None = Field(default=None, ge=0.0, le=1.0)
    debounce: int | None = Field(default=None, ge=1, le=30)
    strength: int | None = Field(default=None, ge=1, le=10)
    blur_kind: str | None = None
    privacy_mode: str | None = None
    vol_top: float | None = Field(default=None, ge=0.0, le=1.0)
    vol_bottom: float | None = Field(default=None, ge=0.0, le=1.0)
    sensitivity: str | None = None
    debug: bool | None = None


@app.get("/api/settings")
async def settings_get():
    return dict(manager.settings)


@app.put("/api/settings")
async def settings_put(patch: SettingsPatch):
    from privacy import BLUR_KINDS, MODES
    data = {k: v for k, v in patch.model_dump().items() if v is not None}
    if "blur_kind" in data and data["blur_kind"] not in BLUR_KINDS:
        raise HTTPException(422, f"blur_kind must be one of {list(BLUR_KINDS)}")
    if "privacy_mode" in data and data["privacy_mode"] not in MODES:
        raise HTTPException(422, f"privacy_mode must be one of {list(MODES)}")
    if "sensitivity" in data and data["sensitivity"] not in ("low", "normal", "high"):
        raise HTTPException(422, "sensitivity must be low/normal/high")
    if ("vol_top" in data and "vol_bottom" in data
            and not data["vol_top"] < data["vol_bottom"]):
        raise HTTPException(422, "vol_top must be < vol_bottom")
    return manager.apply_settings(data)


@app.post("/api/media/{op}")
async def media_op(op: str):
    if op == "play_pause":
        ok, info = manager.player.play_pause()
    elif op == "next":
        ok, info = manager.player.next()
    elif op == "previous":
        ok, info = manager.player.previous()
    else:
        raise HTTPException(404, "unknown media op")
    ok2, np_ = manager.player.now_playing()
    if ok2:
        manager.ctrl["now_playing"] = np_
    manager.log(f"{op} -> {'ok' if ok else 'failed'}")
    return {"ok": ok, "info": info,
            "now_playing": manager.ctrl["now_playing"]}


class VolumeSet(BaseModel):
    pct: int = Field(ge=0, le=100)


@app.post("/api/volume/set")
async def volume_set(body: VolumeSet):
    ok, info = manager.vol.set_absolute(body.pct)
    if ok:
        manager.ctrl["volume_pct"] = body.pct
    manager.log(f"volume set {body.pct}% -> {'ok' if ok else 'failed'}")
    return {"ok": ok, "info": info, "volume_pct": manager.ctrl["volume_pct"]}


@app.post("/api/volume/{op}")
async def volume_op(op: str):
    if op == "up":
        ok, info = manager.vol.up()
    elif op == "down":
        ok, info = manager.vol.down()
    elif op == "mute":
        ok, info = manager.vol.mute_toggle()
    else:
        raise HTTPException(404, "unknown volume op")
    ok2, info2 = manager.vol.percent()
    if ok2:
        manager.ctrl["volume_pct"], manager.ctrl["muted"] = info2
    manager.log(f"volume {op} -> {'ok' if ok else 'failed'}")
    return {"ok": ok, "info": info, "volume_pct": manager.ctrl["volume_pct"],
            "muted": manager.ctrl["muted"]}


class PrivacyMode(BaseModel):
    mode: str


@app.post("/api/privacy/mode")
async def privacy_mode(body: PrivacyMode):
    from privacy import MODES
    if body.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {list(MODES)}")
    manager.privacy.mode = body.mode
    manager.log(f"privacy -> {body.mode}")
    return {"ok": True, "mode": body.mode}


@app.get("/video")
async def video():
    if not manager.running:
        return JSONResponse({"ok": False, "error": "camera not running"},
                            status_code=503)

    async def frames():
        boundary = b"--frame"
        while True:
            with manager._lock:
                jpg = manager.jpg
                running = manager.running
            if not running:
                break
            if jpg is None:
                await asyncio.sleep(0.05)
                continue
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
                   + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                   + jpg + b"\r\n")
            await asyncio.sleep(0.033)
    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws_feed(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(manager.snapshot())
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
