"""Tests for the localhost web backend — no camera/display needed.

Uses httpx-backed TestClient in-process. Camera start is tested against
a nonexistent device to prove graceful failure, never a crash.
"""

import pytest
from fastapi.testclient import TestClient

import webapp
from webapp import app, manager

client = TestClient(app)


def test_index_serves_dashboard():
    r = client.get("/")
    assert r.status_code == 200
    assert "FINGER-POS" in r.text
    assert "/static/app.js" in r.text


def test_status_shape_camera_off():
    manager.stop()
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("camera", "hands", "gesture", "faces", "privacy",
                "control", "analytics", "events", "backend"):
        assert key in body
    assert body["camera"]["running"] is False
    assert body["gesture"]["gesture"] == "NONE"


def test_video_503_when_camera_off():
    manager.stop()
    r = client.get("/video")
    assert r.status_code == 503


def test_camera_start_bad_device_graceful():
    r = client.post("/api/camera/start", json={"camera": 60})
    assert r.status_code == 200
    body = r.json()
    assert set(("ok", "msg", "error")) <= set(body)
    import time
    time.sleep(1.5)  # let the loop attempt and fail
    st = client.get("/api/status").json()
    assert st["camera"]["running"] is False  # failed cleanly, no crash
    r = client.post("/api/camera/stop")
    assert r.json()["ok"] is True


def test_camera_start_validation():
    r = client.post("/api/camera/start", json={"camera": -1})
    assert r.status_code == 422


def test_gestures_enable_disable():
    assert client.post("/api/gestures/enable").json() == {"ok": True, "enabled": True}
    assert manager.engine.enabled is True
    assert client.post("/api/gestures/disable").json() == {"ok": True, "enabled": False}
    assert manager.engine.enabled is False


def test_settings_get_put_roundtrip():
    assert client.get("/api/settings").json()["smoothing"] == pytest.approx(0.2)
    r = client.put("/api/settings", json={"smoothing": 0.7, "debounce": 5})
    assert r.status_code == 200
    assert r.json()["smoothing"] == pytest.approx(0.7)
    assert client.put("/api/settings", json={"smoothing": 0.2, "debounce": 3}).status_code == 200


def test_settings_reject_unknown_key():
    r = client.put("/api/settings", json={"exec": "rm -rf /"})
    assert r.status_code == 422


def test_settings_reject_bad_values():
    assert client.put("/api/settings", json={"blur_kind": "NOPE"}).status_code == 422
    assert client.put("/api/settings", json={"privacy_mode": "NOPE"}).status_code == 422
    assert client.put("/api/settings", json={"sensitivity": "ultra"}).status_code == 422
    assert client.put("/api/settings",
                      json={"vol_top": 0.8, "vol_bottom": 0.2}).status_code == 422
    assert client.put("/api/settings", json={"strength": 99}).status_code == 422


def test_no_arbitrary_exec_endpoint():
    assert client.get("/api/exec").status_code == 404
    assert client.post("/api/exec", json={"cmd": "id"}).status_code == 404


def test_media_ops_shape_and_404():
    r = client.post("/api/media/next")
    assert r.status_code == 200
    assert set(("ok", "info", "now_playing")) <= set(r.json())
    assert client.post("/api/media/bogus").status_code == 404


def test_volume_ops_shape_and_validation():
    r = client.post("/api/volume/mute")
    assert r.status_code == 200
    assert set(("ok", "info", "volume_pct", "muted")) <= set(r.json())
    assert client.post("/api/volume/bogus").status_code == 404
    assert client.post("/api/volume/set", json={"pct": 150}).status_code == 422
    r = client.post("/api/volume/set", json={"pct": 42})
    assert r.status_code == 200 and set(("ok", "info", "volume_pct")) <= set(r.json())


def test_privacy_mode_validation():
    assert client.post("/api/privacy/mode", json={"mode": "NOPE"}).status_code == 422
    r = client.post("/api/privacy/mode", json={"mode": "AUTO"})
    assert r.json() == {"ok": True, "mode": "AUTO"}
    client.post("/api/privacy/mode", json={"mode": "OFF"})


def test_websocket_snapshot():
    with client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
    assert set(("camera", "gesture", "control", "events")) <= set(data)
