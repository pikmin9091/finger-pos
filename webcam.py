"""Localhost web dashboard backend (FastAPI).

Serves the futuristic dashboard (`static/`), an MJPEG camera stream with
server-side overlays, a WebSocket state feed, and a validated REST API.
Everything runs on 127.0.0.1 only — no cloud, no uploads, no recordings.

Reuse, not rewrite: the camera thread runs the exact same pipeline as
main.py (detect → slots → privacy → analytics → control → actions).
"""

import json
import threading
import time
from collections import deque

import cv2

SETTINGS_FIELDS = ("smoothing", "debounce", "strength", "blur_kind",
                    "privacy_mode", "vol_top", "vol_bottom", "sensitivity")


class CameraManager:
    """Owns the single camera thread. Idempotent start/stop, bounded
    buffers (latest JPEG frame only), lock-protected state."""

    def __init__(self):
        from multi import HandSlots, FaceSlots
        from privacy import PrivacyController, PrivacyRules
        from analytics import AnalyticsSession
        from control import ControlEngine
        from actions import MediaPlayer, Volume, Hyprland

        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self.running = False
        self.last_error = ""
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.jpg = None  # latest annotated JPEG bytes (bounded: 1 frame)
        self.frame_idx = 0

        self.settings = {"smoothing": 0.2, "debounce": 3, "strength": 5,
                         "blur_kind": "GAUSSIAN", "privacy_mode": "OFF",
                         "vol_top": 0.25, "vol_bottom": 0.75,
                         "sensitivity": "normal", "debug": False,
                         "camera": 0}
        self.slots = HandSlots()
        self.face_slots = FaceSlots()
        self.privacy = PrivacyController(
            rules=PrivacyRules(blur_kind="GAUSSIAN", strength=5))
        self.session = AnalyticsSession()
        self.engine = ControlEngine(min_confidence=0.8, dwell=0.7)
        self.player, self.vol, self.hypr = MediaPlayer(), Volume(), Hyprland()
        self.ctrl = {"last_action": None, "notif": "", "notif_until": 0.0,
                     "now_playing": "", "volume_pct": None, "muted": None,
                     "status_at": 0.0}
        self.last_summary = {"hands": 0, "faces": 0,
                             "gesture": {"gesture": "NONE", "confidence": 0.0,
                                         "state": "UNKNOWN"}}
        self.events = deque(maxlen=50)  # {"t","text"} for the WS log
        self._ev_id = 0

    # -- lifecycle ----------------------------------------------------
    def start(self, camera=None):
        with self._lock:
            if self.running:
                return True, "already running"
            if camera is not None:
                self.settings["camera"] = int(camera)
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True, "starting"

    def stop(self):
        with self._lock:
            thread = self._thread
        if thread is None:
            return True, "already stopped"
        self._stop.set()
        thread.join(timeout=5.0)
        with self._lock:
            self._thread = None
            self.running = False
        return True, "stopped"

    def log(self, text):
        self._ev_id += 1
        self.events.append({"id": self._ev_id, "t": round(time.time(), 3),
                            "text": str(text)[:160]})

    # -- settings (allowlisted, validated by the API layer) -----------
    def apply_settings(self, patch):
        s = self.settings
        if "smoothing" in patch:
            v = max(0.0, min(1.0, float(patch["smoothing"])))
            s["smoothing"] = v
            self.slots.set_smoothing(v)
        if "debounce" in patch:
            v = max(1, int(patch["debounce"]))
            s["debounce"] = v
            self.slots.set_debounce(v)
        if "strength" in patch:
            v = max(1, min(10, int(patch["strength"])))
            s["strength"] = v
            self.privacy.rules.strength = v
        if "blur_kind" in patch:
            from privacy import BLUR_KINDS
            if patch["blur_kind"] in BLUR_KINDS:
                s["blur_kind"] = patch["blur_kind"]
                self.privacy.rules.blur_kind = patch["blur_kind"]
        if "privacy_mode" in patch:
            from privacy import MODES
            if patch["privacy_mode"] in MODES:
                s["privacy_mode"] = patch["privacy_mode"]
                self.privacy.mode = patch["privacy_mode"]
        if "vol_top" in patch:
            self.engine.vol_top = float(patch["vol_top"])
            s["vol_top"] = self.engine.vol_top
        if "vol_bottom" in patch:
            self.engine.vol_bottom = float(patch["vol_bottom"])
            s["vol_bottom"] = self.engine.vol_bottom
        if "sensitivity" in patch:
            from detect import set_confidence as set_hand_conf
            from face import set_confidence as set_face_conf
            c = {"low": 0.65, "normal": 0.5, "high": 0.35}[patch["sensitivity"]]
            set_hand_conf(c, c, c)
            set_face_conf(c, c, c)
            s["sensitivity"] = patch["sensitivity"]
        if "debug" in patch:
            s["debug"] = bool(patch["debug"])
        return dict(s)

    # -- main loop (same pipeline as main.py) --------------------------
    def _loop(self):
        from capture import Camera
        from detect import detect_hands
        from face import detect_faces
        from blur import blur_faces
        from positions import (draw_landmarks, draw_gesture, draw_status,
                               draw_face_text, draw_blur_status,
                               draw_box, landmarks_box)
        try:
            cam = Camera(self.settings["camera"])
        except Exception as e:
            self.last_error = f"camera open failed: {e}"
            return
        self.running = True
        self.last_error = ""
        fps = 0.0
        try:
            while not self._stop.is_set():
                ret, frame = cam.read()
                if not ret:
                    self.last_error = "camera unavailable"
                    break
                timestamp = time.time()
                infer_t0 = time.time()
                try:
                    detected = detect_hands(frame)["hands"]
                except Exception as e:
                    self.last_error = f"hand model failed: {e}"
                    break
                hands = self.slots.update(detected, timestamp=timestamp)
                faces = []
                try:
                    faces = self.face_slots.update(
                        detect_faces(frame)["faces"])
                except FileNotFoundError as e:
                    self.last_error = f"face model missing: {e}"
                    faces = []
                except Exception:
                    faces = []
                infer_ms = (time.time() - infer_t0) * 1000.0

                pinch_now = any(
                    h["gesture"]["gesture"] == "PINCH"
                    and h["gesture"]["stable"] for h in hands)
                if self.engine.enabled:
                    primary = hands[0] if hands else None
                    if primary is not None:
                        g = primary["gesture"]
                        res = self.engine.update(
                            g["gesture"], g["confidence"],
                            primary["movement"]["direction"], timestamp,
                            hand_y=primary["hand_center"]["y"],
                            hand_present=True, gesture_stable=g["stable"])
                        if res is not None:
                            self._fire(res, timestamp)
                else:
                    if hands:
                        g = hands[0]["gesture"]
                        self.engine.update(
                            g["gesture"], g["confidence"],
                            hands[0]["movement"]["direction"], timestamp,
                            hand_present=True, gesture_stable=g["stable"])
                    self.privacy.update_pinch(pinch_now, timestamp)

                priv = self.privacy.evaluate(faces, timestamp)
                self.session.update(hands, faces, infer_ms, fps, timestamp)

                overlay = frame.copy()
                if detected:
                    for i, h in enumerate(detected):
                        overlay = draw_landmarks(overlay, h["landmarks"])
                        overlay = draw_gesture(overlay, hands[i]["gesture"],
                                               y=30 + i * 30)
                else:
                    overlay = draw_status(overlay, "NO HAND")
                n_blurred = 0
                if priv["boxes"]:
                    n_blurred = blur_faces(
                        overlay, priv["boxes"], expand=priv["expand"],
                        kind=priv["kind"], strength=priv["strength"])
                for i, h in enumerate(detected):
                    g = hands[i]["gesture"]["gesture"]
                    overlay = draw_box(overlay, landmarks_box(h["landmarks"]),
                                       color=(0, 255, 0),
                                       label=f"{hands[i]['hand']} {g}")
                for face in faces:
                    overlay = draw_box(overlay, face["box"], color=(255, 0, 0),
                                       label=f"FACE {face.get('looking', '')}")
                overlay = draw_blur_status(overlay, priv["active"], n_blurred,
                                           y=120)
                if self.engine is not None:
                    cv2.putText(
                        overlay,
                        f"CONTROL: {'ON' if self.engine.enabled else 'OFF'}",
                        (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if self.engine.enabled else (150, 150, 150),
                        2)
                ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
                h, w = frame.shape[:2]
                g0 = hands[0]["gesture"] if hands else None
                with self._lock:
                    self.jpg = bytes(buf) if ok else None
                    self.width, self.height = w, h
                    self.frame_idx += 1
                    self.last_summary = {
                        "hands": len(hands), "faces": len(faces),
                        "gesture": {"gesture": g0["gesture"],
                                    "confidence": g0["confidence"],
                                    "state": g0.get("state", "?")}
                        if g0 else {"gesture": "NONE", "confidence": 0.0,
                                    "state": "UNKNOWN"},
                    }
                frame_time = time.time() - timestamp
                inst = 1.0 / max(frame_time, 1e-6)
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
                self.fps = fps
        finally:
            try:
                cam.release()
            except Exception:
                pass
            with self._lock:
                self.running = False

    # -- gesture actions (same allowlist mapping as main._fire) --------
    def _fire(self, action, timestamp):
        engine, ctrl = self.engine, self.ctrl
        if isinstance(action, tuple):
            kind, val = action
            if kind == "volume_mode":
                self.log(f"VOLUME MODE {'ON' if val else 'OFF (locked)'}")
            elif kind == "volume_set":
                try:
                    ok, _ = self.vol.set_absolute(val)
                except Exception as e:
                    ok = False
                    self.log(f"volume failed: {e}")
                if ok:
                    ctrl["volume_pct"] = val
                    self.log(f"VOLUME {val}%")
            ctrl["last_action"] = action[0]
            return
        label = engine.label(action)
        ctrl["last_action"] = action
        if action == "control_toggle":
            self.log(f"CONTROL {'ON' if engine.enabled else 'OFF'}")
            return
        try:
            if action == "media_play_pause":
                ok, _ = self.player.play_pause()
            elif action == "media_next":
                ok, _ = self.player.next()
            elif action == "media_previous":
                ok, _ = self.player.previous()
            elif action == "volume_up":
                ok, _ = self.vol.up()
            elif action == "volume_down":
                ok, _ = self.vol.down()
            elif action == "mute_toggle":
                ok, _ = self.vol.mute_toggle()
            elif action == "workspace_next":
                ok, info = self.hypr.workspace_delta(1)
                label = info if ok else label
            elif action == "workspace_prev":
                ok, info = self.hypr.workspace_delta(-1)
                label = info if ok else label
            else:
                return
        except Exception as e:
            ok, label = False, f"{label}: {e}"
        if action.startswith("volume") or action == "mute_toggle":
            ok2, info2 = self.vol.percent()
            if ok2:
                ctrl["volume_pct"], ctrl["muted"] = info2
                label = f"VOLUME {info2[0]}%" + (" (muted)" if info2[1] else "")
        if action.startswith("media_"):
            ok2, np_ = self.player.now_playing()
            if ok2:
                ctrl["now_playing"] = np_
        self.log(label if ok else f"{label} (failed)")

    # -- snapshots ------------------------------------------------------
    def snapshot(self):
        with self._lock:
            jpg_ok = self.jpg is not None
            summary = dict(self.last_summary)
        snap = self.session.snapshot()
        return {
            "camera": {"running": self.running, "width": self.width,
                       "height": self.height, "fps": round(self.fps, 1),
                       "has_frame": jpg_ok, "error": self.last_error},
            "hands": summary["hands"],
            "gesture": summary["gesture"],
            "faces": summary["faces"],
            "privacy": {"mode": self.privacy.mode},
            "control": {"enabled": self.engine.enabled,
                        "volume_mode": self.engine.volume_mode,
                        "volume_state": self.engine.get_volume_status(),
                        "last_action": self.ctrl["last_action"],
                        "volume_pct": self.ctrl["volume_pct"],
                        "muted": self.ctrl["muted"],
                        "now_playing": self.ctrl["now_playing"]},
            "analytics": {"frames": snap["frames"],
                          "unique_hands": snap["unique_hands"],
                          "unique_faces": snap["unique_faces"],
                          "active": snap["active"],
                          "avg_confidence": snap["avg_confidence"],
                          "infer_ms": snap["infer_ms"]},
            "events": list(self.events)[-10:],
        }
