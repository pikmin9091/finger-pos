import os
# Only force xcb on X11; on Wayland-only systems forcing xcb breaks window creation.
if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse
import json
import time

import cv2

from capture import Camera
from detect import detect_hands
from face import detect_faces
from multi import HandSlots, FaceSlots
from privacy import PrivacyController, MODES, BLUR_KINDS
from analytics import AnalyticsSession
from blur import blur_faces
from control import ControlEngine
from actions import MediaPlayer, Volume, Hyprland
from positions import (draw_landmarks, draw_gesture, draw_status,
                       draw_face_text, draw_blur_status,
                       draw_box, landmarks_box, UiToasts)


def has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _notify(ctrl, toasts, text, timestamp, ttl=2.5, color=(0, 255, 255)):
    ctrl["notif"] = text
    ctrl["notif_until"] = timestamp + ttl
    toasts.push(text, color, timestamp)
    print(text, flush=True)


def _pinch_ratio(landmarks):
    """Thumb-index distance / hand scale (wrist-middle_mcp), or None."""
    try:
        if not landmarks or len(landmarks) < 10:
            return None
        t, i, w, m = landmarks[4], landmarks[8], landmarks[0], landmarks[9]
        import math
        num = math.sqrt((t["x"] - i["x"]) ** 2 + (t["y"] - i["y"]) ** 2
                        + (t["z"] - i["z"]) ** 2)
        den = math.sqrt((w["x"] - m["x"]) ** 2 + (w["y"] - m["y"]) ** 2
                        + (w["z"] - m["z"]) ** 2)
        return num / den if den > 1e-9 else None
    except Exception:
        return None


def _refresh_status(player, vol, ctrl):
    ok, info = player.now_playing()
    ctrl["now_playing"] = info if ok else ""
    ok, info = vol.percent()
    if ok:
        ctrl["volume_pct"], ctrl["muted"] = info
    else:
        ctrl["volume_pct"], ctrl["muted"] = None, None


def _fire(action, engine, player, vol, hypr, ctrl, toasts, timestamp):
    """Execute one allowlisted control action. Never raises."""
    from control import ALLOWLIST
    label = engine.label(action) if isinstance(action, str) else action[0]
    ctrl["last_action"] = action if isinstance(action, str) else action[0]
    if action == "control_toggle":
        _notify(ctrl, toasts, f"CONTROL {'ON' if engine.enabled else 'OFF'}",
                timestamp)
        return
    if isinstance(action, tuple):
        kind, val = action
        if kind == "volume_mode":
            _notify(ctrl, toasts,
                    f"VOLUME MODE {'ON — spread pinch, pinch to lock' if val else 'OFF (locked)'}",
                    timestamp)
        elif kind == "volume_set":
            try:
                ok, info = vol.set_absolute(val)
            except Exception as e:
                ok, info = False, str(e)
            ctrl["volume_pct"] = val if ok else ctrl["volume_pct"]
            _notify(ctrl, toasts,
                    f"VOLUME {val}%" if ok else f"VOLUME (failed)",
                    timestamp)
        return
    try:
        if action == "media_play_pause":
            ok, _ = player.play_pause()
        elif action == "media_next":
            ok, _ = player.next()
        elif action == "media_previous":
            ok, _ = player.previous()
        elif action == "volume_up":
            ok, _ = vol.up()
        elif action == "volume_down":
            ok, _ = vol.down()
        elif action == "mute_toggle":
            ok, _ = vol.mute_toggle()
        elif action == "workspace_next":
            ok, info = hypr.workspace_delta(1)
            label = info if ok else label
        elif action == "workspace_prev":
            ok, info = hypr.workspace_delta(-1)
            label = info if ok else label
        else:
            return  # not allowlisted — never executed
    except Exception as e:
        ok, label = False, f"{label}: {e}"
    if action.startswith("volume") or action == "mute_toggle":
        _refresh_status(player, vol, ctrl)
        if ctrl["volume_pct"] is not None:
            label = (f"VOLUME {ctrl['volume_pct']}%"
                     + (" (muted)" if ctrl["muted"] else ""))
    if action.startswith("media_"):
        ok2, np_ = player.now_playing()
        if ok2:
            ctrl["now_playing"] = np_
    _notify(ctrl, toasts, label if ok else f"{label} (failed)", timestamp)


def main():
    parser = argparse.ArgumentParser(description="AI camera: hand + face tracking, privacy blur, analytics")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--smoothing", type=float, default=0.2,
                        help="EMA alpha 0.0-1.0: lower = smoother but more lag (default: 0.2)")
    parser.add_argument("--debounce", type=int, default=3,
                        help="frames a new gesture must persist before switching (default: 3)")
    parser.add_argument("--pinch-threshold", type=float, default=0.06,
                        help="thumb-index distance for PINCH (default: 0.06)")
    parser.add_argument("--no-face", action="store_true",
                        help="disable face detection (hand tracking only)")
    parser.add_argument("--no-blur", action="store_true",
                        help="disable gesture-controlled face blur")
    parser.add_argument("--blur-cooldown", type=float, default=1.0,
                        help="min seconds between blur on/off toggles (default: 1.0)")
    parser.add_argument("--privacy", default="OFF", choices=list(MODES),
                        help="initial privacy mode (default: OFF)")
    parser.add_argument("--blur-kind", default="GAUSSIAN", choices=list(BLUR_KINDS),
                        help="face obfuscation style (default: GAUSSIAN)")
    parser.add_argument("--blur-strength", type=int, default=5,
                        help="obfuscation strength 1-10 (default: 5)")
    parser.add_argument("--auto-faces", type=int, default=1,
                        help="AUTO privacy engages at this face count (default: 1)")
    parser.add_argument("--control", action="store_true",
                        help="enable gesture control (media/volume/Hyprland)")
    parser.add_argument("--control-profile", default="default",
                        choices=["default", "calm"],
                        help="control sensitivity: default or calm "
                             "(confident held gestures only, no motion) (default: default)")
    parser.add_argument("--no-motion", action="store_true",
                        help="disable motion/swipe triggers in gesture control")
    args = parser.parse_args()

    print("Live AI camera — PINCH toggles privacy | keys: M P C S R Q", flush=True)
    cam = Camera(args.camera)
    slots = HandSlots(smoothing=args.smoothing,
                      pinch_threshold=args.pinch_threshold,
                      debounce_frames=args.debounce)
    face_slots = None if args.no_face else FaceSlots()
    face_failed = False
    face_warned = False

    from privacy import PrivacyRules
    privacy = PrivacyController(
        mode="OFF" if args.no_blur else args.privacy,
        rules=PrivacyRules(blur_kind=args.blur_kind,
                           strength=args.blur_strength,
                           min_faces=args.auto_faces),
        cooldown=args.blur_cooldown,
    )
    blur_locked_off = args.no_blur
    session = AnalyticsSession()
    toasts = UiToasts()  # all runtime feedback lives in the camera window

    engine = None
    if args.control:
        if args.control_profile == "calm":
            engine = ControlEngine(preset="calm")
        else:  # recommended timing: confident + held gestures
            engine = ControlEngine(min_confidence=0.8, dwell=0.7)
        if args.no_motion:
            engine.motion = False
    player, vol, hypr = MediaPlayer(), Volume(), Hyprland()
    if args.control:
        print(f"Gesture control ON (OPEN PALM toggles, C key too) | "
              f"media={'OK' if player.players() else 'none'} "
              f"volume={'OK' if vol.get()[0] else 'n/a'} "
              f"hypr={'OK' if hypr.ok() else 'n/a'}", flush=True)
    ctrl = {"last_action": None, "notif": "", "notif_until": 0.0,
            "now_playing": "", "volume_pct": None, "muted": None,
            "status_at": 0.0}

    use_window = has_display()
    if not use_window:
        print("No DISPLAY — JSON output only", flush=True)

    fps = 0.0
    _gui_warned = False
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("ERROR: camera unavailable", flush=True)
                break

            timestamp = time.time()
            infer_t0 = time.time()
            detected = detect_hands(frame)["hands"]
            hands = slots.update(detected, timestamp=timestamp)

            faces = []
            if face_slots is not None and not face_failed:
                try:
                    faces = face_slots.update(detect_faces(frame)["faces"])
                except FileNotFoundError as e:
                    print(f"WARNING: {e} — continuing without faces", flush=True)
                    face_failed = True
                except Exception as e:
                    if not face_warned:
                        print(f"WARNING: face detection error: {e}", flush=True)
                        face_warned = True
            infer_ms = (time.time() - infer_t0) * 1000.0

            # PINCH routing: volume mode while gesture-control master is ON,
            # privacy toggle otherwise — never both, no ambiguity.
            pinch_now = any(h["gesture"]["gesture"] == "PINCH"
                            and h["gesture"]["stable"] for h in hands)
            if engine is not None and engine.enabled:
                ratio = (_pinch_ratio(detected[0]["landmarks"])
                         if detected else None)
                primary = hands[0] if hands else None
                if primary is not None:
                    g = primary["gesture"]
                    res = engine.update(g["gesture"], g["confidence"],
                                        primary["movement"]["direction"],
                                        timestamp, pinch=pinch_now,
                                        pinch_ratio=ratio)
                    if res is not None:
                        _fire(res, engine, player, vol, hypr, ctrl, toasts,
                              timestamp)
                if timestamp - ctrl["status_at"] > 2.5:
                    ctrl["status_at"] = timestamp
                    _refresh_status(player, vol, ctrl)
            else:
                if engine is not None and hands:
                    # master OFF: keep gesture state fresh, no firing.
                    g = hands[0]["gesture"]
                    engine.update(g["gesture"], g["confidence"],
                                  hands[0]["movement"]["direction"], timestamp)
                if not blur_locked_off:
                    before = privacy.mode
                    privacy.update_pinch(pinch_now, timestamp)
                    if privacy.mode != before:
                        msg = f"Privacy mode -> {privacy.mode}"
                        toasts.push(msg, (0, 255, 0), timestamp)
                        print(msg, flush=True)

            priv = privacy.evaluate(faces, timestamp)
            session.update(hands, faces, infer_ms, fps, timestamp)
            snap = session.snapshot()

            data = {"hands": hands, "faces": faces,
                    "blur": {"enabled": priv["active"],
                             "faces_blurred": 0},
                    "privacy": {"mode": privacy.mode,
                                "active": priv["active"],
                                "kind": priv["kind"],
                                "reason": priv["reason"]},
                    "analytics": snap}
            if engine is not None:
                data["control"] = {
                    "enabled": engine.enabled,
                    "volume_mode": engine.volume_mode,
                    "last_action": ctrl["last_action"],
                    "volume_pct": ctrl["volume_pct"],
                    "muted": ctrl["muted"],
                    "now_playing": ctrl["now_playing"],
                }

            frame_time = time.time() - timestamp
            inst_fps = 1.0 / max(frame_time, 1e-6)
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps

            key = None
            if use_window:
                # Text + plain boxes only: one box per detected object.
                overlay = frame.copy()
                if detected:
                    for i, h in enumerate(detected):
                        overlay = draw_landmarks(overlay, h["landmarks"])
                        overlay = draw_gesture(overlay, hands[i]["gesture"],
                                               y=30 + i * 30)
                else:
                    overlay = draw_status(overlay, "NO HAND")
                if priv["boxes"]:
                    data["blur"]["faces_blurred"] = blur_faces(
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
                for i, face in enumerate(faces):
                    overlay = draw_face_text(overlay, face, y=90 + i * 25)
                overlay = draw_blur_status(overlay, priv["active"],
                                           data["blur"]["faces_blurred"], y=120)
                if engine is not None:
                    cv2.putText(overlay,
                                f"CONTROL: {'ON' if engine.enabled else 'OFF'}"
                                + (f" | VOL: {ctrl['volume_pct']}%"
                                   + (" (muted)" if ctrl["muted"] else "")
                                   if ctrl["volume_pct"] is not None else ""),
                                (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if engine.enabled else (150, 150, 150), 2)
                    if ctrl["now_playing"]:
                        cv2.putText(overlay, f"NOW: {ctrl['now_playing'][:44]}",
                                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)
                    if timestamp < ctrl["notif_until"]:
                        cv2.putText(overlay, ctrl["notif"], (10, 195),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if engine.volume_mode:
                        cv2.putText(overlay, "VOLUME MODE: spread pinch = louder"
                                    " (pinch to lock)",
                                    (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 165, 255), 2)
                overlay = toasts.draw(overlay, timestamp)
                cv2.putText(overlay, f"FPS: {fps:.0f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                try:
                    cv2.imshow("Finger Detection", overlay)
                except Exception as e:
                    if not _gui_warned:
                        print(f"WARNING: preview unavailable ({e}) — "
                              "install the GUI OpenCV build for the camera window; "
                              "continuing JSON-only", flush=True)
                        _gui_warned = True
                else:
                    key = cv2.waitKey(1) & 0xFF

            print(json.dumps(data), flush=True)

            if key is not None:
                if key == ord("q"):
                    break
                elif key == ord("m") and not blur_locked_off:
                    msg = f"Privacy mode -> {privacy.cycle_mode()}"
                    toasts.push(msg, (0, 255, 0), timestamp)
                    print(msg, flush=True)
                elif key == ord("p") and not blur_locked_off:
                    msg = f"Privacy mode -> {privacy.toggle_mode()}"
                    toasts.push(msg, (0, 255, 0), timestamp)
                    print(msg, flush=True)
                elif key == ord("s"):
                    tag = int(time.time())
                    jp = session.export_json(f"analytics_session_{tag}.json")
                    cp = session.export_csv(f"analytics_session_{tag}.csv")
                    msg = f"Session exported -> {jp}, {cp}"
                    toasts.push(msg, timestamp=timestamp)
                    print(msg, flush=True)
                elif key == ord("r"):
                    session.reset()
                    msg = "Analytics session reset"
                    toasts.push(msg, timestamp=timestamp)
                    print(msg, flush=True)
                elif key == ord("c") and engine is not None:
                    engine.enabled = not engine.enabled
                    msg = f"Gesture control {'ON' if engine.enabled else 'OFF'}"
                    toasts.push(msg, (0, 255, 0) if engine.enabled else (150, 150, 150),
                                timestamp)
                    print(msg, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if use_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
