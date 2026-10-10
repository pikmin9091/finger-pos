import os
# Only force xcb on X11; on Wayland-only systems forcing xcb breaks window creation.
if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse
import json
import time

import cv2

from capture import Camera
from detect import detect_hands, set_confidence as set_hand_conf
from face import detect_faces, set_confidence as set_face_conf
from multi import HandSlots, FaceSlots
from privacy import PrivacyController, MODES, BLUR_KINDS
from analytics import AnalyticsSession
from blur import blur_faces
from control import ControlEngine
from actions import MediaPlayer, Volume, Hyprland
from calib import load_calib, save_calib, summarize
from positions import (draw_landmarks, draw_gesture, draw_status,
                       draw_face_text, draw_blur_status,
                       draw_box, landmarks_box, UiToasts, draw_debug,
                       draw_volume_zone, layout_buttons, hit_test,
                       draw_buttons, draw_help)


def _noop_trackbar(_value):
    pass


def has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _run_calibrate(cam, per_pose=12, timeout_s=12.0):
    """Interactive calibration: hold each pose, collect samples, save JSON.

    Only landmark geometry is used — no images are kept. Rejects and
    reports weak sessions instead of saving bad thresholds.
    """
    import time as _t
    from gesture import GestureDetector
    from calib import summarize, save_calib

    detector = GestureDetector()  # only for angle/pinch math, no state kept
    samples = {}
    for pose, prompt in (("open", "OPEN PALM facing camera, fingers spread"),
                         ("fist", "FIST, thumb across fingers"),
                         ("pinch", "PINCH thumb+index tips together")):
        print(f"Hold {prompt} ...", flush=True)
        got, t0 = [], _t.time()
        while len(got) < per_pose and _t.time() - t0 < timeout_s:
            ret, frame = cam.read()
            if ret:
                try:
                    hands = detect_hands(frame)["hands"]
                except Exception as e:
                    print(f"  detector error: {e}", flush=True)
                    break
                if hands and len(hands[0]["landmarks"]) >= 21:
                    got.append(hands[0]["landmarks"])
            _t.sleep(0.05)
        samples[pose] = got
        print(f"  collected {len(got)}/{per_pose} ({pose})", flush=True)
    calib, warnings = summarize(detector, samples)
    for w in warnings:
        print(f"  WARNING: {w}", flush=True)
    if not calib:
        print("Calibration REJECTED — keeping previous thresholds.", flush=True)
        return
    path = save_calib(calib)
    print(f"Calibration saved -> {path}: {calib}", flush=True)


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


def _debug_info(hands, detected, ctrl, engine, timestamp, infer_ms=0.0):
    """Collect real pipeline values for the debug overlay (never simulated)."""
    info = {"raw": "-", "final": "NONE", "state": "UNKNOWN", "reason": "-",
            "bits": "-", "scores": "-", "pinch": "-", "cooldown": "-",
            "track": f"hands={len(hands)}", "handy": "-", "vol": "-",
            "thumb": "-", "latency": f"{infer_ms:.1f}ms"}
    if hands:
        g = hands[0]["gesture"]
        sc = g.get("scores", {})
        bits = "".join("1" if sc.get(f, 0) >= 0.5 else "0"
                       for f in ("thumb", "index", "middle", "ring", "pinky"))
        th = g.get("thumb", {})
        info.update(raw=g.get("raw", "?"), final=g.get("gesture", "?"),
                    state=g.get("state", "?"), reason=g.get("reason", "?"),
                    bits=f"T{bits[0]} I{bits[1]} M{bits[2]} R{bits[3]} P{bits[4]}",
                    scores=" ".join(f"{k[0].upper()}:{sc.get(k, 0):.2f}"
                                    for k in ("thumb", "index", "middle",
                                              "ring", "pinky")),
                    handy=f"y={hands[0]['hand_center']['y']:.3f}",
                    thumb=f"reach={th.get('reach', '?')} align={th.get('alignment', '?')}")
        if detected:
            ratio = _pinch_ratio(detected[0]["landmarks"])
            info["pinch"] = f"ratio={ratio:.3f}" if ratio is not None else "n/a"
    if engine is not None:
        vs = engine.get_volume_status()
        info["cooldown"] = (f"{engine.cooldown_remaining(timestamp):.1f}s"
                            + (" VOLMODE" if engine.volume_mode else ""))
        info["vol"] = (f"{vs['state']} [{engine.vol_top:.2f}-{engine.vol_bottom:.2f}]"
                       f" tgt={vs['target']}")
        info["track"] += f" ctrl={'ON' if engine.enabled else 'OFF'}"
    return info


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
                    f"VOLUME MODE {'ON — move hand up/down, palm to lock' if val else 'OFF (locked)'}",
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
    parser.add_argument("--min-velocity", type=float, default=0.05,
                        help="units/s below which motion reads STATIONARY "
                             "(default: 0.05; raise if drift misfires moves)")
    parser.add_argument("--pinch-threshold", type=float, default=None,
                        help="hand-size-normalized thumb-index gap for PINCH "
                             "(default: calib file, else 0.06)")
    parser.add_argument("--calibrate", action="store_true",
                        help="interactive threshold calibration, then exit")
    parser.add_argument("--debug", action="store_true",
                        help="debug overlay: raw/final gesture, finger bits, "
                             "pinch, cooldown, tracking state")
    parser.add_argument("--no-face", action="store_true",
                        help="disable face detection (hand tracking only)")
    parser.add_argument("--sensitivity", default="normal",
                        choices=["low", "normal", "high"],
                        help="detection sensitivity: low=0.65 (fewer false "
                             "positives), normal=0.5, high=0.35 (finds distant "
                             "hands, more flicker) (default: normal)")
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
    parser.add_argument("--vol-top", type=float, default=0.25,
                        help="hand height (0 top) mapping to volume 100%% "
                             "(default: 0.25)")
    parser.add_argument("--vol-bottom", type=float, default=0.75,
                        help="hand height mapping to volume 0%% (default: 0.75)")
    args = parser.parse_args()

    print("Live AI camera — PINCH toggles privacy | keys: M P B C S R H Q", flush=True)
    conf = {"low": 0.65, "normal": 0.5, "high": 0.35}[args.sensitivity]
    set_hand_conf(conf, conf, conf)
    set_face_conf(conf, conf, conf)
    cam = Camera(args.camera)

    calib = load_calib()
    if calib:
        print(f"Calibration loaded: {calib}", flush=True)
    pinch_thr = (args.pinch_threshold if args.pinch_threshold is not None
                 else calib.get("pinch_threshold", 0.06))

    if args.calibrate:
        _run_calibrate(cam)
        cam.release()
        return

    slots = HandSlots(smoothing=args.smoothing,
                      pinch_threshold=pinch_thr,
                      debounce_frames=args.debounce,
                      calib=calib,
                      min_velocity=args.min_velocity)
    face_slots = None if args.no_face else FaceSlots()
    face_failed = False
    face_warned = False

    from privacy import PrivacyRules
    privacy = PrivacyController(
        mode="OFF" if args.no_blur else args.privacy,
        rules=PrivacyRules(auto_on_face=False,
                           blur_kind=args.blur_kind,
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
        engine.vol_top = args.vol_top
        engine.vol_bottom = args.vol_bottom
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
    show_help = False
    clicks = []  # pending mouse clicks (x, y), consumed in-loop
    ui_mouse = False
    ui_trackbars = False
    ui_inited = False
    buttons = []  # current-frame button rects for hit-testing
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

            # PINCH owns the privacy toggle whenever gesture-control
            # master is off (or control is absent). Volume no longer
            # depends on pinch at all — it uses OPEN_PALM + hand height.
            pinch_now = any(h["gesture"]["gesture"] == "PINCH"
                            and h["gesture"]["stable"] for h in hands)
            if engine is not None:
                if hands:
                    g = hands[0]["gesture"]
                    res = engine.update(
                        g["gesture"], g["confidence"],
                        hands[0]["movement"]["direction"], timestamp,
                        hand_y=hands[0]["hand_center"]["y"],
                        hand_present=True, gesture_stable=g["stable"])
                    if res is not None:
                        _fire(res, engine, player, vol, hypr, ctrl, toasts,
                              timestamp)
                else:
                    engine.update(None, 0.0, "STATIONARY", timestamp,
                                  hand_present=False)
                if timestamp - ctrl["status_at"] > 2.5:
                    ctrl["status_at"] = timestamp
                    _refresh_status(player, vol, ctrl)
            if (engine is None or not engine.enabled) and not blur_locked_off:
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
                vol_status = engine.get_volume_status()
                data["control"] = {
                    "enabled": engine.enabled,
                    "volume_mode": engine.volume_mode,
                    "volume_state": vol_status["state"],
                    "vol_target": vol_status["target"],
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
                    if engine.volume_mode or (args.debug and engine is not None):
                        overlay = draw_volume_zone(
                            overlay, engine.vol_top, engine.vol_bottom,
                            hand_y=(hands[0]["hand_center"]["y"] if hands else None))
                    if engine.volume_mode:
                        vs = engine.get_volume_status()
                        actual = (f"{ctrl['volume_pct']}%" if ctrl["volume_pct"]
                                  is not None else "-")
                        cv2.putText(overlay,
                                    f"VOLUME MODE: {vs['state']} target={vs['target']} "
                                    f"actual={actual}",
                                    (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 165, 255), 2)
                overlay = toasts.draw(overlay, timestamp)
                if args.debug:
                    overlay = draw_debug(overlay, _debug_info(
                        hands, detected, ctrl, engine, timestamp, infer_ms))
                # Clickable button bar (labels follow live state).
                btn_defs = [
                    (f"PRIV:{privacy.mode}", ord("m")),
                    (f"BLUR:{privacy.rules.blur_kind}", ord("b")),
                    (f"CTRL:{'ON' if engine is not None and engine.enabled else 'OFF'}",
                     ord("c") if engine is not None else None),
                    ("SNAP", ord("s")),
                    ("RESET", ord("r")),
                    ("HELP", ord("h")),
                    ("QUIT", ord("q")),
                ]
                buttons = layout_buttons(overlay.shape[1], overlay.shape[0], btn_defs)
                active_keys = set()
                if priv["active"]:
                    active_keys.add(ord("m"))
                if engine is not None and engine.enabled:
                    active_keys.add(ord("c"))
                overlay = draw_buttons(overlay, buttons, active_keys)
                if show_help:
                    overlay = draw_help(overlay)
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
                    if not ui_inited:
                        ui_inited = True
                        try:
                            cv2.setMouseCallback(
                                "Finger Detection",
                                lambda ev, x, y, _f, _p: clicks.append((x, y))
                                if ev == cv2.EVENT_LBUTTONDOWN else None)
                            ui_mouse = True
                        except Exception:
                            ui_mouse = False
                        try:
                            cv2.createTrackbar("smooth", "Finger Detection",
                                               int(args.smoothing * 100), 100,
                                               _noop_trackbar)
                            cv2.createTrackbar("debounce", "Finger Detection",
                                               args.debounce, 10, _noop_trackbar)
                            cv2.createTrackbar("strength", "Finger Detection",
                                               args.blur_strength, 10,
                                               _noop_trackbar)
                            ui_trackbars = True
                        except Exception:
                            ui_trackbars = False
                    if ui_trackbars:
                        try:
                            slots.set_smoothing(
                                cv2.getTrackbarPos("smooth", "Finger Detection") / 100.0)
                            slots.set_debounce(
                                cv2.getTrackbarPos("debounce", "Finger Detection"))
                            privacy.rules.strength = max(
                                1, min(10, cv2.getTrackbarPos(
                                    "strength", "Finger Detection")))
                        except Exception:
                            ui_trackbars = False
                    key = cv2.waitKey(1) & 0xFF
                    if clicks and buttons:
                        hit = hit_test(buttons, *clicks.pop(0))
                        clicks.clear()
                        if hit is not None:
                            key = hit

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
                elif key == ord("b") and not blur_locked_off:
                    msg = f"Blur kind -> {privacy.cycle_kind()}"
                    toasts.push(msg, (0, 255, 255), timestamp)
                    print(msg, flush=True)
                elif key == ord("h"):
                    show_help = not show_help
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
