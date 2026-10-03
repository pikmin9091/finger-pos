import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import argparse
import json
import time

import cv2

from capture import Camera
from detect import detect_hand
from tracker import FingerTracker
from gesture import GestureDetector
from positions import draw_landmarks, draw_gesture, draw_status


def has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def no_hand_data():
    return {
        "hand": None,
        "confidence": 0.0,
        "fingers": {},
        "hand_center": None,
        "movement": {"direction": "STATIONARY", "velocity_x": 0.0, "velocity_y": 0.0},
        "gesture": {"gesture": "NONE", "confidence": 0.0, "stable": False},
    }


def main():
    parser = argparse.ArgumentParser(description="Live finger detection")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--smoothing", type=float, default=0.2,
                        help="EMA alpha 0.0-1.0: lower = smoother but more lag (default: 0.2)")
    parser.add_argument("--debounce", type=int, default=3,
                        help="frames a new gesture must persist before switching (default: 3)")
    parser.add_argument("--pinch-threshold", type=float, default=0.06,
                        help="thumb-index distance for PINCH (default: 0.06)")
    args = parser.parse_args()

    print("Live finger detection — press 'q' to quit", flush=True)
    cam = Camera(args.camera)
    tracker = FingerTracker(smoothing=args.smoothing)
    detector = GestureDetector(
        pinch_threshold=args.pinch_threshold,
        debounce_frames=args.debounce,
    )
    use_window = has_display()
    if not use_window:
        print("No DISPLAY — JSON output only", flush=True)

    fps = 0.0
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("ERROR: camera unavailable", flush=True)
                break

            timestamp = time.time()
            result = detect_hand(frame)
            lm = result["landmarks"]

            if lm:
                data = tracker.update(
                    lm,
                    handedness=result["handedness"],
                    confidence=result["confidence"],
                    timestamp=timestamp,
                )
                gesture = detector.detect(lm, handedness=result["handedness"])
                data["gesture"] = gesture
            else:
                data = no_hand_data()

            frame_time = time.time() - timestamp
            inst_fps = 1.0 / max(frame_time, 1e-6)
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps

            if use_window:
                overlay = frame.copy()
                if lm:
                    overlay = draw_landmarks(overlay, lm)
                    overlay = draw_gesture(overlay, gesture)
                else:
                    overlay = draw_status(overlay, "NO HAND")
                cv2.putText(overlay, f"FPS: {fps:.0f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Finger Detection", overlay)

            print(json.dumps(data), flush=True)

            if use_window and cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if use_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
