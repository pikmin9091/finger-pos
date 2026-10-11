"""Dataset collector: record landmark samples for one gesture label.

Saves landmarks (never video) as one JSONL session file per run, so
train/test splits stay separated by session — never by adjacent frames.
Each line is either a meta header or one sample; features are
recomputed at train time via features.py (guaranteed train/infer parity).

Usage:
    ./myenv/bin/python collect.py --label OPEN_PALM [--camera 0]
    Keys while recording: SPACE start/stop, Q quit+save.
    Record several sessions per label: vary distance (0.4-1.2 m),
    roll (+-45 deg), lighting, and speed. Labels: OPEN_PALM FIST PINCH
    THUMBS_UP THUMBS_DOWN PEACE POINT UNKNOWN.
"""

import argparse
import json
import os
import time

from features import LABELS

KEY_LABELS = {str(i + 1): lab for i, lab in enumerate(LABELS)}


def session_path(out_dir, label, timestamp=None):
    ts = int(timestamp if timestamp is not None else time.time())
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return os.path.join(out_dir, f"{safe}_{ts}.jsonl")


def write_meta(fh, label, session, source="camera"):
    from features import FEATURE_VERSION
    fh.write(json.dumps({"type": "meta", "label": label, "session": session,
                         "source": source,
                         "feature_version": FEATURE_VERSION}) + "\n")


def write_sample(fh, landmarks, handedness=None, confidence=0.0,
                 timestamp=None):
    clean = [{k: p.get(k, 0.0) for k in ("x", "y", "z", "visibility")}
             for p in landmarks]
    fh.write(json.dumps({"type": "sample",
                         "t": round(timestamp if timestamp is not None
                                    else time.time(), 3),
                         "handedness": handedness,
                         "confidence": round(float(confidence or 0.0), 4),
                         "landmarks": clean}) + "\n")


def read_session(path):
    """Read one session file -> (meta, samples). Raises on bad label."""
    meta, samples = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "meta":
                meta = obj
            elif obj.get("type") == "sample":
                samples.append(obj)
    if meta is None or meta.get("label") not in LABELS:
        raise ValueError(f"bad session header in {path}")
    return meta, [s for s in samples
                  if s.get("landmarks") and len(s["landmarks"]) >= 21]


def main():
    ap = argparse.ArgumentParser(description="record landmark samples")
    ap.add_argument("--label", required=True, choices=list(LABELS))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="data/sessions")
    args = ap.parse_args()

    import cv2
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise SystemExit("ERROR: collector needs a display for framing "
                         "feedback (run under Hyprland).")
    from capture import Camera
    from detect import detect_hands

    os.makedirs(args.out, exist_ok=True)
    session = os.path.basename(args.label) + "_" + str(int(time.time()))
    path = session_path(args.out, args.label)
    cam = Camera(args.camera)
    recording, count = False, 0
    fh = open(path, "w")
    try:
        write_meta(fh, args.label, session)
        print(f"Session {session} -> {path}", flush=True)
        print("SPACE record/stop | Q quit+save | vary distance/roll/light",
              flush=True)
        while True:
            ret, frame = cam.read()
            if not ret:
                print("ERROR: camera unavailable", flush=True)
                break
            hands = detect_hands(frame)["hands"]
            n = len(hands)
            if recording and hands:
                h = hands[0]
                write_sample(fh, h["landmarks"], h.get("handedness"),
                             h.get("confidence", 0.0))
                count += 1
            overlay = frame.copy()
            cv2.putText(overlay,
                        f"{args.label} [{'REC' if recording else 'idle'}] "
                        f"samples={count} hands={n}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if recording else (0, 255, 0), 2)
            cv2.putText(overlay, "SPACE rec/stop | Q save+quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Dataset collector", overlay)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == 32:  # SPACE
                recording = not recording
                print(f"recording={'ON' if recording else 'OFF'} "
                      f"samples={count}", flush=True)
    finally:
        fh.close()
        cam.release()
        cv2.destroyAllWindows()
    print(f"Saved {count} samples -> {path}", flush=True)


if __name__ == "__main__":
    main()
