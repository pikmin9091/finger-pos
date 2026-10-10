import argparse
import sys
import cv2
from detect import detect_hands
from face import detect_faces
from face_tracker import FaceTracker
from multi import HandSlots
from blur import blur_faces
from calib import load_calib
from positions import get_finger_tips

parser = argparse.ArgumentParser(description="Static image hand + face report")
parser.add_argument("image", help="path to image file")
parser.add_argument("--blur", action="store_true",
                    help="apply face blur to the preview")
parser.add_argument("--out", default=None,
                    help="save annotated preview to this path (nothing saved by default)")
parser.add_argument("--debug", action="store_true",
                    help="print raw classifier internals (scores, state, reason)")
args = parser.parse_args()

frame = cv2.imread(args.image)
if frame is None:
    print(f"ERROR: cannot read {args.image}")
    sys.exit(1)

hands = detect_hands(frame)["hands"]
calib = load_calib()
if calib:
    print(f"Calibration: {calib}")
slots = HandSlots(debounce_frames=1, calib=calib)  # immediate gesture for single images
entries = slots.update(hands, timestamp=0.0)
print(f"Hands: {len(entries)}")
for n, (h, e) in enumerate(zip(hands, entries)):
    tips = get_finger_tips(h["landmarks"])
    g = e["gesture"]
    pinch = "PINCH" in g["gesture"]
    print(f"Hand {n}: {h['handedness']} (conf={h['confidence']:.2f}) "
          f"gesture={g['gesture']} (conf={g['confidence']:.2f} stable={g['stable']}) "
          f"{'<-- blur toggle gesture' if pinch else ''}")
    if args.debug:
        print(f"  raw={g.get('raw')} state={g.get('state')} reason={g.get('reason')} "
              f"scores={g.get('scores')}")
    for i, t in enumerate(tips or []):
        print(f"  finger {i}: x={t['x']:.3f} y={t['y']:.3f} conf={t['conf']:.2f}")

faces = detect_faces(frame)["faces"]
print(f"Faces: {len(faces)}")
for n, f in enumerate(faces):
    face = FaceTracker().update(f["landmarks"], confidence=f["confidence"])
    if face is None:
        continue
    c, b = face["center"], face["box"]
    print(f"Face {n}: center=({c['x']:.3f}, {c['y']:.3f}) "
          f"box=({b['x_min']:.3f}, {b['y_min']:.3f})-({b['x_max']:.3f}, {b['y_max']:.3f})")
    print(f"  zone={face['horizontal']}/{face['vertical']} "
          f"looking={face['looking']} (yaw={face['yaw']:.3f} pitch={face['pitch']:.3f}) "
          f"proximity={face['proximity']}")

if args.out:
    out = frame.copy()
    if args.blur:
        boxes = []
        for f in faces:
            face = FaceTracker().update(f["landmarks"], confidence=f["confidence"])
            if face is not None:
                boxes.append(face["box"])
        n = blur_faces(out, boxes)
        print(f"Blurred {n} face(s) -> {args.out}")
    if not cv2.imwrite(args.out, out):
        print(f"ERROR: cannot write {args.out}")
        sys.exit(1)
    print(f"Saved {args.out}")
