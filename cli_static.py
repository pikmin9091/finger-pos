import sys
import cv2
from detect import detect_hand
from positions import get_finger_tips

if len(sys.argv) < 2:
    print("Usage: python3 cli_static.py <image_path>")
    sys.exit(1)

frame = cv2.imread(sys.argv[1])
if frame is None:
    print(f"ERROR: cannot read {sys.argv[1]}")
    sys.exit(1)

res = detect_hand(frame)
lm = res["landmarks"]
tips = get_finger_tips(lm)
print(f"Hand: {res['handedness']} (conf={res['confidence']:.2f})")
print(f"Landmarks: {len(lm)}, Tips: {len(tips) if tips else 0}")
for i, t in enumerate(tips or []):
    print(f"  finger {i}: x={t['x']:.3f} y={t['y']:.3f} conf={t['conf']:.2f}")