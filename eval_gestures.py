"""Synthetic gesture robustness evaluation (NOT real-world accuracy).

Builds a deterministic grid of poses per gesture — in-plane rotations,
scales, and seeded noise — plus ambiguous/borderline cases, then runs the
real GestureDetector pipeline (debounce included) and reports accuracy,
per-class precision/recall/F1, confusion matrix, UNKNOWN rate, and
confirm latency in frames.

What this measures: geometric robustness of the classifier on clean
synthetic poses. What it does NOT measure: real-camera accuracy, which
depends on lighting, skin tone, background, motion blur, and occlusion.
Do not quote these numbers as product accuracy; use them to catch
regressions and compare thresholds. Real validation needs the manual
protocol in README.md.
"""

import math
import random
from collections import Counter, defaultdict

from gesture import GestureDetector
from test_gesture import make_landmarks

ROLLS = [0, 30, 90, 180, 270]
SCALES = [0.8, 1.0, 1.3]
NOISE = 0.004
FRAMES = 6


def transform(lm, roll_deg=0, scale=1.0, rng=None):
    rad = math.radians(roll_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    out = []
    for p in lm:
        x = (p["x"] - 0.5) * scale
        y = (p["y"] - 0.5) * scale
        rx = x * cos_a - y * sin_a + 0.5
        ry = x * sin_a + y * cos_a + 0.5
        if rng is not None:
            rx += rng.gauss(0, NOISE)
            ry += rng.gauss(0, NOISE)
        out.append({"x": rx, "y": ry, "z": p.get("z", 0.0)})
    return out


def weak_trio_pose():
    """Genuinely weak three-finger pose: index+middle+ring at ~140 deg.

    No rule fully matches (PEACE needs ring folded, POINT needs middle
    folded, OPEN_PALM needs 4+), and single-finger noise flips land in
    weak-pattern or no-match — expect UNKNOWN robustly, not a snap.
    """
    lm = make_landmarks({"index": True, "middle": True, "ring": True})
    lm[8] = {"x": 0.5686, "y": 0.2568, "z": 0.0}
    lm[12] = {"x": 0.6286, "y": 0.2568, "z": 0.0}
    lm[16] = {"x": 0.6886, "y": 0.2568, "z": 0.0}
    return lm


def half_bent_index():
    """Backward-compatible alias for the weak ambiguous pose."""
    return weak_trio_pose()


CASES = [
    ("POINT", lambda: make_landmarks({"index": True})),
    ("PEACE", lambda: make_landmarks({"index": True, "middle": True})),
    ("FIST", lambda: make_landmarks({})),
    ("OPEN_PALM", lambda: make_landmarks(
        {f: True for f in ["thumb", "index", "middle", "ring", "pinky"]})),
    ("THUMBS_UP", lambda: _thumbs(True)),
    ("THUMBS_DOWN", lambda: _thumbs(False)),
]


def _thumbs(up):
    lm = make_landmarks({"thumb": True})
    lm[4] = {"x": 0.5, "y": 0.2 if up else 0.8, "z": 0.0}
    return lm


def run_case(detector, lm):
    last = None
    for _ in range(FRAMES):
        last = detector.detect(lm)
    return last


def main():
    rng = random.Random(7)
    labels, preds, latencies = [], [], []
    for want, build in CASES:
        for roll in ROLLS:
            for scale in SCALES:
                lm = transform(build(), roll, scale, rng)
                d = GestureDetector(debounce_frames=2)
                got, frames = "NONE", FRAMES
                for i in range(FRAMES):
                    r = d.detect(lm)
                    if r["gesture"] == want:
                        got, frames = want, i + 1
                        break
                    got = r["gesture"]
                labels.append(want)
                preds.append(got)
                latencies.append(frames)
    # ambiguous set: correct answer is rejection (NONE)
    amb_rejected = 0
    amb_total = 0
    for roll in ROLLS:
        for variant in (weak_trio_pose,
                        lambda: make_landmarks({"index": True, "ring": True})):
            lm = transform(variant(), roll, 1.0, rng)
            d = GestureDetector(debounce_frames=2)
            got = run_case(d, lm)["gesture"]
            amb_total += 1
            amb_rejected += (got == "NONE")

    classes = sorted(set(labels))
    print(f"cases={len(labels)} accuracy="
          f"{sum(a == b for a, b in zip(labels, preds)) / len(labels):.3f}")
    print(f"ambiguous rejected: {amb_rejected}/{amb_total}")
    print(f"mean confirm latency: {sum(latencies) / len(latencies):.1f} frames")
    print("\nper-class P/R/F1:")
    for c in classes:
        tp = sum(1 for a, b in zip(labels, preds) if a == c and b == c)
        fp = sum(1 for a, b in zip(labels, preds) if a != c and b == c)
        fn = sum(1 for a, b in zip(labels, preds) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  {c:10s} P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print("\nconfusion (true x pred):")
    pred_classes = sorted(set(preds))
    print(f"  {'':10s} " + " ".join(f"{c[:6]:>6s}" for c in pred_classes))
    for c in classes:
        row = Counter(b for a, b in zip(labels, preds) if a == c)
        print(f"  {c:10s} " + " ".join(f"{row.get(p, 0):>6d}" for p in pred_classes))


if __name__ == "__main__":
    main()
