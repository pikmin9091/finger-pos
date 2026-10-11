"""ML feature extraction from MediaPipe hand landmarks (versioned).

Single source of truth for training AND inference: both call
extract_features(), so the format can never drift between
`train_model.py` and the live loop. Bump FEATURE_VERSION whenever the
layout changes and retrain (old models refuse to load — see ml_model).

Layout v1 (75 floats):
  - 21 joints x/y/z relative to the wrist, divided by hand scale (63)
  - tip-to-wrist distances / scale for thumb..pinky (5)
  - PIP joint angles /180 for index..pinky (4, 0.5 when degenerate)
  - thumb tip-to-palm reach / scale (1)
  - thumb-vs-palm-axis alignment cosine (1)
  - thumb-index pinch ratio (1)

Invariants: translation + scale free by construction; rotation handled
by training augmentation (mirroring + rolls in train_model.py), the same
way the deterministic classifier stays rotation-invariant. Raw pixel
coordinates are never used. Handedness is NOT a feature (MediaPipe
labels flicker); left/right are covered by mirrored training samples.
"""

import math

from gesture import FINGER_JOINTS

FEATURE_VERSION = 1

LABELS = ("OPEN_PALM", "FIST", "PINCH", "THUMBS_UP", "THUMBS_DOWN",
          "PEACE", "POINT", "UNKNOWN")

TIP_INDICES = (4, 8, 12, 16, 20)
ANGLE_FINGERS = ("index", "middle", "ring", "pinky")

FEATURE_DIM = 21 * 3 + 5 + 4 + 1 + 1 + 1  # 75


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                     + (a[2] - b[2]) ** 2)


def _angle(a, vertex, c):
    """Joint angle in degrees, or None when degenerate."""
    v1 = (a[0] - vertex[0], a[1] - vertex[1], a[2] - vertex[2])
    v2 = (c[0] - vertex[0], c[1] - vertex[1], c[2] - vertex[2])
    n1 = math.sqrt(sum(v * v for v in v1))
    n2 = math.sqrt(sum(v * v for v in v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos_val = max(-1.0, min(1.0, sum(x * y for x, y in zip(v1, v2))
                            / (n1 * n2)))
    return math.degrees(math.acos(cos_val))


def _hand_scale(pts):
    """Larger palm axis (same definition as the deterministic detector)."""
    wrist_to_middle = _dist(pts[0], pts[9])
    palm_width = _dist(pts[5], pts[17])
    return max(wrist_to_middle, palm_width, 1e-6)


def _thumb_alignment(pts):
    """Cosine between thumb axis and wrist->middle-MCP palm axis."""
    tip, mcp = pts[4], pts[2]
    wrist, middle = pts[0], pts[9]
    tax = (tip[0] - mcp[0], tip[1] - mcp[1], tip[2] - mcp[2])
    pax = (middle[0] - wrist[0], middle[1] - wrist[1], middle[2] - wrist[2])
    tl = math.sqrt(sum(v * v for v in tax))
    pl = math.sqrt(sum(v * v for v in pax))
    if tl < 1e-6 or pl < 1e-6:
        return 0.0
    return (sum(a * b for a, b in zip(tax, pax)) / (tl * pl))


def _pinch_ratio(pts):
    """Thumb-index gap / hand scale (same formula as the detector)."""
    return _dist(pts[4], pts[8]) / _hand_scale(pts) * 0.15


def mirror(landmarks):
    """Mirror landmarks on the vertical axis (left<->right augmentation)."""
    return [{**p, "x": 1.0 - p["x"]} for p in landmarks]


def extract_features(landmarks, handedness=None):
    """Convert 21 MediaPipe landmark dicts to a 75-float feature list.

    Returns None when the input is unusable (truncated, non-numeric,
    NaN/inf) — callers must treat that as UNKNOWN, never classify it.
    `handedness` is accepted for API symmetry and ignored (see above).
    """
    if not landmarks or len(landmarks) < 21:
        return None
    try:
        pts = [(float(p["x"]), float(p["y"]), float(p.get("z", 0.0)))
               for p in landmarks[:21]]
    except (KeyError, TypeError, ValueError):
        return None
    if any(not math.isfinite(v) for p in pts for v in p):
        return None

    scale = _hand_scale(pts)
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    wrist = pts[0]

    feats = []
    for p in pts:  # 63: wrist-relative, scale-normalized joints
        feats += [(p[0] - wrist[0]) / scale,
                  (p[1] - wrist[1]) / scale,
                  (p[2] - wrist[2]) / scale]
    for tip in TIP_INDICES:  # 5: tip reaches
        feats.append(_dist(pts[tip], wrist) / scale)
    for name in ANGLE_FINGERS:  # 4: joint angles in [0,1]
        joints = FINGER_JOINTS[name]
        ang = _angle(pts[joints["mcp"]], pts[joints["pip"]],
                     pts[joints["tip"]])
        feats.append(0.5 if ang is None else ang / 180.0)
    feats.append(_dist(pts[4], pts[9]) / scale)  # thumb reach
    feats.append(_thumb_alignment(pts))          # thumb alignment
    feats.append(_pinch_ratio(pts))              # pinch ratio
    assert len(feats) == FEATURE_DIM
    return feats
