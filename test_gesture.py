"""Tests for GestureDetector — synthetic landmarks only, no webcam needed."""

import pytest
from gesture import GestureDetector, FINGER_JOINTS


def make_landmarks(extended=None):
    """Create 21 synthetic landmarks matching MediaPipe indices.

    Anatomically proportioned (palm length wrist->middle_mcp ~= 0.22):
    palm faces camera, fingers point upward (lower y = higher on screen).
    Folded fingers park the tip at its PIP (degenerate, like a real curl);
    extended fingers point straight up. Folded thumb lies across the palm
    near the index base (as in a real fist), NOT far from the palm.
    """
    if extended is None:
        extended = {f: False for f in ["thumb", "index", "middle", "ring", "pinky"]}

    lm = [
        {"x": 0.50, "y": 0.72, "z": 0.0},   # 0  wrist
        {"x": 0.46, "y": 0.66, "z": 0.0},   # 1  thumb_cmc
        {"x": 0.44, "y": 0.60, "z": 0.0},   # 2  thumb_mcp
        {"x": 0.40, "y": 0.56, "z": 0.0},   # 3  thumb_ip
        {"x": 0.47, "y": 0.55, "z": 0.0},   # 4  thumb_tip (folded: across palm)
        {"x": 0.44, "y": 0.50, "z": 0.0},   # 5  index_mcp
        {"x": 0.44, "y": 0.41, "z": 0.0},   # 6  index_pip
        {"x": 0.44, "y": 0.34, "z": 0.0},   # 7  index_dip
        {"x": 0.44, "y": 0.41, "z": 0.0},   # 8  index_tip (folded = at pip)
        {"x": 0.50, "y": 0.50, "z": 0.0},   # 9  middle_mcp
        {"x": 0.50, "y": 0.41, "z": 0.0},   # 10 middle_pip
        {"x": 0.50, "y": 0.34, "z": 0.0},   # 11 middle_dip
        {"x": 0.50, "y": 0.41, "z": 0.0},   # 12 middle_tip (folded)
        {"x": 0.56, "y": 0.50, "z": 0.0},   # 13 ring_mcp
        {"x": 0.56, "y": 0.41, "z": 0.0},   # 14 ring_pip
        {"x": 0.56, "y": 0.34, "z": 0.0},   # 15 ring_dip
        {"x": 0.56, "y": 0.41, "z": 0.0},   # 16 ring_tip (folded)
        {"x": 0.615, "y": 0.52, "z": 0.0},  # 17 pinky_mcp
        {"x": 0.615, "y": 0.45, "z": 0.0},  # 18 pinky_pip
        {"x": 0.615, "y": 0.40, "z": 0.0},  # 19 pinky_dip
        {"x": 0.615, "y": 0.45, "z": 0.0},  # 20 pinky_tip (folded)
    ]

    # Extended fingertips point straight up from their MCPs; the extended
    # thumb sticks out to the side, away from the palm.
    ext_tip = {"thumb": (0.30, 0.55), "index": (0.44, 0.26),
               "middle": (0.50, 0.26), "ring": (0.56, 0.26),
               "pinky": (0.615, 0.28)}
    tip_map = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    for fname, is_ext in extended.items():
        if is_ext:
            lm[tip_map[fname]]["x"], lm[tip_map[fname]]["y"] = ext_tip[fname]

    return lm


def _detect_stable(detector, lm, frames=8, handedness=None):
    """Run detect() for N frames and return the last result."""
    result = None
    for _ in range(frames):
        result = detector.detect(lm, handedness=handedness)
    return result


# ── Fingertip extension ──────────────────────────────────────

def test_extended_fingers_point():
    detector = GestureDetector(debounce_frames=1)
    lm = make_landmarks({"index": True})
    assert _detect_stable(detector, lm)["gesture"] == "POINT"


def test_extended_fingers_peace():
    detector = GestureDetector(debounce_frames=1)
    lm = make_landmarks({"index": True, "middle": True})
    assert _detect_stable(detector, lm)["gesture"] == "PEACE"


def test_extended_fingers_fist():
    detector = GestureDetector(debounce_frames=1)
    assert _detect_stable(detector, make_landmarks({}))["gesture"] == "FIST"


def test_extended_fingers_open_palm():
    detector = GestureDetector(debounce_frames=1)
    lm = make_landmarks({f: True for f in ["thumb", "index", "middle", "ring", "pinky"]})
    assert _detect_stable(detector, lm)["gesture"] == "OPEN_PALM"


# ── THUMBS_UP / THUMBS_DOWN ───────────────────────────────────

def test_thumbs_up():
    detector = GestureDetector(debounce_frames=1)
    lm = make_landmarks({"thumb": True})
    lm[4] = {"x": 0.5, "y": 0.2, "z": 0.0}
    assert _detect_stable(detector, lm)["gesture"] == "THUMBS_UP"


def test_thumbs_down():
    detector = GestureDetector(debounce_frames=1)
    lm = make_landmarks({"thumb": True})
    lm[4] = {"x": 0.5, "y": 0.8, "z": 0.0}
    assert _detect_stable(detector, lm)["gesture"] == "THUMBS_DOWN"


# ── PINCH ─────────────────────────────────────────────────────

def test_pinch_close():
    detector = GestureDetector(pinch_threshold=0.06, debounce_frames=1)
    lm = make_landmarks({})
    lm[4] = {"x": 0.4, "y": 0.4, "z": 0.0}
    lm[8] = {"x": 0.4, "y": 0.4, "z": 0.0}
    assert _detect_stable(detector, lm)["gesture"] == "PINCH"


def test_pinch_far():
    detector = GestureDetector(pinch_threshold=0.06, debounce_frames=1)
    lm = make_landmarks({})
    lm[4] = {"x": 0.2, "y": 0.5, "z": 0.0}
    lm[8] = {"x": 0.8, "y": 0.5, "z": 0.0}
    assert _detect_stable(detector, lm)["gesture"] != "PINCH"


def test_pinch_threshold_configurable():
    loose = GestureDetector(pinch_threshold=0.15, debounce_frames=1)
    lm = make_landmarks({})
    lm[4] = {"x": 0.4, "y": 0.45, "z": 0.0}
    lm[8] = {"x": 0.42, "y": 0.45, "z": 0.0}
    assert _detect_stable(loose, lm)["gesture"] == "PINCH"


# ── Confidence ─────────────────────────────────────────────────

def test_confidence_pinch_close():
    detector = GestureDetector(pinch_threshold=0.06, debounce_frames=1)
    lm = make_landmarks({})
    lm[4] = {"x": 0.4, "y": 0.4, "z": 0.0}
    lm[8] = {"x": 0.4, "y": 0.4, "z": 0.0}
    assert _detect_stable(detector, lm)["confidence"] > 0.9


def test_confidence_fist_all_folded():
    detector = GestureDetector(debounce_frames=1)
    assert _detect_stable(detector, make_landmarks({}))["confidence"] >= 0.8


def test_confidence_open_palm_all_extended():
    detector = GestureDetector(debounce_frames=1)
    lm = make_landmarks({f: True for f in ["thumb", "index", "middle", "ring", "pinky"]})
    assert _detect_stable(detector, lm)["confidence"] >= 0.8


# ── Stability / Debounce ───────────────────────────────────────

def test_stable_after_multiple_frames():
    detector = GestureDetector(stable_frames=5, debounce_frames=1)
    lm = make_landmarks({"index": True})
    for _ in range(6):
        result = detector.detect(lm)
    assert result["stable"] is True
    assert result["gesture"] == "POINT"


def test_unstable_before_threshold():
    detector = GestureDetector(stable_frames=5)
    assert detector.detect(make_landmarks({"index": True}))["stable"] is False


def test_debounce_transition():
    detector = GestureDetector(debounce_frames=3, stable_frames=1)
    lm_point = make_landmarks({"index": True})
    lm_fist = make_landmarks({})

    for _ in range(5):
        detector.detect(lm_point)

    for _ in range(2):
        assert detector.detect(lm_fist)["gesture"] == "POINT"

    assert detector.detect(lm_fist)["gesture"] == "FIST"


def test_debounce_noise_ignored():
    detector = GestureDetector(debounce_frames=3, stable_frames=3)
    lm_point = make_landmarks({"index": True})
    lm_fist = make_landmarks({})

    for _ in range(5):
        detector.detect(lm_point)

    detector.detect(lm_fist)
    detector.detect(lm_point)
    detector.detect(lm_fist)

    assert detector.detect(lm_point)["gesture"] == "POINT"


# ── Left / Right hand ─────────────────────────────────────────

def test_left_hand_point():
    detector = GestureDetector(debounce_frames=1)
    assert _detect_stable(detector, make_landmarks({"index": True}), handedness="Left")["gesture"] == "POINT"


def test_right_hand_peace():
    detector = GestureDetector(debounce_frames=1)
    assert _detect_stable(detector, make_landmarks({"index": True, "middle": True}), handedness="Right")["gesture"] == "PEACE"


# ── Noisy landmarks ────────────────────────────────────────────

def test_noisy_landmarks_stable():
    detector = GestureDetector(stable_frames=3, debounce_frames=2)
    base = make_landmarks({"index": True})

    for _ in range(3):
        detector.detect(base)

    noisy = [dict(l) for l in base]
    noisy[8]["x"] += 0.01

    assert detector.detect(noisy)["gesture"] == "POINT"


# ── Empty / edge cases ─────────────────────────────────────────

def test_no_landmarks_returns_none():
    detector = GestureDetector()
    result = detector.detect([])
    assert result["gesture"] == "NONE"
    assert result["confidence"] == 0.0


def test_unknown_gesture():
    detector = GestureDetector(debounce_frames=1)
    lm = make_landmarks({"index": True, "ring": True})
    assert _detect_stable(detector, lm)["gesture"] == "NONE"


# ── Output structure ───────────────────────────────────

def test_output_structure():
    detector = GestureDetector(debounce_frames=1)
    result = _detect_stable(detector, make_landmarks({"index": True}))
    assert "gesture" in result
    assert "confidence" in result
    assert "stable" in result


def test_all_gesture_names():
    detector = GestureDetector()
    for gesture in GestureDetector.GESTURES:
        assert gesture in detector.GESTURES