"""Tests for FaceTracker — synthetic landmarks only, no webcam needed."""

import pytest
from face_tracker import (FaceTracker, NOSE_TIP, FOREHEAD, CHIN,
                          LEFT_CHEEK, RIGHT_CHEEK)


def make_face_landmarks(cx=0.5, cy=0.4, w=0.3, h=0.4,
                        nose_dx=0.0, nose_dy=0.0):
    """Create 478 synthetic landmarks on an ellipse bbox + key points.

    nose_dx: nose shift as fraction of face width (right positive).
    nose_dy: nose shift as fraction of face height (down positive).
    """
    import math
    lm = []
    for i in range(478):
        a = 2 * math.pi * i / 478
        lm.append({"x": cx + (w / 2) * math.cos(a),
                   "y": cy + (h / 2) * math.sin(a), "z": 0.0})
    lm[NOSE_TIP] = {"x": cx + nose_dx * w, "y": cy + nose_dy * h, "z": 0.0}
    lm[FOREHEAD] = {"x": cx, "y": cy - h / 2, "z": 0.0}
    lm[CHIN] = {"x": cx, "y": cy + h / 2, "z": 0.0}
    lm[LEFT_CHEEK] = {"x": cx - w / 2, "y": cy, "z": 0.0}
    lm[RIGHT_CHEEK] = {"x": cx + w / 2, "y": cy, "z": 0.0}
    return lm


# ── Empty / invalid ──────────────────────────────────────────────

def test_no_face_returns_none():
    assert FaceTracker().update([]) is None


def test_none_returns_none():
    assert FaceTracker().update(None) is None


def test_truncated_returns_none():
    assert FaceTracker().update([{"x": 0.5, "y": 0.5, "z": 0.0}] * 10) is None


# ── Center / box ─────────────────────────────────────────────────

def test_center_calculation():
    face = FaceTracker().update(make_face_landmarks(cx=0.5, cy=0.4))
    assert abs(face["center"]["x"] - 0.5) < 1e-6
    assert abs(face["center"]["y"] - 0.4) < 1e-6


def test_box_matches_spread():
    face = FaceTracker().update(make_face_landmarks(cx=0.5, cy=0.4, w=0.3, h=0.4))
    assert abs(face["box"]["x_min"] - 0.35) < 1e-6
    assert abs(face["box"]["x_max"] - 0.65) < 1e-6


# ── Zones ────────────────────────────────────────────────────────

def test_zone_left():
    assert FaceTracker().update(make_face_landmarks(cx=0.1))["horizontal"] == "LEFT"


def test_zone_right():
    assert FaceTracker().update(make_face_landmarks(cx=0.9))["horizontal"] == "RIGHT"


def test_zone_center():
    face = FaceTracker().update(make_face_landmarks(cx=0.5, cy=0.5))
    assert face["horizontal"] == "CENTER"
    assert face["vertical"] == "CENTER"


def test_zone_top():
    assert FaceTracker().update(make_face_landmarks(cy=0.1))["vertical"] == "TOP"


def test_zone_bottom():
    assert FaceTracker().update(make_face_landmarks(cy=0.9))["vertical"] == "BOTTOM"


# ── Looking direction ────────────────────────────────────────────

def test_looking_center():
    assert FaceTracker().update(make_face_landmarks())["looking"] == "CENTER"


def test_looking_right():
    face = FaceTracker().update(make_face_landmarks(nose_dx=0.2))
    assert face["looking"] == "RIGHT"
    assert face["yaw"] > 0


def test_looking_left():
    face = FaceTracker().update(make_face_landmarks(nose_dx=-0.2))
    assert face["looking"] == "LEFT"
    assert face["yaw"] < 0


def test_looking_down():
    face = FaceTracker().update(make_face_landmarks(nose_dy=0.25))
    assert face["looking"] == "DOWN"


def test_looking_up():
    face = FaceTracker().update(make_face_landmarks(nose_dy=-0.25))
    assert face["looking"] == "UP"


def test_looking_threshold_configurable():
    strict = FaceTracker(look_threshold=0.5)
    assert strict.update(make_face_landmarks(nose_dx=0.2))["looking"] == "CENTER"


# ── Proximity ────────────────────────────────────────────────────

def test_proximity_close():
    assert FaceTracker().update(make_face_landmarks(w=0.6, h=0.8))["proximity"] == "CLOSE"


def test_proximity_far():
    assert FaceTracker().update(make_face_landmarks(w=0.1, h=0.15))["proximity"] == "FAR"


def test_proximity_medium():
    assert FaceTracker().update(make_face_landmarks(w=0.3, h=0.4))["proximity"] == "MEDIUM"


# ── Structure ────────────────────────────────────────────────────

def test_output_structure():
    face = FaceTracker().update(make_face_landmarks(), confidence=1.0)
    for key in ("present", "confidence", "center", "box", "size",
                "proximity", "horizontal", "vertical", "yaw", "pitch", "looking"):
        assert key in face
    assert face["present"] is True
    assert face["confidence"] == 1.0
