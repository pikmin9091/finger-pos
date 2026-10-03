"""Tests for FingerTracker — synthetic landmarks only, no webcam needed."""

import pytest
from tracker import FingerTracker, TIP_MAP, PALM_INDICES


# ── Fixtures ────────────────────────────────────────────────────────────────

def make_landmarks(offsets=None):
    """Create 21 synthetic landmarks. offsets: dict {index: {"x":..., "y":..., "z":...}}."""
    landmarks = []
    for i in range(21):
        landmarks.append({"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0})
    if offsets:
        for idx, val in offsets.items():
            landmarks[idx] = {**landmarks[idx], **val}
    return landmarks


# ── Fingertip extraction ────────────────────────────────────────────────────

def test_fingertip_extraction_keys():
    """All 5 fingertip names must be present in output."""
    tracker = FingerTracker()
    lm = make_landmarks()
    result = tracker.update(lm, timestamp=0.0)
    expected = {"thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"}
    assert set(result["fingers"].keys()) == expected


def test_fingertip_coordinates_from_landmarks():
    """Tip coordinates must match the landmark at the correct index."""
    tracker = FingerTracker()
    lm = make_landmarks({4: {"x": 0.1, "y": 0.2, "z": -0.1}})
    result = tracker.update(lm, timestamp=0.0)
    assert result["fingers"]["thumb_tip"]["x"] == 0.1
    assert result["fingers"]["thumb_tip"]["y"] == 0.2
    assert result["fingers"]["thumb_tip"]["z"] == -0.1


# ── Normalized coordinates ──────────────────────────────────────────────────

def test_normalized_range():
    """All x, y must be in 0.0-1.0, z must be present."""
    tracker = FingerTracker()
    lm = make_landmarks()
    result = tracker.update(lm, timestamp=0.0)
    for name, tip in result["fingers"].items():
        assert 0.0 <= tip["x"] <= 1.0, f"{name}.x={tip['x']}"
        assert 0.0 <= tip["y"] <= 1.0, f"{name}.y={tip['y']}"
        assert "z" in tip


def test_z_can_be_negative():
    """z is relative to camera plane — negative is toward camera."""
    tracker = FingerTracker()
    lm = make_landmarks({8: {"z": -0.5}})
    result = tracker.update(lm, timestamp=0.0)
    assert result["fingers"]["index_tip"]["z"] == -0.5


# ── Position classification ─────────────────────────────────────────────────

def test_classify_left():
    tracker = FingerTracker(pos_threshold=0.3)
    lm = make_landmarks({8: {"x": 0.1}})
    result = tracker.update(lm, timestamp=0.0)
    assert result["fingers"]["index_tip"]["horizontal"] == "LEFT"


def test_classify_right():
    tracker = FingerTracker(pos_threshold=0.3)
    lm = make_landmarks({8: {"x": 0.9}})
    result = tracker.update(lm, timestamp=0.0)
    assert result["fingers"]["index_tip"]["horizontal"] == "RIGHT"


def test_classify_center_horizontal():
    tracker = FingerTracker(pos_threshold=0.3)
    lm = make_landmarks({8: {"x": 0.5}})
    result = tracker.update(lm, timestamp=0.0)
    assert result["fingers"]["index_tip"]["horizontal"] == "CENTER"


def test_classify_top():
    tracker = FingerTracker(pos_threshold=0.3)
    lm = make_landmarks({8: {"y": 0.1}})
    result = tracker.update(lm, timestamp=0.0)
    assert result["fingers"]["index_tip"]["vertical"] == "TOP"


def test_classify_bottom():
    tracker = FingerTracker(pos_threshold=0.3)
    lm = make_landmarks({8: {"y": 0.9}})
    result = tracker.update(lm, timestamp=0.0)
    assert result["fingers"]["index_tip"]["vertical"] == "BOTTOM"


def test_classify_center_vertical():
    tracker = FingerTracker(pos_threshold=0.3)
    lm = make_landmarks({8: {"y": 0.5}})
    result = tracker.update(lm, timestamp=0.0)
    assert result["fingers"]["index_tip"]["vertical"] == "CENTER"


def test_classify_threshold_configurable():
    """Different threshold changes classification boundary."""
    tight = FingerTracker(pos_threshold=0.1)
    loose = FingerTracker(pos_threshold=0.5)
    lm = make_landmarks({8: {"x": 0.35}})  # outside tight threshold, inside loose
    r1 = tight.update(lm, timestamp=0.0)
    r2 = loose.update(lm, timestamp=0.0)
    assert r1["fingers"]["index_tip"]["horizontal"] == "LEFT"
    assert r2["fingers"]["index_tip"]["horizontal"] == "CENTER"


# ── Hand center ─────────────────────────────────────────────────────────────

def test_hand_center_calculation():
    tracker = FingerTracker()
    lm = make_landmarks()
    result = tracker.update(lm, timestamp=0.0)
    center = result["hand_center"]
    assert "x" in center and "y" in center and "z" in center
    assert center["x"] == 0.5 and center["y"] == 0.5


def test_hand_center_from_palm_only():
    """Hand center uses landmarks 0-9, not fingertips."""
    tracker = FingerTracker()
    lm = make_landmarks()
    # Shift palm landmarks only
    for i in range(10):
        lm[i]["x"] = 0.3
    result = tracker.update(lm, timestamp=0.0)
    assert result["hand_center"]["x"] == 0.3


# ── Movement tracking ───────────────────────────────────────────────────────

def test_movement_stationary_first_frame():
    """First frame always reports STATIONARY (no previous data)."""
    tracker = FingerTracker()
    lm = make_landmarks()
    result = tracker.update(lm, timestamp=0.0)
    assert result["movement"]["direction"] == "STATIONARY"
    assert result["movement"]["velocity_x"] == 0.0
    assert result["movement"]["velocity_y"] == 0.0


def test_movement_right():
    """Moving index tip right → direction RIGHT."""
    tracker = FingerTracker()
    lm1 = make_landmarks({8: {"x": 0.3}})
    tracker.update(lm1, timestamp=0.0)
    lm2 = make_landmarks({8: {"x": 0.7}})
    result = tracker.update(lm2, timestamp=0.1)
    assert result["movement"]["direction"] == "RIGHT"


def test_movement_left():
    tracker = FingerTracker()
    lm1 = make_landmarks({8: {"x": 0.7}})
    tracker.update(lm1, timestamp=0.0)
    lm2 = make_landmarks({8: {"x": 0.3}})
    result = tracker.update(lm2, timestamp=0.1)
    assert result["movement"]["direction"] == "LEFT"


def test_movement_up():
    tracker = FingerTracker()
    lm1 = make_landmarks({8: {"y": 0.7}})
    tracker.update(lm1, timestamp=0.0)
    lm2 = make_landmarks({8: {"y": 0.3}})
    result = tracker.update(lm2, timestamp=0.1)
    assert result["movement"]["direction"] == "UP"


def test_movement_down():
    tracker = FingerTracker()
    lm1 = make_landmarks({8: {"y": 0.3}})
    tracker.update(lm1, timestamp=0.0)
    lm2 = make_landmarks({8: {"y": 0.7}})
    result = tracker.update(lm2, timestamp=0.1)
    assert result["movement"]["direction"] == "DOWN"


def test_movement_velocity_calculation():
    """Velocity = delta / dt. Moving 0.4 in x over 0.2s → vx=2.0 (raw, no smoothing)."""
    tracker = FingerTracker(smoothing=1.0)  # no smoothing for exact velocity
    lm1 = make_landmarks({8: {"x": 0.3}})
    tracker.update(lm1, timestamp=0.0)
    lm2 = make_landmarks({8: {"x": 0.7}})
    result = tracker.update(lm2, timestamp=0.2)
    assert abs(result["movement"]["velocity_x"] - 2.0) < 0.01


def test_stationary_no_movement():
    """Same position → STATIONARY."""
    tracker = FingerTracker()
    lm1 = make_landmarks({8: {"x": 0.5}})
    tracker.update(lm1, timestamp=0.0)
    lm2 = make_landmarks({8: {"x": 0.5}})
    result = tracker.update(lm2, timestamp=0.1)
    assert result["movement"]["direction"] == "STATIONARY"


def test_zero_time_diff_no_crash():
    """dt=0 must not crash, returns STATIONARY."""
    tracker = FingerTracker()
    lm1 = make_landmarks({8: {"x": 0.3}})
    tracker.update(lm1, timestamp=0.0)
    lm2 = make_landmarks({8: {"x": 0.7}})
    result = tracker.update(lm2, timestamp=0.0)  # same timestamp
    assert result["movement"]["direction"] == "STATIONARY"


def test_very_small_time_diff_no_crash():
    """dt < 0.001 must not crash."""
    tracker = FingerTracker()
    lm1 = make_landmarks({8: {"x": 0.3}})
    tracker.update(lm1, timestamp=0.0)
    lm2 = make_landmarks({8: {"x": 0.7}})
    result = tracker.update(lm2, timestamp=0.0001)
    assert result["movement"]["direction"] == "STATIONARY"


# ── Smoothing ───────────────────────────────────────────────────────────────

def test_smoothing_reduces_jitter():
    """EMA smoothing must reduce variance between raw and smoothed."""
    tracker = FingerTracker(smoothing=0.3)
    # Alternating positions
    lm1 = make_landmarks({8: {"x": 0.3}})
    tracker.update(lm1, timestamp=0.0)
    lm2 = make_landmarks({8: {"x": 0.7}})
    result = tracker.update(lm2, timestamp=0.1)
    # Smoothed value should be between raw (0.7) and prev smoothed (0.3)
    smoothed_x = result["fingers"]["index_tip"]["x"]
    assert 0.3 < smoothed_x < 0.7


def test_smoothing_first_frame_equals_raw():
    """First frame: smoothed == raw (no previous data)."""
    tracker = FingerTracker(smoothing=0.3)
    lm = make_landmarks({8: {"x": 0.4}})
    result = tracker.update(lm, timestamp=0.0)
    assert result["fingers"]["index_tip"]["x"] == 0.4


def test_smoothing_preserves_raw_in_output():
    """Tracker output includes smoothed; raw landmarks are passed in separately."""
    tracker = FingerTracker(smoothing=0.5)
    lm_raw = make_landmarks({8: {"x": 0.1}})
    result1 = tracker.update(lm_raw, timestamp=0.0)
    lm_raw2 = make_landmarks({8: {"x": 0.9}})
    result2 = tracker.update(lm_raw2, timestamp=0.1)
    # result2 smoothed should be between 0.1 and 0.9
    smoothed = result2["fingers"]["index_tip"]["x"]
    assert 0.1 < smoothed < 0.9


# ── Integration ─────────────────────────────────────────────────────────────

def test_full_output_structure():
    """Output must contain hand, confidence, fingers, hand_center, movement."""
    tracker = FingerTracker()
    lm = make_landmarks({8: {"x": 0.7, "y": 0.3}})
    result = tracker.update(lm, handedness="Right", confidence=0.95, timestamp=1.0)
    assert "hand" in result
    assert "confidence" in result
    assert "fingers" in result
    assert "hand_center" in result
    assert "movement" in result
    assert result["hand"] == "Right"
    assert result["confidence"] == 0.95


def test_no_hand_landmarks_empty():
    """Empty landmarks → STATIONARY, no crash."""
    tracker = FingerTracker()
    result = tracker.update([], timestamp=0.0)
    assert result["movement"]["direction"] == "STATIONARY"