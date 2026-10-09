"""Tests for multi-hand/multi-face input — synthetic landmarks only,
except the blank-image detector shape tests (shipped models, no webcam)."""

import pytest
from multi import HandSlots
from face_tracker import FaceTracker
from test_face import make_face_landmarks


def make_hand(offsets=None, base=0.5):
    lm = [{"x": base, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(21)]
    if offsets:
        for idx, val in offsets.items():
            lm[idx] = {**lm[idx], **val}
    return lm


def make_point_hand():
    """Index straight up, others folded, thumb folded at IP."""
    return make_hand({
        2: {"x": 0.42, "y": 0.60}, 3: {"x": 0.38, "y": 0.52}, 4: {"x": 0.38, "y": 0.52},
        5: {"x": 0.45, "y": 0.56}, 6: {"x": 0.45, "y": 0.40}, 8: {"x": 0.45, "y": 0.10},
        9: {"x": 0.50, "y": 0.56}, 10: {"x": 0.50, "y": 0.40}, 12: {"x": 0.50, "y": 0.40},
        13: {"x": 0.55, "y": 0.56}, 14: {"x": 0.55, "y": 0.40}, 16: {"x": 0.55, "y": 0.40},
        17: {"x": 0.58, "y": 0.56}, 18: {"x": 0.58, "y": 0.41}, 20: {"x": 0.58, "y": 0.41},
    })


def make_fist_hand():
    """All tips parked at their PIPs (degenerate = folded)."""
    return make_hand({
        2: {"x": 0.42, "y": 0.60}, 3: {"x": 0.38, "y": 0.52}, 4: {"x": 0.38, "y": 0.52},
        5: {"x": 0.45, "y": 0.56}, 6: {"x": 0.45, "y": 0.40}, 8: {"x": 0.45, "y": 0.40},
        9: {"x": 0.50, "y": 0.56}, 10: {"x": 0.50, "y": 0.40}, 12: {"x": 0.50, "y": 0.40},
        13: {"x": 0.55, "y": 0.56}, 14: {"x": 0.55, "y": 0.40}, 16: {"x": 0.55, "y": 0.40},
        17: {"x": 0.58, "y": 0.56}, 18: {"x": 0.58, "y": 0.41}, 20: {"x": 0.58, "y": 0.41},
    })


def hand_input(lm, handedness, confidence=0.9):
    return {"landmarks": lm, "handedness": handedness, "confidence": confidence}


# ── Slot tracking ────────────────────────────────────────────────

def test_two_hands_tracked_separately():
    slots = HandSlots(debounce_frames=1)
    entries = None
    for t in [0.0, 0.1, 0.2]:
        entries = slots.update([hand_input(make_point_hand(), "Right"),
                                hand_input(make_fist_hand(), "Left")], timestamp=t)
    assert len(entries) == 2
    assert entries[0]["hand"] == "Right"
    assert entries[0]["gesture"]["gesture"] == "POINT"
    assert entries[1]["hand"] == "Left"
    assert entries[1]["gesture"]["gesture"] == "FIST"


def test_smoothing_state_not_shared():
    """Moving the right hand must not move the left hand's output."""
    slots = HandSlots(smoothing=1.0)
    left = make_fist_hand()
    slots.update([hand_input(make_hand({8: {"x": 0.3}}), "Right"),
                  hand_input(left, "Left")], timestamp=0.0)
    entries = slots.update([hand_input(make_hand({8: {"x": 0.7}}), "Right"),
                            hand_input(left, "Left")], timestamp=0.1)
    assert entries[0]["fingers"]["index_tip"]["x"] == pytest.approx(0.7)
    assert entries[1]["movement"]["direction"] == "STATIONARY"


def test_debounce_per_hand():
    slots = HandSlots(debounce_frames=3)
    first = slots.update([hand_input(make_point_hand(), "Right")], timestamp=0.0)
    assert first[0]["gesture"]["gesture"] == "NONE"  # still debouncing
    last = None
    for t in [0.1, 0.2, 0.3]:
        last = slots.update([hand_input(make_point_hand(), "Right")], timestamp=t)
    assert last[0]["gesture"]["gesture"] == "POINT"


def test_duplicate_handedness_keeps_both():
    slots = HandSlots(debounce_frames=1)
    entries = slots.update([hand_input(make_point_hand(), "Right"),
                            hand_input(make_fist_hand(), "Right")], timestamp=0.0)
    assert len(entries) == 2
    assert len(slots) == 2


def test_empty_input_returns_empty():
    assert HandSlots().update([], timestamp=0.0) == []
    assert HandSlots().update(None, timestamp=0.0) == []


def test_detector_resets_after_prolonged_loss():
    slots = HandSlots(debounce_frames=3)
    for t in [0.0, 0.1, 0.2, 0.3]:
        slots.update([hand_input(make_fist_hand(), "Left")], timestamp=t)
    for i in range(20):  # left hand gone long enough to reset
        slots.update([], timestamp=0.4 + i * 0.1)
    back = slots.update([hand_input(make_fist_hand(), "Left")], timestamp=3.0)
    assert back[0]["gesture"]["gesture"] == "NONE"  # fresh debounce, not stale FIST


# ── Two faces ────────────────────────────────────────────────────

def test_two_faces_independent():
    # One tracker instance per face (FaceSlots does this in production);
    # smoothing state must never leak between faces.
    left = FaceTracker().update(make_face_landmarks(cx=0.25))
    right = FaceTracker().update(make_face_landmarks(cx=0.75))
    assert left["horizontal"] == "LEFT"
    assert right["horizontal"] == "RIGHT"


# ── Detector shapes on empty frames ──────────────────────────────

def test_detect_hands_empty_frame():
    import numpy as np
    from detect import detect_hands, detect_hand
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    assert detect_hands(blank) == {"hands": []}
    assert detect_hand(blank)["landmarks"] == []


def test_detect_faces_empty_frame():
    import numpy as np
    from face import detect_faces, detect_face
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    assert detect_faces(blank) == {"faces": []}
    assert detect_face(blank)["landmarks"] == []
