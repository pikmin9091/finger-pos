"""Tests for multi-hand/multi-face input — synthetic landmarks only,
except the blank-image detector shape tests (shipped models, no webcam)."""

import pytest
from multi import HandSlots
from face_tracker import FaceTracker
from test_face import make_face_landmarks


def make_hand(offsets=None, dx=0.0):
    """Anatomically proportioned hand (palm length ~= 0.22), shifted by dx.

    All fingers folded (tips at PIPs), thumb across the palm — a FIST
    base that `offsets` can reshape. Matches test_gesture proportions.
    """
    pts = [
        (0.50, 0.72),  # 0  wrist
        (0.46, 0.66),  # 1  thumb_cmc
        (0.44, 0.60),  # 2  thumb_mcp
        (0.40, 0.56),  # 3  thumb_ip
        (0.47, 0.55),  # 4  thumb_tip (folded: across palm)
        (0.44, 0.50),  # 5  index_mcp
        (0.44, 0.41),  # 6  index_pip
        (0.44, 0.34),  # 7  index_dip
        (0.44, 0.41),  # 8  index_tip (folded)
        (0.50, 0.50),  # 9  middle_mcp
        (0.50, 0.41),  # 10 middle_pip
        (0.50, 0.34),  # 11 middle_dip
        (0.50, 0.41),  # 12 middle_tip (folded)
        (0.56, 0.50),  # 13 ring_mcp
        (0.56, 0.41),  # 14 ring_pip
        (0.56, 0.34),  # 15 ring_dip
        (0.56, 0.41),  # 16 ring_tip (folded)
        (0.615, 0.52),  # 17 pinky_mcp
        (0.615, 0.45),  # 18 pinky_pip
        (0.615, 0.40),  # 19 pinky_dip
        (0.615, 0.45),  # 20 pinky_tip (folded)
    ]
    lm = [{"x": x + dx, "y": y, "z": 0.0, "visibility": 1.0} for x, y in pts]
    if offsets:
        for idx, val in offsets.items():
            lm[idx] = {**lm[idx], **val}
    return lm


def make_point_hand(dx=0.0):
    """Index straight up, others folded, thumb across palm."""
    return make_hand({8: {"x": 0.44 + dx, "y": 0.26}}, dx=dx)


def make_fist_hand(dx=0.0):
    """All tips parked at their PIPs (degenerate = folded)."""
    return make_hand(dx=dx)


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


def test_set_confidence_clamps_and_switches_model():
    import numpy as np
    from detect import set_confidence as set_hand_conf, detect_hands
    from face import set_confidence as set_face_conf
    assert set_hand_conf(2.0, -1.0, 0.65) == (1.0, 0.0, 0.65)
    assert set_face_conf(9.0, 0.5, 0.5) == (1.0, 0.5, 0.5)
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    assert detect_hands(blank) == {"hands": []}  # works on alt triple
    set_hand_conf()
    set_face_conf()
    assert detect_hands(blank) == {"hands": []}  # and back on default


def test_set_smoothing_updates_slots_and_default():
    slots = HandSlots(smoothing=0.2)
    slots.update([hand_input(make_fist_hand(), "Right")], timestamp=0.0)
    slots.set_smoothing(0.9)
    assert slots.smoothing == 0.9
    assert slots._slots[0]["tracker"].smoothing == 0.9
    slots.set_smoothing(5.0)  # clamped
    assert slots._slots[0]["tracker"].smoothing == 1.0


def test_set_debounce_updates_slots_and_default():
    slots = HandSlots()
    slots.update([hand_input(make_fist_hand(), "Right")], timestamp=0.0)
    slots.set_debounce(7)
    assert slots.debounce_frames == 7
    assert slots._slots[0]["detector"].debounce_frames == 7
    slots.set_debounce(0)  # clamped to >= 1
    assert slots._slots[0]["detector"].debounce_frames == 1


def test_handedness_vote_stabilizes_flips():
    from multi import HandSlots
    from test_multi import make_fist_hand, hand_input
    slots = HandSlots(debounce_frames=1)
    labels = ["Right", "Left", "Right", "Left", "Right", "Right", "Right"]
    out = None
    for i, lab in enumerate(labels):
        out = slots.update([hand_input(make_fist_hand(), lab)], timestamp=i * 0.1)
    assert out[0]["hand"] == "Right"  # 4/7 majority wins over flicker
    assert slots._slots[0]["label"] == "Right"


def test_handedness_none_falls_back():
    from multi import HandSlots
    from test_multi import make_fist_hand, hand_input
    slots = HandSlots(debounce_frames=1)
    out = slots.update([hand_input(make_fist_hand(), None)], timestamp=0.0)
    assert out[0]["hand"] == "Unknown"
