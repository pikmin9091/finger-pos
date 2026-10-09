"""Stability tests: spike rejection, slot identity across reorder/gaps,
pinch hysteresis, and smoothing convergence. Synthetic landmarks only,
except the VIDEO-mode repeat-call shape test (shipped models)."""

import pytest
from tracker import FingerTracker
from gesture import GestureDetector
from face_tracker import FaceTracker
from multi import HandSlots, FaceSlots
from test_multi import make_point_hand, make_fist_hand, hand_input
from test_face import make_face_landmarks
from test_gesture import make_landmarks as make_gesture_lm


def shifted(hand, dx):
    return [{**lm, "x": lm["x"] + dx} for lm in hand]


# ── Teleport guard ───────────────────────────────────────────────

def test_spike_clamped():
    t = FingerTracker(smoothing=1.0, max_jump=0.2)
    base = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(21)]
    t.update(base, timestamp=0.0)
    spike = [{"x": 0.9 if i == 8 else 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}
             for i in range(21)]
    out = t.update(spike, timestamp=0.1)
    # 0.4 jump clamped to 0.2 → output 0.7, not 0.9
    assert out["fingers"]["index_tip"]["x"] == pytest.approx(0.7)


def test_normal_motion_unaffected_by_guard():
    t = FingerTracker(smoothing=1.0)  # default max_jump=0.5
    base = [{"x": 0.3, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(21)]
    t.update(base, timestamp=0.0)
    moved = [{"x": 0.7 if i == 8 else 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}
             for i in range(21)]
    out = t.update(moved, timestamp=0.2)
    assert out["movement"]["velocity_x"] == pytest.approx(2.0)


# ── Slot identity across reorder ─────────────────────────────────

def test_reordered_hands_keep_identity():
    slots = HandSlots(debounce_frames=1)
    right = shifted(make_point_hand(), -0.2)  # POINT on the left side
    left = shifted(make_fist_hand(), 0.2)     # FIST on the right side
    for t in [0.0, 0.1]:
        slots.update([hand_input(right, "Right"), hand_input(left, "Left")],
                     timestamp=t)
    # Same hands, swapped input order — smoothing slots must not swap.
    entries = slots.update([hand_input(left, "Left"), hand_input(right, "Right")],
                           timestamp=0.2)
    by_pos = sorted(entries, key=lambda e: e["hand_center"]["x"])
    assert by_pos[0]["gesture"]["gesture"] == "POINT"
    assert by_pos[1]["gesture"]["gesture"] == "FIST"
    assert len(slots) == 2


def test_brief_gap_rejoins_slot():
    slots = HandSlots(debounce_frames=1)
    for t in [0.0, 0.1, 0.2]:
        slots.update([hand_input(make_point_hand(), "Right")], timestamp=t)
    slots.update([], timestamp=0.3)  # single dropped frame
    back = slots.update([hand_input(make_point_hand(), "Right")], timestamp=0.4)
    assert back[0]["gesture"]["gesture"] == "POINT"  # no debounce restart
    assert len(slots) == 1


def test_reordered_faces_keep_identity():
    slots = FaceSlots()
    slots.update([{"landmarks": make_face_landmarks(cx=0.3), "confidence": 1.0},
                  {"landmarks": make_face_landmarks(cx=0.7), "confidence": 1.0}])
    out = slots.update([{"landmarks": make_face_landmarks(cx=0.7), "confidence": 1.0},
                        {"landmarks": make_face_landmarks(cx=0.3), "confidence": 1.0}])
    assert [f["center"]["x"] for f in out] == pytest.approx([0.3, 0.7])
    assert len(slots) == 2


# ── Pinch hysteresis ─────────────────────────────────────────────

def _pinch_lm(dist):
    lm = make_gesture_lm({})
    lm[4] = {"x": 0.40, "y": 0.47, "z": 0.0}
    lm[8] = {"x": 0.40 + dist, "y": 0.47, "z": 0.0}
    return lm


def test_pinch_holds_inside_exit_band():
    d = GestureDetector(pinch_threshold=0.06, debounce_frames=1)
    assert d.detect(_pinch_lm(0.05))["gesture"] == "PINCH"   # enter
    assert d.detect(_pinch_lm(0.08))["gesture"] == "PINCH"   # 0.08 < 0.09 exit
    assert d.detect(_pinch_lm(0.20))["gesture"] != "PINCH"   # beyond exit band


def test_pinch_does_not_enter_inside_exit_band():
    d = GestureDetector(pinch_threshold=0.06, debounce_frames=1)
    assert d.detect(_pinch_lm(0.08))["gesture"] != "PINCH"   # 0.08 > 0.06 enter


# ── Smoothing convergence ────────────────────────────────────────

def test_face_smoothing_damps_jitter():
    t = FaceTracker(smoothing=0.5)
    t.update(make_face_landmarks(cx=0.5))
    out = t.update(make_face_landmarks(cx=0.6))
    assert out["center"]["x"] == pytest.approx(0.55)
    out2 = t.update(make_face_landmarks(cx=0.6))
    assert out2["center"]["x"] == pytest.approx(0.575)


def test_gesture_confidence_converges():
    d = GestureDetector(debounce_frames=1)
    last = None
    for _ in range(8):
        last = d.detect(make_gesture_lm({}))
    assert last["gesture"] == "FIST"
    assert last["confidence"] == pytest.approx(1.0)


# ── VIDEO-mode repeat calls ──────────────────────────────────────

def test_video_mode_repeat_calls():
    import numpy as np
    from detect import detect_hands
    from face import detect_faces
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(3):  # exercises monotonic timestamps
        assert detect_hands(blank) == {"hands": []}
        assert detect_faces(blank) == {"faces": []}
