import numpy as np
import pytest
from positions import get_finger_tips, draw_box, landmarks_box, UiToasts

def test_no_hand_returns_none():
    assert get_finger_tips([]) is None

def test_returns_5_tips():
    fake = [{"x": 0, "y": 0, "z": 0, "visibility": 1.0}] * 21
    tips = get_finger_tips(fake)
    assert len(tips) == 5
    assert all("x" in t and "y" in t and "conf" in t for t in tips)

def test_landmarks_box_tight():
    lm = [{"x": 0.4, "y": 0.4, "z": 0.0}, {"x": 0.6, "y": 0.7, "z": 0.0}]
    box = landmarks_box(lm, pad=0.0)
    assert box == {"x_min": 0.4, "y_min": 0.4, "x_max": 0.6, "y_max": 0.7}


def test_landmarks_box_empty():
    assert landmarks_box([]) is None
    assert landmarks_box(None) is None


def test_draw_box_changes_pixels():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_box(frame, {"x_min": 0.2, "y_min": 0.2, "x_max": 0.8, "y_max": 0.8},
             label="HAND")
    assert frame.sum() > 0


def test_draw_box_clamps_and_skips():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    before = frame.copy()
    # degenerate box → untouched
    draw_box(frame, {"x_min": 0.5, "y_min": 0.5, "x_max": 0.5, "y_max": 0.5})
    assert np.array_equal(frame, before)
    # wild coords → clamped, no crash
    draw_box(frame, {"x_min": -5.0, "y_min": -5.0, "x_max": 9.0, "y_max": 9.0})
    assert frame.sum() > 0
    # no box → untouched
    draw_box(frame, None)


def test_toasts_expire_and_cap():
    t = UiToasts(ttl=1.0, max_show=2)
    t.push("a", timestamp=0.0)
    t.push("b", timestamp=0.0)
    t.push("c", timestamp=0.0)
    assert [x for x, _ in t.visible(0.5)] == ["b", "c"]  # capped to last 2
    assert t.visible(5.0) == []  # expired


def test_toasts_draw():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t = UiToasts()
    t.push("NEXT TRACK", (0, 255, 255), 1.0)
    t.draw(frame, 1.0)
    assert frame.sum() > 0
    t.draw(np.zeros((480, 640, 3), dtype=np.uint8), 99.0)  # expired: no crash