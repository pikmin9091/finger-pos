import numpy as np
import pytest
from positions import (get_finger_tips, draw_box, landmarks_box, UiToasts,
                       layout_buttons, hit_test, draw_buttons, draw_help)

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

def _btns():
    return [("PRIV:OFF", ord("m")), ("BLUR:X", ord("b")), ("X", None)]


def test_layout_buttons_fit_frame():
    rects = layout_buttons(640, 480, _btns())
    assert len(rects) == 3
    for (x1, y1, x2, y2, _label, _key) in rects:
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480


def test_layout_buttons_tiny_frame_no_crash():
    rects = layout_buttons(120, 100, _btns())
    assert len(rects) == 3


def test_hit_test_inside_outside_and_none_key():
    rects = layout_buttons(640, 480, _btns())
    x1, y1, x2, y2, _label, key = rects[0]
    assert hit_test(rects, (x1 + x2) // 2, (y1 + y2) // 2) == ord("m")
    assert hit_test(rects, 320, 100) is None  # above the bar
    xb = [r for r in rects if r[5] is None][0]
    assert hit_test(rects, (xb[0] + xb[2]) // 2, (xb[1] + xb[3]) // 2) is None


def test_draw_buttons_and_help_change_pixels():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_buttons(frame, layout_buttons(640, 480, _btns()), {ord("m")})
    assert frame.sum() > 0
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_help(frame2)
    assert frame2.sum() > 0
    frame3 = np.zeros((120, 100, 3), dtype=np.uint8)
    draw_help(frame3)  # tiny frame: clips, no crash
