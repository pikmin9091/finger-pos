"""Tests for gesture-controlled face blur — synthetic only, no webcam."""

import numpy as np
import pytest
from blur import BlurToggle, FaceBoxMemory, blur_faces
from multi import HandSlots
from test_multi import make_fist_hand, hand_input
from test_stability import _pinch_lm


BOX = {"x_min": 0.35, "y_min": 0.20, "x_max": 0.65, "y_max": 0.60}


# ── Toggle state machine ─────────────────────────────────────────

def test_starts_off():
    assert BlurToggle().update(False, 0.0) is False


def test_rising_edge_toggles():
    b = BlurToggle(cooldown=1.0)
    assert b.update(False, 0.0) is False
    assert b.update(True, 0.1) is True    # pinch → ON
    assert b.update(True, 0.2) is True    # holding → no retrigger
    assert b.update(True, 5.0) is True    # holding long → still no retrigger


def test_release_then_pinch_toggles_back():
    b = BlurToggle(cooldown=1.0)
    b.update(True, 0.1)
    assert b.enabled is True
    b.update(False, 0.2)                  # release
    assert b.update(True, 0.3) is True    # too fast: cooldown blocks
    assert b.update(False, 0.4) is True   # release again, still ON
    assert b.update(True, 2.0) is False   # cooldown elapsed → OFF


def test_persists_without_hands():
    b = BlurToggle()
    b.update(True, 0.1)
    for t in [0.2, 0.3, 1.0, 5.0]:
        assert b.update(False, t) is True  # hand gone → stays ON


# ── End-to-end via HandSlots ─────────────────────────────────────

def _feed_pinch(slots, n, t0, dt=0.1):
    out = None
    for i in range(n):
        out = slots.update([hand_input(_pinch_lm(0.05), "Right")], timestamp=t0 + i * dt)
    return out


def test_pinch_toggles_blur_once_end_to_end():
    from blur import BlurToggle as BT
    slots = HandSlots()  # defaults: debounce 3, stable 5
    blur = BT(cooldown=0.0)
    enabled = []
    t = 0.0
    for _ in range(12):  # hold pinch well past stable
        entries = slots.update([hand_input(_pinch_lm(0.05), "Right")], timestamp=t)
        g = entries[0]["gesture"]
        enabled.append(blur.update(g["gesture"] == "PINCH" and g["stable"], t))
        t += 0.1
    assert any(enabled) and enabled[-1] is True  # exactly one rising edge
    assert enabled.count(True) == len(enabled) - enabled.index(True)


def test_release_and_pinch_toggles_off():
    slots = HandSlots()
    blur = BlurToggle(cooldown=0.0)
    t = 0.0
    for _ in range(10):
        e = slots.update([hand_input(_pinch_lm(0.05), "Right")], timestamp=t)
        blur.update(e[0]["gesture"]["gesture"] == "PINCH", t)
        t += 0.1
    assert blur.enabled is True
    for _ in range(10):  # release to fist
        e = slots.update([hand_input(make_fist_hand(), "Right")], timestamp=t)
        blur.update(e[0]["gesture"]["gesture"] == "PINCH", t)
        t += 0.1
    assert blur.enabled is True  # release alone never toggles
    for _ in range(10):
        e = slots.update([hand_input(_pinch_lm(0.05), "Right")], timestamp=t)
        blur.update(e[0]["gesture"]["gesture"] == "PINCH", t)
        t += 0.1
    assert blur.enabled is False


# ── Box memory ───────────────────────────────────────────────────

def test_memory_grace_period():
    m = FaceBoxMemory(grace=0.75)
    m.update([BOX], 1.0)
    assert m.boxes(1.5) == [BOX]
    assert m.boxes(2.0) == []  # expired


def test_memory_ignores_empty_update():
    m = FaceBoxMemory()
    m.update([BOX], 1.0)
    m.update([], 1.2)  # detection drop must not erase memory
    assert m.boxes(1.3) == [BOX]


# ── Blur rendering ───────────────────────────────────────────────

def test_blur_changes_face_only():
    img = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
    before = img.copy()
    n = blur_faces(img, [BOX])
    assert n == 1
    assert not np.array_equal(img[60:120, 80:120], before[60:120, 80:120])  # inside
    assert np.array_equal(img[0, 0], before[0, 0])                          # outside


def test_blur_empty_and_tiny():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert blur_faces(img, []) == 0
    assert blur_faces(img, None) == 0
    tiny = {"x_min": 0.5, "y_min": 0.5, "x_max": 0.51, "y_max": 0.51}
    assert blur_faces(img, [tiny]) == 0
