import pytest
from positions import get_finger_tips

def test_no_hand_returns_none():
    assert get_finger_tips([]) is None

def test_returns_5_tips():
    fake = [{"x": 0, "y": 0, "z": 0, "visibility": 1.0}] * 21
    tips = get_finger_tips(fake)
    assert len(tips) == 5
    assert all("x" in t and "y" in t and "conf" in t for t in tips)