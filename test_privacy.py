"""Tests for Smart Privacy Mode — synthetic only, no camera."""

import pytest
from privacy import PrivacyController, PrivacyRules, MODES


BOX = {"x_min": 0.35, "y_min": 0.20, "x_max": 0.65, "y_max": 0.60}
FACE = {"box": BOX}


def test_invalid_mode_sanitized():
    assert PrivacyController(mode="BOGUS").mode == "OFF"
    assert PrivacyRules(blur_kind="BOGUS").blur_kind == "GAUSSIAN"


def test_toggle_off_to_last_and_back():
    p = PrivacyController()
    assert p.toggle_mode() == "FACE_BLUR"
    assert p.toggle_mode() == "OFF"


def test_toggle_remembers_last_blur_mode():
    p = PrivacyController()
    p.cycle_mode()  # FACE_BLUR
    p.cycle_mode()  # STRICT
    p.toggle_mode()  # -> OFF
    assert p.toggle_mode() == "STRICT"  # back to STRICT, not FACE_BLUR


def test_cycle_order():
    p = PrivacyController()
    seen = [p.cycle_mode() for _ in range(5)]
    assert seen == ["FACE_BLUR", "STRICT", "AUTO", "OFF", "FACE_BLUR"]


def test_off_never_blurs():
    p = PrivacyController(mode="OFF")
    r = p.evaluate([FACE], 0.0)
    assert r == {"active": False, "boxes": [], "expand": 0.0,
                 "kind": "GAUSSIAN", "strength": 5, "reason": "off"}


def test_face_blur_uses_fresh_boxes():
    p = PrivacyController(mode="FACE_BLUR")
    r = p.evaluate([FACE], 1.0)
    assert r["active"] is True and r["boxes"] == [BOX]
    assert r["expand"] == 0.15 and r["reason"] == "mode"


def test_face_blur_memory_covers_drop():
    p = PrivacyController(mode="FACE_BLUR")
    p.evaluate([FACE], 1.0)
    r = p.evaluate([], 1.5)  # within grace
    assert r["active"] is True and r["reason"] == "memory"
    r = p.evaluate([], 5.0)  # expired
    assert r["active"] is False and r["boxes"] == []


def test_strict_bigger_and_stronger():
    p = PrivacyController(mode="STRICT",
                           rules=PrivacyRules(strength=3, blur_kind="PIXELATE"))
    r = p.evaluate([FACE], 0.0)
    assert r["active"] is True
    assert r["expand"] == 0.60
    assert r["strength"] == 7  # floored up
    assert r["kind"] == "PIXELATE"


def test_auto_on_face_default():
    p = PrivacyController(mode="AUTO")
    assert p.evaluate([FACE], 0.0)["reason"] == "auto"
    assert p.evaluate([], 5.0)["reason"] == "auto-idle"


def test_auto_min_faces_rule():
    p = PrivacyController(mode="AUTO",
                           rules=PrivacyRules(auto_on_face=False, min_faces=2))
    assert p.evaluate([FACE], 0.0)["active"] is False
    assert p.evaluate([FACE, FACE], 0.1)["active"] is True


def test_auto_memory_while_engaged():
    p = PrivacyController(mode="AUTO")
    p.evaluate([FACE], 1.0)
    r = p.evaluate([], 1.2)
    assert r["active"] is True and r["reason"] == "auto-memory"


def test_pinch_rising_edge_flips_mode():
    p = PrivacyController(cooldown=1.0)
    assert p.update_pinch(False, 0.0) == "OFF"
    assert p.update_pinch(True, 0.1) == "FACE_BLUR"   # rising edge
    assert p.update_pinch(True, 0.2) == "FACE_BLUR"   # hold: nothing
    assert p.update_pinch(False, 0.3) == "FACE_BLUR"  # release: nothing
    assert p.update_pinch(True, 0.4) == "FACE_BLUR"   # cooldown blocks
    assert p.update_pinch(False, 0.5) == "FACE_BLUR"
    assert p.update_pinch(True, 2.0) == "OFF"         # release+cooldown → flip


def test_cycle_kind_toggles_style():
    p = PrivacyController()
    assert p.cycle_kind() == "PIXELATE"
    assert p.cycle_kind() == "GAUSSIAN"
