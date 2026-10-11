"""Tests for the open-palm privacy toggle — synthetic only."""

from blur import PalmHoldToggle
from privacy import PrivacyController


def test_palm_hold_toggles_once():
    p = PalmHoldToggle()
    assert p.update(True, True, 0.0) is False
    assert p.update(True, True, 0.5) is True  # 0.5 s hold fires
    assert p.update(True, True, 5.0) is True  # holding never refires


def test_fist_never_toggles():
    p = PalmHoldToggle()
    for t in (0.0, 0.5, 1.0, 5.0):
        assert p.update(False, True, t) is False


def test_no_hand_never_toggles_but_persists():
    p = PalmHoldToggle()
    assert p.update(True, True, 0.0) is False
    assert p.update(True, True, 0.5) is True
    for t in (1.0, 2.0, 10.0):  # hand gone: state persists
        assert p.update(False, False, t) is True


def test_release_rearms_for_next_toggle():
    p = PalmHoldToggle()
    p.update(True, True, 0.0)
    assert p.update(True, True, 0.5) is True
    # quick re-press without a 0.5 s release: nothing happens
    p.update(False, True, 0.6)
    assert p.update(True, True, 0.7) is True
    # sustained release re-arms; the next hold toggles back off
    p.update(False, True, 2.0)
    assert p.update(False, True, 2.5) is True  # 0.5 s released: re-armed
    assert p.update(True, True, 2.6) is True  # hold starts, not yet fired
    assert p.update(True, True, 3.1) is False  # 0.5 s hold: toggles off


def test_hand_leaving_rearms_immediately():
    p = PalmHoldToggle()
    p.update(True, True, 0.0)
    assert p.update(True, True, 0.5) is True
    p.update(False, False, 0.6)  # hand left: re-armed at once
    assert p.update(True, True, 0.7) is True  # hold restarts
    assert p.update(True, True, 1.2) is False


def test_controller_palm_integration():
    c = PrivacyController(mode="OFF")
    c.update_palm(True, True, 0.0)
    assert c.update_palm(True, True, 0.5) == "FACE_BLUR"  # OFF -> last mode
    assert c.update_palm(False, False, 9.0) == "FACE_BLUR"  # persists, no hand
    # release, then hold again -> back OFF
    c.update_palm(False, True, 10.0)
    c.update_palm(False, True, 10.6)
    assert c.update_palm(True, True, 10.7) == "FACE_BLUR"  # hold restarts
    assert c.update_palm(True, True, 11.2) == "OFF"


def test_open_palm_both_hands():
    from gesture import GestureDetector
    from test_gesture import make_landmarks
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    for handed in ("Left", "Right"):
        d = GestureDetector(debounce_frames=1)
        r = None
        for _ in range(6):
            r = d.detect(make_landmarks({f: True for f in fingers}),
                         handedness=handed)
        assert r["gesture"] == "OPEN_PALM", handed
        assert r["confidence"] >= 0.5, handed
        assert all(v >= 0.5 for v in r["scores"].values()), handed


def test_fist_is_not_palm():
    from gesture import GestureDetector
    from test_gesture import make_landmarks
    d = GestureDetector(debounce_frames=1)
    r = None
    for _ in range(6):
        r = d.detect(make_landmarks({}))
    assert r["gesture"] == "FIST"
