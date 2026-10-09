"""Tests for the gesture control engine — synthetic only."""

import pytest
from control import ControlEngine, ALLOWLIST, DEFAULT_MAP


def eng(**kw):
    kw.setdefault("enabled", True)
    return ControlEngine(**kw)


def test_open_palm_fires_once_on_edge():
    e = eng()
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 0.0) == "media_play_pause"
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 0.1) is None  # held: silent
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 5.0) is None


def test_gesture_change_fires_again_after_cooldown():
    e = eng()
    e.update("THUMBS_UP", 0.9, "STATIONARY", 0.0)
    e.update("OPEN_PALM", 0.9, "STATIONARY", 0.2)  # toggles master OFF
    assert e.enabled is False
    e.update("NONE", 0.0, "STATIONARY", 0.5)  # release
    assert e.update("OPEN_PALM", 0.9, "STATIONARY", 2.0) == "control_toggle"
    assert e.enabled is True


def test_master_off_blocks_all():
    e = eng(enabled=False)
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 0.0) is None
    assert e.update("POINT", 0.9, "RIGHT", 0.1) is None


def test_fist_toggles_master():
    e = eng(enabled=True)
    assert e.update("OPEN_PALM", 0.9, "STATIONARY", 0.0) == "control_toggle"
    assert e.enabled is False
    e.update("NONE", 0.0, "STATIONARY", 1.0)  # release
    assert e.update("OPEN_PALM", 0.9, "STATIONARY", 2.0) == "control_toggle"
    assert e.enabled is True


MOTION_MAP = {"MOVE_RIGHT": "media_next", "MOVE_LEFT": "media_previous",
              "MOVE_UP": "volume_up", "MOVE_DOWN": "volume_down"}


def test_move_transition_fires_no_repeat():
    e = eng(action_map=MOTION_MAP)
    assert e.update("NONE", 0.0, "RIGHT", 0.0) == "media_next"
    assert e.update("NONE", 0.0, "RIGHT", 0.5) is None
    assert e.update("NONE", 0.0, "RIGHT", 5.0) is None
    assert e.update("NONE", 0.0, "LEFT", 6.0) == "media_previous"


def test_volume_ramps_while_sustained():
    e = eng(volume_interval=0.5, action_map=MOTION_MAP)
    assert e.update("NONE", 0.0, "UP", 0.0) == "volume_up"   # transition
    assert e.update("NONE", 0.0, "UP", 0.2) is None           # too soon
    assert e.update("NONE", 0.0, "UP", 0.6) == "volume_up"    # interval elapsed
    assert e.update("NONE", 0.0, "STATIONARY", 1.2) is None


def test_low_confidence_blocked():
    e = eng(min_confidence=0.5)
    assert e.update("POINT", 0.3, "STATIONARY", 0.0) is None


def test_none_and_stationary_silent():
    e = eng()
    assert e.update(None, 0.0, "STATIONARY", 0.0) is None
    assert e.update("NONE", 0.0, "STATIONARY", 0.1) is None


def test_custom_map_validated():
    e = eng(action_map={"POINT": "media_next", "FIST": "rm -rf /"})
    assert e.update("POINT", 0.9, "STATIONARY", 0.0) == "media_next"
    assert e.update("FIST", 0.9, "STATIONARY", 1.0) == "media_previous"  # default kept


def test_thumbs_and_peace_defaults():
    e = eng()
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 0.0) == "media_play_pause"
    assert e.update("PEACE", 0.9, "STATIONARY", 2.0) == "media_next"


def test_labels():
    e = eng()
    assert e.label("media_next") == "NEXT TRACK"
    assert set(ALLOWLIST) >= set(DEFAULT_MAP.values())


def test_neutral_reset_required():
    e = eng()  # dwell 0
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 0.0) == "media_play_pause"
    # same gesture re-appears without neutral: no refire even past cooldown
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 5.0) is None
    e.update("NONE", 0.0, "STATIONARY", 6.0)  # neutral observed
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 7.0) == "media_play_pause"


def test_neutral_other_gesture_clears():
    e = eng()
    e.update("THUMBS_UP", 0.9, "STATIONARY", 0.0)
    e.update("PEACE", 0.9, "STATIONARY", 2.0)  # different gesture = neutral
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 4.0) == "media_play_pause"


def test_pinch_toggles_volume_mode():
    e = eng()
    assert e.volume_mode is False
    assert e.update("NONE", 0.0, "STATIONARY", 0.0, pinch=True) == ("volume_mode", True)
    assert e.update("NONE", 0.0, "STATIONARY", 0.1, pinch=True) is None  # held
    e.update("NONE", 0.0, "STATIONARY", 0.2, pinch=False)  # release
    assert e.update("NONE", 0.0, "STATIONARY", 0.3, pinch=True) == ("volume_mode", False)


def test_pinch_ignored_when_master_off():
    e = eng(enabled=False)
    assert e.update("NONE", 0.0, "STATIONARY", 0.0, pinch=True) is None
    assert e.volume_mode is False


def test_volume_adjust_deadzone_smooth_throttle():
    e = eng(dwell=0.0)
    e.update("NONE", 0.0, "STATIONARY", 0.0, pinch=True)  # enter
    assert e.volume_mode is True
    first = e.update("NONE", 0.0, "STATIONARY", 0.1, pinch_ratio=0.5)
    assert first[0] == "volume_set"  # first ratio seeds + sends
    pct0 = first[1]
    # tiny jitter inside deadzone band → no new target
    assert e.update("NONE", 0.0, "STATIONARY", 0.6, pinch_ratio=0.5) is None
    # big spread → new target after throttle window
    big = e.update("NONE", 0.0, "STATIONARY", 1.2, pinch_ratio=0.9)
    assert big[0] == "volume_set" and big[1] > pct0


def test_volume_mode_suppresses_discrete():
    e = eng()
    e.update("NONE", 0.0, "STATIONARY", 0.0, pinch=True)
    # thumbs up while adjusting → swallowed, volume mode stays
    r = e.update("THUMBS_UP", 0.9, "STATIONARY", 0.5, pinch_ratio=0.5)
    assert r is None or (isinstance(r, tuple) and r[0] == "volume_set")
    assert r != "media_play_pause"
    assert e.volume_mode is True


def test_dwell_requires_hold():
    e = eng(dwell=1.0)
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 0.0) is None  # too early
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 0.5) is None
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 1.0) == "media_play_pause"
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 1.5) is None  # fired once


def test_dwell_cancels_on_change_of_mind():
    e = eng(dwell=1.0)
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 0.0) is None
    assert e.update("PEACE", 0.9, "STATIONARY", 0.5) is None  # switched early
    assert e.update("PEACE", 0.9, "STATIONARY", 2.0) == "media_next"


def test_no_motion_disables_swipes_but_keeps_gestures():
    e = eng(motion=False)
    assert e.update("NONE", 0.0, "RIGHT", 0.0) is None
    assert e.update("NONE", 0.0, "UP", 1.0) is None
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 2.0) == "media_play_pause"


def test_calm_preset():
    e = eng(preset="calm")
    assert e.min_confidence == 0.8 and e.dwell == 1.0 and e.motion is False
    assert e.update("POINT", 0.7, "STATIONARY", 0.0) is None  # below 0.8
    assert e.update("POINT", 0.9, "STATIONARY", 1.0) is None  # dwell not met
    assert e.update("POINT", 0.9, "STATIONARY", 2.0) == "mute_toggle"
