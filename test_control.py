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
    e.update("THUMBS_DOWN", 0.9, "STATIONARY", 0.2)  # toggles master OFF
    assert e.enabled is False
    e.update("NONE", 0.0, "STATIONARY", 0.5)  # release
    assert e.update("THUMBS_DOWN", 0.9, "STATIONARY", 2.0) == "control_toggle"
    assert e.enabled is True


def test_master_off_blocks_all():
    e = eng(enabled=False)
    assert e.update("THUMBS_UP", 0.9, "STATIONARY", 0.0) is None
    assert e.update("POINT", 0.9, "RIGHT", 0.1) is None


def test_palm_enables_thumbs_disables_master():
    e = eng(enabled=False)
    assert e.update("OPEN_PALM", 0.9, "STATIONARY", 0.0) == "control_toggle"
    assert e.enabled is True
    e.update("NONE", 0.0, "STATIONARY", 1.0)  # release
    # palm while ON enters volume mode instead of disabling (hand_y None ok)
    assert e.update("OPEN_PALM", 0.9, "STATIONARY", 2.0) == ("volume_mode", True)
    assert e.enabled is True and e.volume_mode is True
    e.update("NONE", 0.0, "STATIONARY", 3.0)
    # thumbs-down toggles the master off (and kills volume mode)
    assert e.update("THUMBS_DOWN", 0.9, "STATIONARY", 4.0) == "control_toggle"
    assert e.enabled is False and e.volume_mode is False


MOTION_MAP = {"MOVE_RIGHT": "media_next", "MOVE_LEFT": "media_previous",
              "MOVE_UP": "volume_up", "MOVE_DOWN": "volume_down"}


def test_move_transition_fires_no_repeat():
    e = eng(action_map=MOTION_MAP)
    assert e.update("NONE", 0.0, "RIGHT", 0.0) == "media_next"
    assert e.update("NONE", 0.0, "RIGHT", 0.5) is None
    assert e.update("NONE", 0.0, "RIGHT", 5.0) is None
    assert e.update("NONE", 0.0, "LEFT", 6.0) == "media_previous"


def test_volume_ramps_while_sustained():
    # Repeat-kind was removed: one command per continuous movement.
    # Sustained motion in the same direction never refires.
    e = eng(action_map=MOTION_MAP)
    assert e.update("NONE", 0.0, "UP", 0.0) == "volume_up"   # transition
    assert e.update("NONE", 0.0, "UP", 0.2) is None
    assert e.update("NONE", 0.0, "UP", 5.0) is None
    e.update("NONE", 0.0, "STATIONARY", 6.0)  # pause re-arms
    assert e.update("NONE", 0.0, "UP", 7.0) == "volume_up"


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


def test_palm_enters_and_exits_volume_mode():
    e = eng()  # enabled
    assert e.volume_state == "INACTIVE"
    assert e.update("OPEN_PALM", 0.9, "STATIONARY", 0.0) == ("volume_mode", True)
    assert e.volume_state == "ACTIVE"
    e.update("NONE", 0.0, "STATIONARY", 1.0)  # release palm
    assert e.update("OPEN_PALM", 0.9, "STATIONARY", 2.0) == ("volume_mode", False)
    assert e.volume_mode is False
    assert e.enabled is True  # master untouched by volume lock
    assert e.get_volume_status()["state"] in ("LOCKING", "INACTIVE")


def test_palm_enables_master_but_not_volume():
    e = eng(enabled=False)
    assert e.update("OPEN_PALM", 0.9, "STATIONARY", 0.0) == "control_toggle"
    assert e.enabled is True and e.volume_mode is False


def test_height_up_raises_down_lowers():
    e = eng()
    e.update("OPEN_PALM", 0.9, "STATIONARY", 0.0, hand_y=0.5)
    seen, t = [], 0.5
    for _ in range(8):  # hold hand high: targets climb toward 100
        r = e.update("NONE", 0.0, "STATIONARY", t, hand_y=0.25)
        t += 0.5
        if r:
            seen.append(r[1])
    assert seen and seen == sorted(seen) and seen[-1] >= 90
    seen, t = [], t
    for _ in range(8):  # then low: targets fall toward 0
        r = e.update("NONE", 0.0, "STATIONARY", t, hand_y=0.75)
        t += 0.5
        if r:
            seen.append(r[1])
    assert seen and seen == sorted(seen, reverse=True) and seen[-1] <= 10


def test_still_hand_sends_nothing_more():
    e = eng()
    e.update("OPEN_PALM", 0.9, "STATIONARY", 0.0, hand_y=0.5)
    e.update("NONE", 0.0, "STATIONARY", 0.5, hand_y=0.5)
    assert e.update("NONE", 0.0, "STATIONARY", 1.0, hand_y=0.5) is None
    assert e.update("NONE", 0.0, "STATIONARY", 5.0, hand_y=0.5) is None


def test_tracking_lost_pauses_never_zeroes():
    e = eng()
    e.update("OPEN_PALM", 0.9, "STATIONARY", 0.0, hand_y=0.5)
    assert e.update("NONE", 0.0, "STATIONARY", 0.1,
                    hand_present=False) is None
    assert e.get_volume_status()["state"] == "TRACKING_LOST"
    # hand returns elsewhere: no jump-send, resumes smoothly
    assert e.update("NONE", 0.0, "STATIONARY", 0.2, hand_y=0.9) is None
    assert e.get_volume_status()["state"] == "ACTIVE"


def test_volume_clamped_to_valid_range():
    e = eng(vol_top=0.4, vol_bottom=0.6)
    e.update("OPEN_PALM", 0.9, "STATIONARY", 0.0, hand_y=0.5)
    assert e.update("NONE", 0.0, "STATIONARY", 0.5, hand_y=0.0) == ("volume_set", 100)
    r = None
    for i in range(6):  # converges, then clamps exactly at 0
        r = e.update("NONE", 0.0, "STATIONARY", 1.0 + i * 0.5, hand_y=1.0)
        if r == ("volume_set", 0):
            break
    assert r == ("volume_set", 0)


def test_no_volume_when_inactive():
    e = eng()
    # never entered: height motion alone changes nothing
    assert e.update("NONE", 0.0, "STATIONARY", 0.0, hand_y=0.1) is None
    assert e.get_volume_status() == {"state": "INACTIVE", "target": None,
                                     "top": 0.25, "bottom": 0.75,
                                     "deadband": 0.008}


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
