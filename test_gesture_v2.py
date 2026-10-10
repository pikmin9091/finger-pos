"""Tests for scoring classifier: UNKNOWN rejection, guards, debug fields."""

import pytest
from gesture import GestureDetector
from test_gesture import make_landmarks
from eval_gestures import half_bent_index


def detect_n(lm, n=6, **kw):
    d = GestureDetector(debounce_frames=1, **kw)
    r = None
    for _ in range(n):
        r = d.detect(lm)
    return r


def test_truncated_landmarks_rejected():
    d = GestureDetector()
    r = d.detect([{"x": 0.5, "y": 0.5, "z": 0.0}] * 10)
    assert r["gesture"] == "NONE"
    assert r["state"] == "UNKNOWN" and r["reason"] == "truncated"


def test_empty_reason():
    assert GestureDetector().detect([])["reason"] == "empty"


def test_weak_pattern_gate_unit():
    # PEACE rule matches (both binary-extended) but joint pattern
    # 0.7*0.7=0.49 < MIN_SCORE → rejected, not forced.
    d = GestureDetector()
    extended = {"thumb": False, "index": True, "middle": True,
                "ring": False, "pinky": False}
    scores = {"thumb": 0.0, "index": 0.7, "middle": 0.7,
              "ring": 0.0, "pinky": 0.0}
    lm = make_landmarks({"index": True, "middle": True})
    gesture, reason, _ = d._classify(extended, scores, False, 9.0, lm)
    assert (gesture, reason) == ("NONE", "weak-pattern")


def test_strong_pattern_accepted_unit():
    d = GestureDetector()
    extended = {"thumb": False, "index": True, "middle": True,
                "ring": False, "pinky": False}
    scores = {"thumb": 0.0, "index": 1.0, "middle": 1.0,
              "ring": 0.0, "pinky": 0.0}
    lm = make_landmarks({"index": True, "middle": True})
    gesture, reason, _ = d._classify(extended, scores, False, 9.0, lm)
    assert (gesture, reason) == ("PEACE", "clear")


def test_weak_pattern_rejected_as_unknown():
    r = detect_n(half_bent_index())
    assert r["gesture"] == "NONE"
    # Three half-curled fingers match no rule at all (PEACE needs ring
    # folded, POINT needs middle folded) — honest no-match rejection.
    assert r["reason"] == "no-match"
    # Stable NONE still reports CONFIRMED: the filter confirmed absence.
    assert r["state"] == "CONFIRMED"


def test_strong_poses_accepted_with_clear_reason():
    r = detect_n(make_landmarks({"index": True}))
    assert (r["gesture"], r["reason"]) == ("POINT", "clear")
    assert r["state"] == "CONFIRMED" and r["stable"] is True


def test_candidate_state_before_stable():
    d = GestureDetector(debounce_frames=3, stable_frames=5)
    r = d.detect(make_landmarks({"index": True}))
    assert r["state"] == "CANDIDATE"
    assert r["gesture"] == "NONE"  # debounced, not yet switched


def test_debug_fields_present():
    r = detect_n(make_landmarks({}))
    assert set(("gesture", "confidence", "stable", "state", "reason",
                "raw", "scores")) <= set(r)
    assert set(r["scores"]) == {"thumb", "index", "middle", "ring", "pinky"}
    assert r["raw"] == "FIST"


def test_scores_grade_extension():
    d = GestureDetector()
    d.detect(make_landmarks({"index": True}))
    _, scores = d._get_extended_fingers(make_landmarks({"index": True}))
    assert scores["index"] == 1.0
    assert scores["middle"] == 0.0


def test_mirror_invariance_thumbs_and_fist():
    # Mirrored geometry (other hand) must read identically; handedness
    # label never influences classification.
    for handed in ("Left", "Right"):
        lm = make_landmarks({"thumb": True})
        lm[4] = {"x": 0.34, "y": 0.40, "z": 0.0}  # up-and-out, clear of index
        mir = [{**p, "x": 1.0 - p["x"]} for p in lm]
        d = GestureDetector(debounce_frames=1)
        d2 = GestureDetector(debounce_frames=1)
        for _ in range(4):
            r = d.detect(lm, handedness=handed)
            rm = d2.detect(mir, handedness=handed)
        assert r["gesture"] == rm["gesture"] == "THUMBS_UP"
    for handed in ("Left", "Right"):
        d = GestureDetector(debounce_frames=1)
        for _ in range(4):
            r = d.detect(make_landmarks({}), handedness=handed)
        assert r["gesture"] == "FIST"


def test_thumbs_down_anatomical():
    d = GestureDetector(debounce_frames=1)
    lm = make_landmarks({"thumb": True})
    lm[4] = {"x": 0.46, "y": 0.82, "z": 0.0}
    for _ in range(4):
        r = d.detect(lm)
    assert r["gesture"] == "THUMBS_DOWN"


def test_curl_guard_folds_noisy_coincident_joints():
    # tip parked at pip (+epsilon noise scale): curled, not 50/50.
    lm = make_landmarks({})
    lm[8] = {"x": 0.451, "y": 0.401, "z": 0.0}
    r = detect_n(lm)
    assert r["gesture"] == "FIST"


def test_calib_overrides_thresholds(tmp_path):
    from calib import summarize, save_calib, load_calib
    from gesture import GestureDetector as GD
    det = GD()
    samples = {"open": [make_landmarks(
        {f: True for f in ["thumb", "index", "middle", "ring", "pinky"]})
        for _ in range(8)],
        "fist": [make_landmarks({}) for _ in range(8)],
        "pinch": [[dict(p) for p in make_landmarks({})] for _ in range(8)]}
    for lm in samples["pinch"]:
        lm[4] = {"x": 0.40, "y": 0.47, "z": 0.0}
        lm[8] = {"x": 0.45, "y": 0.47, "z": 0.0}
    calib, warnings = summarize(det, samples)
    assert calib and not warnings
    assert calib["thumb_extend"] > calib["thumb_fold"] > 0
    assert calib["pinch_threshold"] > 0
    p = save_calib(calib, path := str(tmp_path / "c.json"))
    assert load_calib(p)["pinch_threshold"] == calib["pinch_threshold"]
    d2 = GD()
    d2.apply_calib(calib)
    assert d2.pinch_threshold == calib["pinch_threshold"]


def test_calib_rejects_sparse_samples():
    from calib import summarize
    from gesture import GestureDetector as GD
    calib, warnings = summarize(GD(), {"open": [], "fist": [], "pinch": []})
    assert calib == {} and warnings


def test_calib_load_missing_is_empty(tmp_path):
    from calib import load_calib
    assert load_calib(str(tmp_path / "nope.json")) == {}


def test_hand_slots_accepts_calib():
    from multi import HandSlots
    s = HandSlots(calib={"pinch_threshold": 0.09})
    s.update([{"landmarks": make_landmarks({"index": True}),
               "handedness": "Right", "confidence": 0.9}], timestamp=0.0)
    assert len(s) == 1


def test_cooldown_remaining():
    from control import ControlEngine
    e = ControlEngine()
    assert e.cooldown_remaining(0.0) == 0.0
    e.update("THUMBS_UP", 0.9, "STATIONARY", 0.0)
    assert e.cooldown_remaining(0.1) > 0.0
    assert e.cooldown_remaining(5.0) == 0.0


def test_debug_info_uses_real_pipeline_values():
    import main as main_mod
    from multi import HandSlots
    slots = HandSlots(debounce_frames=1)
    entries = slots.update([{"landmarks": make_landmarks({"index": True}),
                             "handedness": "Right", "confidence": 0.9}],
                           timestamp=0.0)
    info = main_mod._debug_info(entries, [], {"now_playing": ""}, None, 0.0)
    assert info["final"] == "POINT"  # debounce_frames=1 switches immediately
    assert info["bits"] == "T0 I1 M0 R0 P0"
    assert info["track"] == "hands=1"


def test_draw_debug_renders():
    import numpy as np
    from positions import draw_debug
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_debug(f, {"raw": "POINT", "final": "NONE", "state": "CANDIDATE",
                   "reason": "clear", "bits": "T0 I1 M0 R0 P0",
                   "scores": "T:0.1", "pinch": "ratio=0.5",
                   "cooldown": "0.0s", "track": "hands=1"})
    assert f.sum() > 0


def test_pinch_touching_bypasses_thumb_out():
    # Foreshortened thumb (reach collapses, thumb_out False) but tips
    # truly touching + index bent -> still PINCH, never a missed toggle.
    from gesture import GestureDetector as GD
    d = GD(debounce_frames=1)
    lm = make_landmarks({})
    lm[4] = {"x": 0.47, "y": 0.55, "z": 0.0}   # thumb tucked at palm
    lm[8] = {"x": 0.47, "y": 0.55, "z": 0.0}   # index tip touches it
    lm[6] = {"x": 0.44, "y": 0.41, "z": 0.0}   # index pip stays back: bent
    assert not d._thumb_out(lm)
    r = None
    for _ in range(4):
        r = d.detect(lm)
    assert r["gesture"] == "PINCH"


def test_pinch_separate_tucked_thumb_stays_fist():
    # Same tucked thumb but tips apart -> FIST, not PINCH (no new FP).
    from gesture import GestureDetector as GD
    d = GD(debounce_frames=1)
    r = None
    for _ in range(4):
        r = d.detect(make_landmarks({}))
    assert r["gesture"] == "FIST"
