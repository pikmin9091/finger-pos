"""Tests for calibration guards — bad sessions must not break recognition."""

from gesture import GestureDetector


def test_apply_calib_rejects_inseparable_thumb_pair():
    # Real-world bad session: extend/fold nearly identical (span 0.024).
    d = GestureDetector()
    d.apply_calib({"thumb_extend": 0.988, "thumb_fold": 0.964})
    assert (d.thumb_extend, d.thumb_fold) == (0.55, 0.40)


def test_apply_calib_rejects_out_of_range_pinch():
    d = GestureDetector()
    d.apply_calib({"pinch_threshold": 5.0})
    assert d.pinch_threshold == 0.06
    d.apply_calib({"pinch_threshold": 0.0})
    assert d.pinch_threshold == 0.06


def test_apply_calib_accepts_sane_values():
    d = GestureDetector()
    d.apply_calib({"pinch_threshold": 0.09, "thumb_extend": 0.70,
                   "thumb_fold": 0.30, "straight_deg": 150.0,
                   "folded_deg": 120.0})
    assert d.pinch_threshold == 0.09
    assert (d.thumb_extend, d.thumb_fold) == (0.70, 0.30)
    assert (d.straight_deg, d.folded_deg) == (150.0, 120.0)


def test_apply_calib_never_raises_on_garbage():
    d = GestureDetector()
    d.apply_calib(None)
    d.apply_calib("not-a-dict")
    d.apply_calib({"pinch_threshold": "abc", "thumb_extend": None,
                   "straight_deg": [1], "folded_deg": {}})
    assert d.pinch_threshold == 0.06
    assert (d.thumb_extend, d.thumb_fold) == (0.55, 0.40)


def test_summarize_rejects_barely_separable_span():
    from calib import summarize
    from test_gesture import make_landmarks
    det = GestureDetector()
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    open_lm = [make_landmarks({f: True for f in fingers}) for _ in range(8)]
    # Fist samples identical except the thumb parked a hair closer:
    # separable in theory (span ~0.06) but too tight to trust.
    fist_lm = []
    for _ in range(8):
        lm = make_landmarks({f: True for f in fingers})
        lm[4] = {"x": 0.315, "y": 0.55, "z": 0.0}
        fist_lm.append(lm)
    pinch_lm = []
    for _ in range(8):
        lm = make_landmarks({})
        lm[4] = {"x": 0.40, "y": 0.47, "z": 0.0}
        lm[8] = {"x": 0.45, "y": 0.47, "z": 0.0}
        pinch_lm.append(lm)
    calib, warnings = summarize(
        det, {"open": open_lm, "fist": fist_lm, "pinch": pinch_lm})
    assert calib == {} and warnings
