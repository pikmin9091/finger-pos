"""Tests for ML feature extraction — no model/training needed."""

import math

from features import (FEATURE_DIM, FEATURE_VERSION, LABELS,
                      extract_features, mirror)
from test_gesture import make_landmarks


def test_dim_and_version():
    assert FEATURE_DIM == 75
    assert FEATURE_VERSION == 1
    assert set(LABELS) >= {"OPEN_PALM", "FIST", "PINCH", "UNKNOWN"}
    feats = extract_features(make_landmarks({"index": True}))
    assert len(feats) == FEATURE_DIM
    assert all(math.isfinite(v) for v in feats)


def test_deterministic_parity():
    lm = make_landmarks({"thumb": True})
    assert extract_features(lm) == extract_features(lm)


def test_translation_scale_invariance():
    lm = make_landmarks({f: True for f in
                         ["thumb", "index", "middle", "ring", "pinky"]})
    shifted = [{**p, "x": p["x"] + 0.2, "y": p["y"] - 0.1} for p in lm]
    a, b = extract_features(lm), extract_features(shifted)
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-9


def test_mirror_flips_x():
    lm = make_landmarks({"index": True})
    m = mirror(lm)
    assert m[8]["x"] == 1.0 - lm[8]["x"]
    assert m[8]["y"] == lm[8]["y"]


def test_incomplete_landmarks_rejected():
    assert extract_features([]) is None
    assert extract_features([{"x": 0.5, "y": 0.5, "z": 0.0}] * 10) is None
    bad = make_landmarks({})
    bad[0] = {"x": float("nan"), "y": 0.5, "z": 0.0}
    assert extract_features(bad) is None
    bad2 = make_landmarks({})
    bad2[5] = {"y": 0.5}  # missing x
    assert extract_features(bad2) is None
