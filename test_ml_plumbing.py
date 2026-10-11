"""Tests for dataset IO, session splits, and the training plumbing."""

import json

import pytest

sklearn = pytest.importorskip("sklearn")

from collect import read_session, session_path, write_meta, write_sample
from features import LABELS
from test_gesture import make_landmarks


def _write_session(path, label, n=6, **kw):
    with open(path, "w") as fh:
        write_meta(fh, label, kw.get("session", "s1"))
        for _ in range(n):
            write_sample(fh, make_landmarks({}), handedness="Right",
                         confidence=0.9)


def test_session_roundtrip_and_reject_short(tmp_path):
    p = str(tmp_path / "s.jsonl")
    _write_session(p, "FIST", session="sess1")
    meta, samples = read_session(p)
    assert meta["label"] == "FIST" and meta["session"] == "sess1"
    assert len(samples) == 6
    assert len(samples[0]["landmarks"]) == 21


def test_session_path_names(tmp_path):
    p = session_path(str(tmp_path), "OPEN_PALM", timestamp=123)
    assert p.endswith("OPEN_PALM_123.jsonl")


def test_group_split_no_session_leakage():
    from train_model import split_sessions
    sessions = [{"id": f"s{i}", "label": "FIST", "samples": [{}]}
                for i in range(4)]
    train, test = split_sessions(sessions, test_size=0.5, seed=1)
    assert {s["id"] for s in train} & {s["id"] for s in test} == set()
    assert len(train) == 2 and len(test) == 2


def test_synthetic_bootstrap_trains_and_predicts():
    from ml_model import load_model, predict
    from train_model import build_synthetic, split_sessions, train_sessions
    sessions = build_synthetic(sessions_per_label=2)
    assert {s["label"] for s in sessions} == set(LABELS)
    train, test = split_sessions(sessions, seed=7)
    assert {s["id"] for s in train} & {s["id"] for s in test} == set()
    import train_model
    orig = train_model.RF_PARAMS
    train_model.RF_PARAMS = {**orig, "n_estimators": 25}
    try:
        pipe, report = train_sessions(train, test, "rf")
    finally:
        train_model.RF_PARAMS = orig
    assert report["accuracy"] > 0.9  # synthetic geometry: must be easy
    label, conf = predict(pipe, make_landmarks({}))
    assert label == "FIST" and conf >= 0.6
    assert predict(pipe, []) == ("UNKNOWN", 0.0)


def test_model_version_gate(tmp_path):
    import joblib
    from ml_model import load_model, save_model
    from train_model import make_estimator
    from features import extract_features
    X = [extract_features(make_landmarks({})),
         extract_features(make_landmarks({"index": True}))]
    pipe = make_estimator("rf")
    pipe.set_params(randomforestclassifier__n_estimators=5)
    pipe.fit(X, ["FIST", "POINT"])
    p = save_model(pipe, {"model_type": "rf"}, str(tmp_path / "m.joblib"))
    _, meta = load_model(p)
    assert meta["feature_version"] == 1
    payload = joblib.load(p)
    payload["meta"]["feature_version"] = 999
    joblib.dump(payload, p)
    with pytest.raises(ValueError):
        load_model(p)
