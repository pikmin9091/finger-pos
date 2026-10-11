"""Shared model IO for training and evaluation (Tahap E uses this too).

A model file is a joblib dict {"pipeline", "meta"} where meta carries
the feature version, label set, and training provenance. predict()
refuses version mismatches instead of silently misclassifying, and maps
truncated landmarks / low-confidence outputs to UNKNOWN.
"""

import os
import time

from features import FEATURE_VERSION, LABELS, FEATURE_DIM, extract_features

PROBA_THRESHOLD = 0.6  # provisional; calibrate on validation data (Tahap D)


def save_model(pipeline, meta, path):
    """Persist pipeline + meta. Returns path."""
    import joblib
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"pipeline": pipeline, "meta": dict(meta)}
    payload["meta"].update(feature_version=FEATURE_VERSION,
                           labels=list(LABELS),
                           feature_dim=FEATURE_DIM,
                           saved_at=round(time.time(), 3))
    joblib.dump(payload, path)
    return path


def load_model(path):
    """Load a model file; raises ValueError on version/label mismatch."""
    import joblib
    payload = joblib.load(path)
    meta = payload.get("meta", {})
    if meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"feature version mismatch: model={meta.get('feature_version')} "
            f"code={FEATURE_VERSION} — retrain with train_model.py")
    if list(meta.get("labels", [])) != list(LABELS):
        raise ValueError("label set mismatch — retrain with train_model.py")
    return payload["pipeline"], meta


def predict(pipeline, landmarks, threshold=PROBA_THRESHOLD):
    """Classify landmarks -> (label, confidence).

    UNKNOWN (0.0) when landmarks are unusable or no class clears the
    threshold. Never raises on input problems.
    """
    feats = extract_features(landmarks)
    if feats is None:
        return "UNKNOWN", 0.0
    try:
        proba = pipeline.predict_proba([feats])[0]
        classes = list(pipeline.classes_)
    except Exception:
        return "UNKNOWN", 0.0
    best = max(range(len(proba)), key=lambda i: proba[i])
    label = classes[best] if best < len(classes) else "UNKNOWN"
    conf = float(proba[best])
    if label not in LABELS or conf < threshold:
        return "UNKNOWN", round(conf, 4)
    return label, round(conf, 4)
