"""Train a gesture classifier from recorded landmark sessions.

Splits train/test BY SESSION (never by adjacent frames) so reported
accuracy is not inflated by near-duplicate neighbours. Features come
only from features.py; samples are mirror-augmented (left/right).

Usage:
    ./myenv/bin/python train_model.py --data data/sessions --out models/gesture_rf.joblib
    ./myenv/bin/python train_model.py --data data/sessions --model mlp --out models/gesture_mlp.joblib
    ./myenv/bin/python train_model.py --model both --data data/sessions --out models/gesture_rf.joblib
    ./myenv/bin/python train_model.py --synthetic --out models/bootstrap_rf.joblib
        (smoke-test only: geometric bootstrap, NOT a product model —
        real accuracy needs your recorded sessions, Tahap C.)
"""

import argparse
import glob
import os
import random
import time

from collect import read_session
from features import LABELS, extract_features, mirror
from ml_model import save_model

# 100 depth-capped trees: ~7 ms/sample single-predict on CPU (measured),
# inside the ~50 ms frame budget next to MediaPipe (~15 ms hands).
# 300 uncapped trees hit ~80 ms — too slow for the live loop.
RF_PARAMS = {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5,
             "class_weight": "balanced", "random_state": 7, "n_jobs": -1}
MLP_PARAMS = {"hidden_layer_sizes": (64,), "max_iter": 600,
              "random_state": 7}


def make_estimator(model_type):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model_type == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(**MLP_PARAMS))
    return make_pipeline(RandomForestClassifier(**RF_PARAMS))


def load_session_files(paths):
    """Files/dirs -> [{id, label, source, samples}] (single-label each)."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.jsonl")))
        elif os.path.isfile(p):
            files.append(p)
    sessions = []
    for f in files:
        meta, samples = read_session(f)
        sessions.append({"id": meta.get("session", os.path.basename(f)),
                         "label": meta["label"],
                         "source": meta.get("source", "camera"),
                         "samples": samples})
    return [s for s in sessions if s["samples"]]


def split_sessions(sessions, test_size=0.25, seed=7):
    """Stratified group split: per label, sessions go to train or test.

    Splitting inside each label keeps every recorded gesture represented
    on both sides; single-session labels stay in train with a warning
    (splitting one run would leak adjacent frames across the split).
    Returns (train, test).
    """
    rng = random.Random(seed)
    by_label = {}
    for s in sessions:
        by_label.setdefault(s["label"], []).append(s)
    train, test, single = [], [], []
    for label in sorted(by_label):
        ids = sorted({s["id"] for s in by_label[label]})
        if len(ids) < 2:
            single.append(label)
            train += by_label[label]
            continue
        rng.shuffle(ids)
        n_test = max(1, min(len(ids) - 1, round(len(ids) * test_size)))
        test_ids = set(ids[:n_test])
        test += [s for s in by_label[label] if s["id"] in test_ids]
        train += [s for s in by_label[label] if s["id"] not in test_ids]
    if single:
        print(f"WARNING: single-session labels stay in train only: "
              f"{single} — record a 2nd session per label", flush=True)
    if not test:
        raise SystemExit("ERROR: empty test split — need >= 2 sessions "
                         "for at least one label")
    return train, test


def featurize(sessions, augment=True):
    """Sessions -> (X, y). Mirror-augments every sample (left/right)."""
    X, y = [], []
    for s in sessions:
        for sample in s["samples"]:
            lm = sample["landmarks"]
            for variant in (lm, mirror(lm)) if augment else (lm,):
                feats = extract_features(variant)
                if feats is not None:
                    X.append(feats)
                    y.append(s["label"])
    return X, y


def metrics_report(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    labels = list(LABELS)
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0.0)
    return {"accuracy": round(float(acc), 4),
            "per_class": {c: {"p": round(float(p[i]), 3),
                              "r": round(float(r[i]), 3),
                              "f1": round(float(f[i]), 3)}
                          for i, c in enumerate(labels)},
            "confusion": confusion_matrix(y_true, y_pred,
                                          labels=labels).tolist()}


def print_report(report, test_sessions):
    print(f"test sessions: {len(test_sessions)} "
          f"({sum(len(s['samples']) for s in test_sessions)} samples)")
    print(f"accuracy={report['accuracy']:.3f}")
    print("per-class P/R/F1:")
    for c, m in report["per_class"].items():
        print(f"  {c:10s} P={m['p']:.3f} R={m['r']:.3f} F1={m['f1']:.3f}")
    print("confusion (true x pred):")
    head = " ".join(f"{c[:6]:>6s}" for c in LABELS)
    print(f"  {'':10s} {head}")
    for i, c in enumerate(LABELS):
        print(f"  {c:10s} "
              + " ".join(f"{v:>6d}" for v in report["confusion"][i]))


def train_sessions(train, test, model_type="rf"):
    """Train one estimator; returns (pipeline, report-on-test)."""
    Xtr, ytr = featurize(train)
    Xte, yte = featurize(test, augment=False)
    if not Xtr or not Xte:
        raise SystemExit("ERROR: empty train or test features — "
                         "record more sessions first")
    pipe = make_estimator(model_type)
    pipe.fit(Xtr, ytr)
    # Fit parallel, infer serial: n_jobs=-1 at predict time adds ~20 ms
    # joblib dispatch overhead per single sample (measured) — the live
    # loop classifies one hand at a time, so serial is 4x faster.
    try:
        pipe[-1].set_params(n_jobs=1)
    except (ValueError, TypeError, AttributeError):
        pass
    return pipe, metrics_report(yte, pipe.predict(Xte))


def build_synthetic(sessions_per_label=4, seed=7):
    """Geometric bootstrap sessions (train-plumbing smoke test only)."""
    import random as _random
    from eval_gestures import CASES, ROLLS, SCALES, transform
    from eval_gestures import weak_trio_pose
    from test_gesture import make_landmarks

    def pinch():
        lm = make_landmarks({})
        lm[4] = {"x": 0.40, "y": 0.47, "z": 0.0}
        lm[8] = {"x": 0.45, "y": 0.47, "z": 0.0}
        return lm

    def index_ring():
        return make_landmarks({"index": True, "ring": True})

    builders = dict(CASES)
    builders["PINCH"] = pinch
    unknowns = [weak_trio_pose, index_ring]
    sessions = []
    for label, build in builders.items():
        for k in range(sessions_per_label):
            rng = _random.Random(seed * 100 + hash(label) % 97 + k)
            samples = []
            for roll in ROLLS:
                for scale in SCALES:
                    lm = transform(build(), roll, scale, rng)
                    samples.append({"landmarks": lm})
            sessions.append({"id": f"synth_{label}_{k}", "label": label,
                             "source": "synthetic", "samples": samples})
    for k in range(sessions_per_label):
        rng = _random.Random(seed * 1000 + k)
        samples = []
        for builder in unknowns:
            for roll in ROLLS:
                lm = transform(builder(), roll, 1.0, rng)
                samples.append({"landmarks": lm})
        sessions.append({"id": f"synth_UNKNOWN_{k}", "label": "UNKNOWN",
                         "source": "synthetic", "samples": samples})
    return sessions


def main():
    ap = argparse.ArgumentParser(description="train gesture classifier")
    ap.add_argument("--data", nargs="*", default=["data/sessions"])
    ap.add_argument("--out", default="models/gesture_rf.joblib")
    ap.add_argument("--model", default="rf", choices=["rf", "mlp", "both"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.synthetic:
        print("SYNTHETIC BOOTSTRAP — plumbing smoke test, NOT a product "
              "model. Real accuracy needs recorded sessions (Tahap C).",
              flush=True)
        sessions = build_synthetic()
    else:
        sessions = load_session_files(args.data)
        if len({s["id"] for s in sessions}) < 2:
            raise SystemExit("ERROR: need sessions from >= 2 runs "
                             "(one file per run) for a session split")
    train, test = split_sessions(sessions, args.test_size, args.seed)
    print(f"sessions: {len(train)} train / {len(test)} test", flush=True)

    wanted = ("rf", "mlp") if args.model == "both" else (args.model,)
    saved = None
    for mtype in wanted:
        t0 = time.time()
        pipe, report = train_sessions(train, test, mtype)
        ms = (time.time() - t0) * 1000.0
        print(f"--- {mtype} (train {ms:.0f} ms) ---")
        print_report(report, test)
        if mtype == "rf" or args.model != "both":
            meta = {"model_type": mtype, "params": getattr(
                        pipe[-1], "get_params", lambda: {})(),
                    "train_sessions": sorted({s["id"] for s in train}),
                    "test_sessions": sorted({s["id"] for s in test}),
                    "n_train": sum(len(s["samples"]) for s in train),
                    "n_test": sum(len(s["samples"]) for s in test),
                    "metrics": report,
                    "bootstrap": bool(args.synthetic)}
            saved = save_model(pipe, meta, args.out if mtype == "rf"
                               else args.out.replace(".joblib",
                                                     f"_{mtype}.joblib"))
    print(f"Saved -> {saved}", flush=True)


if __name__ == "__main__":
    main()
