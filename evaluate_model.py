"""Evaluate a trained model on held-out sessions (disjoint from training).

Compares against the deterministic classifier on the same samples,
benches inference latency, and calls out OPEN_PALM false positives
(the privacy-critical error: a phantom palm can flip Privacy Mode).

Usage:
    ./myenv/bin/python evaluate_model.py --model models/gesture_rf.joblib --data data/holdout
    ./myenv/bin/python evaluate_model.py --model models/bootstrap_rf.joblib --synthetic --baseline --bench 200
"""

import argparse
import time

from features import LABELS
from ml_model import load_model, predict
from train_model import (build_synthetic, featurize, load_session_files,
                         metrics_report, print_report)


def baseline_predict(samples):
    """Deterministic classifier on the same samples (fresh state each)."""
    from gesture import GestureDetector
    out = []
    for s in samples:
        d = GestureDetector(debounce_frames=1)
        r = None
        for _ in range(6):
            r = d.detect(s["landmarks"])
        g = r["gesture"]
        out.append("UNKNOWN" if g == "NONE" else g)
    return out


def bench(pipeline, feats, n=200):
    import time as _t
    reps = (feats * ((n // len(feats)) + 1))[:n]
    t0 = _t.time()
    for f in reps:
        pipeline.predict([f])
    dt = (_t.time() - t0) * 1000.0 / n
    return round(dt, 3)


def main():
    ap = argparse.ArgumentParser(description="evaluate gesture model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="*", default=["data/sessions"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--bench", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--thresholds", nargs="*", type=float, default=None,
                    help="sweep thresholds, e.g. --thresholds 0.4 0.5 0.6 "
                         "0.7 0.8 (accuracy vs UNKNOWN tradeoff table)")
    args = ap.parse_args()

    pipe, meta = load_model(args.model)
    if meta.get("bootstrap"):
        print("NOTE: bootstrap model — synthetic numbers only, NOT product "
              "accuracy.", flush=True)
    sessions = (build_synthetic() if args.synthetic
                else load_session_files(args.data))
    overlap = {s["id"] for s in sessions} & set(meta.get("train_sessions", []))
    if overlap:
        print(f"WARNING: {len(overlap)} eval sessions were TRAIN sessions "
              f"— numbers are inflated, use disjoint holdout", flush=True)

    y_true, y_pred, feats = [], [], []
    for s in sessions:
        for sample in s["samples"]:
            from features import extract_features
            f = extract_features(sample["landmarks"])
            if f is None:
                continue
            feats.append(f)
            y_true.append(s["label"])
            y_pred.append(predict(pipe, sample["landmarks"],
                                  threshold=args.threshold)[0])
    report = metrics_report(y_true, y_pred)
    print(f"model={args.model} threshold={args.threshold}")
    print_report(report, sessions)

    # Privacy-critical: non-palm read as OPEN_PALM (phantom privacy flip).
    fp = sum(1 for t, p in zip(y_true, y_pred)
             if t != "OPEN_PALM" and p == "OPEN_PALM")
    fn = sum(1 for t, p in zip(y_true, y_pred)
             if t == "OPEN_PALM" and p != "OPEN_PALM")
    print(f"OPEN_PALM false positives: {fp} | missed palms: {fn}")
    unk = sum(1 for p in y_pred if p == "UNKNOWN")
    print(f"UNKNOWN rate: {unk}/{len(y_pred)} "
          f"({unk / max(1, len(y_pred)):.3f})")

    if args.thresholds:
        # Threshold calibration table (Tahap D): accuracy vs UNKNOWN
        # rate vs palm false positives across candidate thresholds.
        flat = [smp for s in sessions for smp in s["samples"]]
        print("--- threshold sweep ---")
        print(f"  {'thr':>5s} {'acc':>6s} {'unknown':>8s} {'palmFP':>7s} "
              f"{'missPalm':>9s}")
        for thr in args.thresholds:
            yp = [predict(pipe, s["landmarks"], threshold=thr)[0]
                  for s in flat]
            acc = sum(a == b for a, b in zip(y_true, yp)) / len(y_true)
            u = sum(1 for p in yp if p == "UNKNOWN")
            fpp = sum(1 for t, p in zip(y_true, yp)
                      if t != "OPEN_PALM" and p == "OPEN_PALM")
            fnn = sum(1 for t, p in zip(y_true, yp)
                      if t == "OPEN_PALM" and p != "OPEN_PALM")
            print(f"  {thr:>5.2f} {acc:>6.3f} "
                  f"{u / len(yp):>8.3f} {fpp:>7d} {fnn:>9d}")

    if args.baseline:
        # Same samples through the deterministic classifier.
        flat = [smp for s in sessions for smp in s["samples"]]
        base = baseline_predict(flat)
        b_true = [s["label"] for s in sessions for _ in s["samples"]]
        print("--- deterministic baseline (same samples) ---")
        b_report = metrics_report(b_true, base)
        print(f"accuracy={b_report['accuracy']:.3f}")
        for c, m in b_report["per_class"].items():
            print(f"  {c:10s} P={m['p']:.3f} R={m['r']:.3f} F1={m['f1']:.3f}")

    if args.bench and feats:
        print(f"infer latency: {bench(pipe, feats, args.bench)} ms/sample "
              f"(n={args.bench}, CPU)")


if __name__ == "__main__":
    main()
