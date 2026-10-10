"""Per-user calibration for gesture thresholds.

Procedure (see `--calibrate`): hold OPEN_PALM, FIST, and PINCH in turn
while the app collects landmark samples. This module derives suggested
thresholds from those samples and stores them as plain JSON — no images,
no video, no biometrics on disk.

A calibration is REJECTED (with reasons) when samples are too few or the
poses are not separable; blindly saving bad thresholds would be worse
than defaults.
"""

import json
import os

MIN_SAMPLES = 8


def default_path():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config")
    return os.path.join(base, "finger-pos", "calib.json")


def load_calib(path=None):
    """Load calibration dict; {} when missing or invalid (never raises)."""
    try:
        with open(path or default_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_calib(calib, path=None):
    """Save calibration dict, creating the config dir. Returns path."""
    path = path or default_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(calib, fh, indent=2)
    return path


def _angles(detector, samples, joints_key="index"):
    from gesture import FINGER_JOINTS
    joints = FINGER_JOINTS[joints_key]
    out = []
    for lm in samples:
        if not lm or len(lm) < 21:
            continue
        a = detector._angle(lm[joints["mcp"]], lm[joints["pip"]],
                            lm[joints["tip"]])
        if a is not None:
            out.append(a)
    return out


def _reaches(detector, samples):
    from gesture import FINGER_JOINTS
    out = []
    for lm in samples:
        if not lm or len(lm) < 21:
            continue
        scale = detector._hand_scale(lm)
        tip = lm[FINGER_JOINTS["thumb"]["tip"]]
        palm = lm[FINGER_JOINTS["middle"]["mcp"]]
        out.append(detector._dist(tip, palm) / scale)
    return out


def summarize(detector, samples_by_pose):
    """Derive thresholds from {pose: [landmarks,...]}.

    Poses: "open", "fist", "pinch". Extreme poses cannot calibrate the
    finger-angle band (joints bend the same for everyone — open is always
    ~180 deg), so angles are only a quality check; what actually varies
    per user is pinch gap and thumb reach, and those are what get tuned.
    Returns (calib_dict, warnings). Empty dict means rejected.
    """
    warnings = []
    for pose in ("open", "fist", "pinch"):
        n = len(samples_by_pose.get(pose) or [])
        if n < MIN_SAMPLES:
            warnings.append(f"pose {pose!r}: only {n} samples, need {MIN_SAMPLES}")
    if warnings:
        return {}, warnings

    open_a = _angles(detector, samples_by_pose["open"])
    if open_a and sum(open_a) / len(open_a) < 150.0:
        return {}, ["open pose not straight — spread fingers fully and retry"]

    open_r = _reaches(detector, samples_by_pose["open"])
    fist_r = _reaches(detector, samples_by_pose["fist"])
    if not open_r or not fist_r:
        return {}, ["no usable thumb landmarks"]
    o_mean = sum(open_r) / len(open_r)
    f_mean = sum(fist_r) / len(fist_r)
    if o_mean <= f_mean:
        return {}, ["thumb reach not separable (open must exceed fist)"]
    mid, span = (o_mean + f_mean) / 2.0, o_mean - f_mean
    thumb_extend = round(mid + span / 4.0, 3)
    thumb_fold = round(max(0.05, mid - span / 4.0), 3)

    pinch_d = [detector._calc_pinch(lm) for lm in samples_by_pose["pinch"]
               if lm and len(lm) >= 21]
    fist_d = [detector._calc_pinch(lm) for lm in samples_by_pose["fist"]
              if lm and len(lm) >= 21]
    if not pinch_d:
        return {}, ["no usable pinch samples"]
    pinch_max = max(pinch_d)
    fist_min = min(fist_d) if fist_d else pinch_max * 3.0
    if fist_min <= pinch_max * 1.5:
        warnings.append("pinch/fist gaps overlap — threshold is a compromise")
    pinch_thr = (pinch_max + max(fist_min, pinch_max * 2.0)) / 2.0

    calib = {
        "pinch_threshold": round(pinch_thr, 4),
        "thumb_extend": thumb_extend,
        "thumb_fold": thumb_fold,
        "samples": {p: len(samples_by_pose[p]) for p in ("open", "fist", "pinch")},
    }
    return calib, warnings
