import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MAX_HANDS = 2

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hand_landmarker.task")

# VIDEO mode enables the model's internal temporal tracking: landmarks
# are propagated frame-to-frame instead of redetected from scratch,
# which removes most per-frame flicker on live streams.
_landmarkers = {}


def _get_landmarker(det=0.5, presence=0.5, track=0.5):
    """Lazy singleton per confidence triple (also survives a missing model
    until first use with a clear error instead of an import crash)."""
    key = (det, presence, track)
    if key not in _landmarkers:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"hand model not found: {_MODEL_PATH} "
                "(download from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)")
        _landmarkers[key] = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=MAX_HANDS,
                min_hand_detection_confidence=det,
                min_hand_presence_confidence=presence,
                min_tracking_confidence=track,
            )
        )
    return _landmarkers[key]


def set_confidence(det=0.5, presence=0.5, track=0.5):
    """Select the active confidence triple. Returns it (clamped 0-1)."""
    det = max(0.0, min(1.0, det))
    presence = max(0.0, min(1.0, presence))
    track = max(0.0, min(1.0, track))
    global _active_conf
    _active_conf = (det, presence, track)
    return _active_conf


_active_conf = (0.5, 0.5, 0.5)

_last_ts = 0


def _next_timestamp_ms():
    """Monotonic millisecond timestamp required by VIDEO mode."""
    global _last_ts
    now = int(time.time() * 1000)
    _last_ts = now if now > _last_ts else _last_ts + 1
    return _last_ts


def _convert(hand_landmarks):
    return [{"x": lm.x, "y": lm.y, "z": lm.z,
             "visibility": float(getattr(lm, "visibility", 1.0) or 1.0)}
            for lm in hand_landmarks]


def detect_hands(frame):
    """Detect up to MAX_HANDS hands. Returns {"hands": [hand_dict, ...]}.

    Each hand_dict has landmarks (21 dicts), handedness, confidence.
    Empty list when no hand is visible.
    """
    if frame is None or not hasattr(frame, "ndim"):
        return {"hands": []}
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = _get_landmarker(*_active_conf).detect_for_video(
        mp_image, _next_timestamp_ms())
    hands = []
    if result.hand_landmarks:
        handed = result.handedness or []
        for i, hand_lm in enumerate(result.hand_landmarks[:MAX_HANDS]):
            name, score = None, 0.0
            if i < len(handed) and handed[i]:
                cat = handed[i][0]
                name = cat.category_name
                score = cat.score if cat.score is not None else 0.0
            hands.append({"landmarks": _convert(hand_lm),
                          "handedness": name, "confidence": score})
    return {"hands": hands}


def detect_hand(frame):
    """Backward compatible single-hand wrapper: first hand or empty."""
    hands = detect_hands(frame)["hands"]
    if hands:
        return hands[0]
    return {"landmarks": [], "handedness": None, "confidence": 0.0}
