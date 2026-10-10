import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MAX_FACES = 4

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "face_landmarker.task")

_landmarkers = {}
_active_conf = (0.5, 0.5, 0.5)
_last_ts = 0


def _next_timestamp_ms():
    """Monotonic millisecond timestamp required by VIDEO mode."""
    global _last_ts
    now = int(time.time() * 1000)
    _last_ts = now if now > _last_ts else _last_ts + 1
    return _last_ts


def _get_landmarker(det=0.5, presence=0.5, track=0.5):
    """Lazy singleton per confidence triple so importing this module never
    crashes when the model file is missing — the error only surfaces on
    first use."""
    key = (det, presence, track)
    if key not in _landmarkers:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"face model not found: {_MODEL_PATH} "
                "(download from https://developers.google.com/mediapipe/solutions/vision/face_landmarker)")
        _landmarkers[key] = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=MAX_FACES,
                min_face_detection_confidence=det,
                min_face_presence_confidence=presence,
                min_tracking_confidence=track,
            )
        )
    return _landmarkers[key]


def set_confidence(det=0.5, presence=0.5, track=0.5):
    """Select the active confidence triple. Returns it (clamped 0-1)."""
    global _active_conf
    _active_conf = (max(0.0, min(1.0, det)), max(0.0, min(1.0, presence)),
                    max(0.0, min(1.0, track)))
    return _active_conf


def detect_faces(frame):
    """Detect up to MAX_FACES faces. Returns {"faces": [face_dict, ...]}.

    Each face_dict has landmarks (478 dicts) and confidence
    (1.0 when found — FaceLandmarker reports no per-face score).
    Empty list when no face is visible.
    """
    if frame is None or not hasattr(frame, "ndim"):
        return {"faces": []}
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = _get_landmarker(*_active_conf).detect_for_video(mp_image, _next_timestamp_ms())
    faces = [{"landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face],
              "confidence": 1.0}
             for face in (result.face_landmarks or [])[:MAX_FACES]]
    return {"faces": faces}


def detect_face(frame):
    """Backward compatible single-face wrapper: first face or empty."""
    faces = detect_faces(frame)["faces"]
    if faces:
        return faces[0]
    return {"landmarks": [], "confidence": 0.0}
