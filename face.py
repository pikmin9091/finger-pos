import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MAX_FACES = 4

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "face_landmarker.task")

_landmarker = None
_last_ts = 0


def _next_timestamp_ms():
    """Monotonic millisecond timestamp required by VIDEO mode."""
    global _last_ts
    now = int(time.time() * 1000)
    _last_ts = now if now > _last_ts else _last_ts + 1
    return _last_ts


def _get_landmarker():
    """Lazy singleton so importing this module never crashes when the
    model file is missing — the error only surfaces on first use."""
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"face model not found: {_MODEL_PATH} "
                "(download from https://developers.google.com/mediapipe/solutions/vision/face_landmarker)")
        _landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=MAX_FACES,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
    return _landmarker


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
    result = _get_landmarker().detect_for_video(mp_image, _next_timestamp_ms())
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
