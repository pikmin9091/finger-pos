import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hand_landmarker.task")

_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

def detect_hand(frame):
    """Detect hand in frame. Returns dict with landmarks, handedness, confidence."""
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = _landmarker.detect(mp_image)
    landmarks = []
    handedness = None
    confidence = 0.0
    if result.hand_landmarks:
        for lm in result.hand_landmarks[0]:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
        if result.handedness:
            cat = result.handedness[0][0]
            handedness = cat.category_name
            confidence = cat.score if cat.score is not None else 0.0
    return {"landmarks": landmarks, "handedness": handedness, "confidence": confidence}