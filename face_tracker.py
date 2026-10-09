"""Face position tracking engine.

Mirrors tracker.py conventions for hands. Provides per frame:
- face center (bbox center, normalized 0.0-1.0)
- bounding box + relative size + proximity (CLOSE/MEDIUM/FAR)
- screen zone (LEFT/CENTER/RIGHT, TOP/CENTER/BOTTOM)
- head orientation: yaw/pitch estimates + looking direction
  (viewer perspective: RIGHT = nose shifted toward image right)
"""

# Key MediaPipe face-mesh indices.
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
REQUIRED_MAX = max(NOSE_TIP, FOREHEAD, CHIN, LEFT_CHEEK, RIGHT_CHEEK)


class FaceTracker:
    """Tracks face position and head orientation across frames."""

    def __init__(self, pos_threshold=0.3, look_threshold=0.08,
                 close_size=0.20, medium_size=0.06, smoothing=0.4):
        self.pos_threshold = pos_threshold    # for LEFT/CENTER/RIGHT, TOP/CENTER/BOTTOM
        self.look_threshold = look_threshold  # min yaw/pitch for a look direction
        self.close_size = close_size          # bbox area above this = CLOSE
        self.medium_size = medium_size        # bbox area above this = MEDIUM, else FAR
        self.smoothing = max(0.0, min(1.0, smoothing))  # EMA alpha for center/box/yaw/pitch
        self._prev = None  # smoothed (cx, cy, x_min, y_min, x_max, y_max, yaw, pitch)

    def reset(self):
        """Clear smoothing state (call when the face is lost)."""
        self._prev = None

    def update(self, landmarks, confidence=0.0):
        """Process face landmarks. Returns face dict, or None when no face.

        Args:
            landmarks: list of 478 dicts with x, y, z
            confidence: detection confidence 0.0-1.0
        """
        if not landmarks or len(landmarks) <= REQUIRED_MAX:
            return None

        xs = [lm["x"] for lm in landmarks]
        ys = [lm["y"] for lm in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        yaw, pitch = self._calc_orientation(landmarks, x_max - x_min, y_max - y_min)

        # EMA smoothing so box/center/angles don't jitter frame-to-frame.
        raw = (cx, cy, x_min, y_min, x_max, y_max, yaw, pitch)
        if self._prev is None:
            s = raw
        else:
            a = self.smoothing
            s = tuple(a * r + (1 - a) * p for r, p in zip(raw, self._prev))
        self._prev = s
        cx, cy, x_min, y_min, x_max, y_max, yaw, pitch = s
        area = max(0.0, (x_max - x_min) * (y_max - y_min))

        h, v = self._classify_position(cx, cy)
        looking = self._calc_looking(yaw, pitch)

        return {
            "present": True,
            "confidence": round(confidence, 4),
            "center": {"x": round(cx, 6), "y": round(cy, 6)},
            "box": {"x_min": round(x_min, 6), "y_min": round(y_min, 6),
                    "x_max": round(x_max, 6), "y_max": round(y_max, 6)},
            "size": round(area, 6),
            "proximity": ("CLOSE" if area > self.close_size
                          else ("MEDIUM" if area > self.medium_size else "FAR")),
            "horizontal": h,
            "vertical": v,
            "yaw": round(yaw, 6),
            "pitch": round(pitch, 6),
            "looking": looking,
        }

    def _classify_position(self, x, y):
        """Same zone scheme as FingerTracker for a consistent API."""
        half = self.pos_threshold / 2
        h = "LEFT" if x < 0.5 - half else ("RIGHT" if x > 0.5 + half else "CENTER")
        v = "TOP" if y < 0.5 - half else ("BOTTOM" if y > 0.5 + half else "CENTER")
        return h, v

    def _calc_orientation(self, landmarks, width, height):
        """Estimate yaw/pitch from nose offset vs face midlines.

        yaw: nose.x relative to cheek midline, normalized by face width.
        pitch: nose.y relative to forehead-chin midline, normalized by height.
        Returns (yaw, pitch); positive yaw = looking image-right,
        positive pitch = looking down (image y grows downward).
        """
        nose = landmarks[NOSE_TIP]
        mid_x = (landmarks[LEFT_CHEEK]["x"] + landmarks[RIGHT_CHEEK]["x"]) / 2
        mid_y = (landmarks[FOREHEAD]["y"] + landmarks[CHIN]["y"]) / 2
        yaw = (nose["x"] - mid_x) / width if width > 1e-9 else 0.0
        pitch = (nose["y"] - mid_y) / height if height > 1e-9 else 0.0
        return yaw, pitch

    def _calc_looking(self, yaw, pitch):
        if abs(yaw) < self.look_threshold and abs(pitch) < self.look_threshold:
            return "CENTER"
        if abs(yaw) >= abs(pitch):
            return "RIGHT" if yaw > 0 else "LEFT"
        return "DOWN" if pitch > 0 else "UP"
