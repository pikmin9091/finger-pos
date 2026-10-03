"""Finger position and movement tracking engine.

Separates position calculations from camera capture, MediaPipe detection,
and visualization. Provides:
- normalized fingertip coordinates (x, y, z in 0.0-1.0)
- position classification (LEFT/CENTER/RIGHT, TOP/CENTER/BOTTOM)
- hand center from palm landmarks
- movement tracking with velocity and direction
- exponential moving average smoothing
"""

TIP_MAP = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}
PALM_INDICES = list(range(10))  # wrist + palm base for center calculation


class FingerTracker:
    """Tracks finger positions and movement across frames."""

    def __init__(self, smoothing=0.2, pos_threshold=0.3, min_velocity=0.01,
                 reset_after_lost=15):
        self.smoothing = smoothing          # EMA alpha (0=full smooth, 1=raw)
        self.pos_threshold = pos_threshold  # for LEFT/CENTER/RIGHT, TOP/CENTER/BOTTOM
        self.min_velocity = min_velocity    # below this = STATIONARY
        self.reset_after_lost = reset_after_lost  # frames without hand before reset

        self._smoothed = None       # current smoothed tip positions
        self._prev_smoothed = None  # previous smoothed tip positions
        self._prev_time = None      # previous timestamp
        self._lost_frames = 0       # consecutive frames without hand

    def update(self, landmarks, handedness=None, confidence=0.0, timestamp=None):
        """Process a new frame and return tracking data.

        Args:
            landmarks: list of 21 dicts with x, y, z, visibility
            handedness: "Right", "Left", or None
            confidence: detection confidence 0.0-1.0
            timestamp: seconds since epoch (float)

        Returns:
            dict with hand, confidence, fingers, hand_center, movement
        """
        if not landmarks:
            self._lost_frames += 1
            # Clear stale smoothed state so a re-detected hand starts fresh
            # instead of lerping from the last known position.
            if self._lost_frames >= self.reset_after_lost:
                self._smoothed = None
                self._prev_smoothed = None
                self._prev_time = None
            return {
                "hand": handedness or "Unknown",
                "confidence": round(confidence, 4),
                "fingers": {},
                "hand_center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "movement": {"direction": "STATIONARY", "velocity_x": 0.0, "velocity_y": 0.0},
            }
        self._lost_frames = 0
        # 1. Extract raw tips
        raw_tips = self._extract_tips(landmarks)

        # 2. Save previous smoothed, compute new smoothed
        self._prev_smoothed = self._smoothed
        self._smoothed = self._smooth(raw_tips)

        # 3. Classify each fingertip position
        fingers = {}
        for name, pos in self._smoothed.items():
            h, v = self._classify_position(pos["x"], pos["y"])
            fingers[name] = {
                "x": round(pos["x"], 6),
                "y": round(pos["y"], 6),
                "z": round(pos["z"], 6),
                "horizontal": h,
                "vertical": v,
            }

        # 4. Hand center
        hand_center = self._calc_hand_center(landmarks)

        # 5. Movement tracking
        movement = self._calc_movement(timestamp)

        return {
            "hand": handedness or "Unknown",
            "confidence": round(confidence, 4),
            "fingers": fingers,
            "hand_center": hand_center,
            "movement": movement,
        }

    def _extract_tips(self, landmarks):
        return {
            name: {"x": landmarks[idx]["x"], "y": landmarks[idx]["y"], "z": landmarks[idx]["z"]}
            for name, idx in TIP_MAP.items()
        }

    def _smooth(self, raw_tips):
        """Exponential moving average smoothing."""
        if self._smoothed is None:
            return {name: dict(pos) for name, pos in raw_tips.items()}
        alpha = self.smoothing
        return {
            name: {
                "x": alpha * raw_tips[name]["x"] + (1 - alpha) * self._smoothed[name]["x"],
                "y": alpha * raw_tips[name]["y"] + (1 - alpha) * self._smoothed[name]["y"],
                "z": alpha * raw_tips[name]["z"] + (1 - alpha) * self._smoothed[name]["z"],
            }
            for name in raw_tips
        }

    def _classify_position(self, x, y):
        """Classify x,y into LEFT/CENTER/RIGHT and TOP/CENTER/BOTTOM."""
        half = self.pos_threshold / 2
        h = "LEFT" if x < 0.5 - half else ("RIGHT" if x > 0.5 + half else "CENTER")
        v = "TOP" if y < 0.5 - half else ("BOTTOM" if y > 0.5 + half else "CENTER")
        return h, v

    def _calc_hand_center(self, landmarks):
        """Calculate approximate palm center from landmarks 0-9."""
        n = len(PALM_INDICES)
        cx = sum(landmarks[i]["x"] for i in PALM_INDICES) / n
        cy = sum(landmarks[i]["y"] for i in PALM_INDICES) / n
        cz = sum(landmarks[i]["z"] for i in PALM_INDICES) / n
        return {"x": round(cx, 6), "y": round(cy, 6), "z": round(cz, 6)}

    def _calc_movement(self, timestamp):
        """Calculate movement direction and velocity from previous frame."""
        if self._prev_smoothed is None or self._prev_time is None or timestamp is None:
            self._prev_time = timestamp
            return {"direction": "STATIONARY", "velocity_x": 0.0, "velocity_y": 0.0}

        dt = timestamp - self._prev_time
        if dt < 0.001:
            return {"direction": "STATIONARY", "velocity_x": 0.0, "velocity_y": 0.0}

        # Use index_tip as primary movement reference
        name = "index_tip"
        dx = self._smoothed[name]["x"] - self._prev_smoothed[name]["x"]
        dy = self._smoothed[name]["y"] - self._prev_smoothed[name]["y"]

        vx = dx / dt
        vy = dy / dt

        if abs(vx) < self.min_velocity and abs(vy) < self.min_velocity:
            direction = "STATIONARY"
        elif abs(vx) > abs(vy):
            direction = "RIGHT" if vx > 0 else "LEFT"
        else:
            direction = "DOWN" if vy > 0 else "UP"

        return {
            "direction": direction,
            "velocity_x": round(vx, 6),
            "velocity_y": round(vy, 6),
        }