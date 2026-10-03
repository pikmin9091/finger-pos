"""Deterministic gesture recognition engine using hand landmarks.

Uses landmark geometry (distances, relative positions) — no raw pixels.
Supports: POINT, PEACE, THUMBS_UP, THUMBS_DOWN, OPEN_PALM, FIST, PINCH, NONE.
Debounces gesture transitions to prevent flickering.
"""

import math

FINGER_JOINTS = {
    "thumb": {"tip": 4, "ip": 3, "mcp": 2, "wrist": 0},
    "index": {"tip": 8, "pip": 6, "mcp": 5, "wrist": 0},
    "middle": {"tip": 12, "pip": 10, "mcp": 9, "wrist": 0},
    "ring": {"tip": 16, "pip": 14, "mcp": 13, "wrist": 0},
    "pinky": {"tip": 20, "pip": 18, "mcp": 17, "wrist": 0},
}


class GestureDetector:
    """Deterministic gesture recognition with debouncing."""

    GESTURES = ["POINT", "PEACE", "THUMBS_UP", "THUMBS_DOWN",
                "OPEN_PALM", "FIST", "PINCH", "NONE"]

    def __init__(self, pinch_threshold=0.06, stable_frames=5, debounce_frames=3):
        self.pinch_threshold = pinch_threshold
        self.stable_frames = stable_frames
        self.debounce_frames = debounce_frames

        self._current = "NONE"
        self._same_count = 0
        self._diff_count = 0
        self._diff_gesture = None

    def detect(self, landmarks, handedness=None):
        """Detect gesture from landmarks. Returns dict with gesture, confidence, stable."""
        if not landmarks:
            return {"gesture": "NONE", "confidence": 0.0, "stable": False}

        extended = self._get_extended_fingers(landmarks)
        pinch_dist = self._calc_pinch(landmarks)
        is_pinch = pinch_dist < self.pinch_threshold

        gesture = self._classify(extended, is_pinch, pinch_dist, landmarks)
        self._current, is_stable = self._stabilize(gesture)

        confidence = self._calc_confidence(gesture, extended, pinch_dist, landmarks)

        return {
            "gesture": self._current,
            "confidence": round(confidence, 4),
            "stable": is_stable,
        }

    @staticmethod
    def _dist(a, b):
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)

    def _get_extended_fingers(self, landmarks):
        """Return dict {finger_name: bool} indicating extension state.

        Uses y-coordinate comparison for index/middle/ring/pinky:
        tip.y < pip.y means finger is pointing up (extended).
        For thumb: distance-based check.
        """
        extended = {}

        for name in ["index", "middle", "ring", "pinky"]:
            joints = FINGER_JOINTS[name]
            tip = landmarks[joints["tip"]]
            pip = landmarks[joints["pip"]]
            extended[name] = tip["y"] < pip["y"]

        # Thumb: tip far from IP joint
        thumb = FINGER_JOINTS["thumb"]
        dist_tip_ip = self._dist(landmarks[thumb["tip"]], landmarks[thumb["ip"]])
        extended["thumb"] = dist_tip_ip > 0.05

        return extended

    def _calc_pinch(self, landmarks):
        thumb_tip = landmarks[FINGER_JOINTS["thumb"]["tip"]]
        index_tip = landmarks[FINGER_JOINTS["index"]["tip"]]
        return self._dist(thumb_tip, index_tip)

    def _classify(self, extended, is_pinch, pinch_dist, landmarks):
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        ext_count = sum(1 for f in fingers if extended[f])

        # PINCH: thumb and index close together
        if is_pinch:
            return "PINCH"

        # THUMBS_UP / THUMBS_DOWN: thumb extended, others folded, direction by y
        if extended["thumb"] and ext_count == 1:
            thumb_tip = landmarks[FINGER_JOINTS["thumb"]["tip"]]
            wrist = landmarks[FINGER_JOINTS["thumb"]["wrist"]]
            if thumb_tip["y"] < wrist["y"]:
                return "THUMBS_UP"
            return "THUMBS_DOWN"

        # FIST: no fingers extended
        if ext_count == 0:
            return "FIST"

        # OPEN_PALM: 4+ fingers extended
        if ext_count >= 4:
            return "OPEN_PALM"

        # POINT: only index extended, thumb folded
        if (extended["index"] and not extended["middle"]
                and not extended["ring"] and not extended["pinky"]
                and not extended["thumb"]):
            return "POINT"

        # PEACE: index + middle extended, others folded
        if (extended["index"] and extended["middle"]
                and not extended["ring"] and not extended["pinky"]
                and not extended["thumb"]):
            return "PEACE"

        return "NONE"

    def _stabilize(self, gesture):
        """Debounce: require debounce_frames of a new gesture before switching."""
        if gesture == self._current:
            self._same_count = min(self._same_count + 1, self.stable_frames)
            self._diff_count = 0
            self._diff_gesture = None
        else:
            self._same_count = 0
            if gesture == self._diff_gesture:
                self._diff_count += 1
            else:
                self._diff_gesture = gesture
                self._diff_count = 1

            if self._diff_count >= self.debounce_frames:
                self._current = gesture
                self._diff_count = 0
                self._diff_gesture = None

        is_stable = self._same_count >= self.stable_frames
        return self._current, is_stable

    def _calc_confidence(self, gesture, extended, pinch_dist, landmarks):
        if gesture == "PINCH":
            return max(0.0, 1.0 - pinch_dist / self.pinch_threshold)

        if gesture == "FIST":
            folded = sum(1 for f in ["thumb", "index", "middle", "ring", "pinky"]
                         if not extended[f])
            return folded / 5.0

        if gesture == "OPEN_PALM":
            ext = sum(1 for f in ["thumb", "index", "middle", "ring", "pinky"]
                      if extended[f])
            return ext / 5.0

        if gesture == "POINT":
            correct = (extended["index"] and not extended["middle"]
                       and not extended["ring"] and not extended["pinky"])
            return 0.9 if correct else 0.5

        if gesture == "PEACE":
            correct = (extended["index"] and extended["middle"]
                       and not extended["ring"] and not extended["pinky"])
            return 0.9 if correct else 0.5

        if gesture in ("THUMBS_UP", "THUMBS_DOWN"):
            return 0.85

        return 0.5