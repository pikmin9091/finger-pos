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

    # A finger counts as extended when the joint angle is this straight
    # (180 = perfectly straight). Angle-based = rotation invariant:
    # works pointing up, down, or sideways, unlike y-comparison.
    STRAIGHT_DEG = 140.0
    # Thumb fallback: absolute tip-IP distance (thumb anatomy bends
    # differently, so angle alone is unreliable for it).
    THUMB_MIN_DIST = 0.05

    def __init__(self, pinch_threshold=0.06, stable_frames=5, debounce_frames=3,
                 confidence_alpha=0.5):
        self.pinch_threshold = pinch_threshold
        self.stable_frames = stable_frames
        self.debounce_frames = debounce_frames
        self.confidence_alpha = max(0.0, min(1.0, confidence_alpha))

        self._current = "NONE"
        self._same_count = 0
        self._diff_count = 0
        self._diff_gesture = None
        self._was_pinch = False  # hysteresis state: harder to leave PINCH
        self._conf = None        # smoothed confidence

    def reset(self):
        """Clear debounce state (call when the hand is lost)."""
        self._current = "NONE"
        self._same_count = 0
        self._diff_count = 0
        self._diff_gesture = None
        self._was_pinch = False
        self._conf = None

    def detect(self, landmarks, handedness=None):
        """Detect gesture from landmarks. Returns dict with gesture, confidence, stable."""
        if not landmarks:
            self.reset()
            return {"gesture": "NONE", "confidence": 0.0, "stable": False}

        extended = self._get_extended_fingers(landmarks)
        pinch_dist = self._calc_pinch(landmarks)
        # Hysteresis band: entering PINCH needs dist < threshold, but once
        # pinching it takes dist > threshold*1.5 to leave — kills flicker
        # when the fingers hover right at the boundary.
        enter_thr = self.pinch_threshold
        exit_thr = self.pinch_threshold * 1.5
        is_pinch = pinch_dist < (exit_thr if self._was_pinch else enter_thr)

        gesture = self._classify(extended, is_pinch, pinch_dist, landmarks)
        self._was_pinch = (gesture == "PINCH")
        self._current, is_stable = self._stabilize(gesture)

        # Confidence must describe the returned (debounced) gesture,
        # not the raw frame classification which may still be in transition.
        raw_conf = self._calc_confidence(self._current, extended, pinch_dist, landmarks)
        if self._conf is None:
            self._conf = raw_conf
        else:
            a = self.confidence_alpha
            self._conf = a * raw_conf + (1 - a) * self._conf
        confidence = self._conf

        return {
            "gesture": self._current,
            "confidence": round(confidence, 4),
            "stable": is_stable,
        }

    @staticmethod
    def _dist(a, b):
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)

    @staticmethod
    def _angle(a, vertex, c):
        """Angle ABC (degrees) at `vertex` using 3D coords. None if degenerate."""
        v1x, v1y, v1z = a["x"] - vertex["x"], a["y"] - vertex["y"], a["z"] - vertex["z"]
        v2x, v2y, v2z = c["x"] - vertex["x"], c["y"] - vertex["y"], c["z"] - vertex["z"]
        n1 = math.sqrt(v1x ** 2 + v1y ** 2 + v1z ** 2)
        n2 = math.sqrt(v2x ** 2 + v2y ** 2 + v2z ** 2)
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        cos_val = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)
        cos_val = max(-1.0, min(1.0, cos_val))
        return math.degrees(math.acos(cos_val))

    def _get_extended_fingers(self, landmarks):
        """Return dict {finger_name: bool} indicating extension state.

        Index/middle/ring/pinky: extended when the PIP joint angle
        (MCP-PIP-tip, in 3D) is near-straight. Rotation and scale
        invariant — works pointing up, down, or sideways.
        Thumb: extended when its IP joint is open OR the tip is far
        from the IP joint (absolute fallback for thumb anatomy).
        """
        extended = {}

        for name in ["index", "middle", "ring", "pinky"]:
            joints = FINGER_JOINTS[name]
            ang = self._angle(landmarks[joints["mcp"]],
                              landmarks[joints["pip"]],
                              landmarks[joints["tip"]])
            extended[name] = ang is not None and ang >= self.STRAIGHT_DEG

        # Thumb: tip far from IP joint
        thumb = FINGER_JOINTS["thumb"]
        thumb_ang = self._angle(landmarks[thumb["mcp"]],
                                landmarks[thumb["ip"]],
                                landmarks[thumb["tip"]])
        dist_tip_ip = self._dist(landmarks[thumb["tip"]], landmarks[thumb["ip"]])
        extended["thumb"] = ((thumb_ang is not None and thumb_ang >= self.STRAIGHT_DEG)
                             or dist_tip_ip > self.THUMB_MIN_DIST)

        return extended

    def _calc_pinch(self, landmarks):
        thumb_tip = landmarks[FINGER_JOINTS["thumb"]["tip"]]
        index_tip = landmarks[FINGER_JOINTS["index"]["tip"]]
        return self._dist(thumb_tip, index_tip)

    def _classify(self, extended, is_pinch, pinch_dist, landmarks):
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        ext_count = sum(1 for f in fingers if extended[f])

        # PINCH: thumb and index close together. Require thumb or index
        # to be extended so a plain FIST whose curled tips happen to sit
        # near each other is not misread as PINCH.
        if is_pinch and (extended["thumb"] or extended["index"]):
            return "PINCH"

        # THUMBS_UP / THUMBS_DOWN: thumb extended, others folded.
        # Direction from the thumb's own MCP->tip vector (local, so it
        # doesn't depend on where the wrist happens to be).
        if extended["thumb"] and ext_count == 1:
            thumb_tip = landmarks[FINGER_JOINTS["thumb"]["tip"]]
            thumb_mcp = landmarks[FINGER_JOINTS["thumb"]["mcp"]]
            if thumb_tip["y"] < thumb_mcp["y"]:
                return "THUMBS_UP"
            return "THUMBS_DOWN"

        # FIST: no fingers extended
        if ext_count == 0:
            return "FIST"

        # OPEN_PALM: 4+ fingers extended
        if ext_count >= 4:
            return "OPEN_PALM"

        # POINT: only index extended. Thumb is ignored — when real users
        # point, the thumb is usually sticking out, not folded.
        if (extended["index"] and not extended["middle"]
                and not extended["ring"] and not extended["pinky"]):
            return "POINT"

        # PEACE: index + middle extended, others folded (thumb ignored).
        if (extended["index"] and extended["middle"]
                and not extended["ring"] and not extended["pinky"]):
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
            # Same hysteresis band as detection, so confidence stays
            # meaningful across the whole pinch region (not 0 at boundary).
            band = self.pinch_threshold * 1.5
            return max(0.0, 1.0 - pinch_dist / band)

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