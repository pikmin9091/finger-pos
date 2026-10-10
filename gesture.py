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
    STRAIGHT_DEG = 150.0
    FOLDED_DEG = 125.0
    # Thumb extension uses tip-to-PALM distance (tip -> middle MCP)
    # relative to hand scale — NOT tip-to-own-MCP, whose bone length is
    # nearly constant whether the thumb is folded or extended and whose
    # old 0.90 threshold no real thumb could ever reach (hence every
    # THUMBS_UP read as FIST). Extended-out ~= 0.65+, tucked ~= 0.25.
    THUMB_EXTEND_RATIO = 0.55
    THUMB_FOLD_RATIO = 0.40
    # PINCH uses its own thumb check: the thumb must leave its base
    # (tip far from its OWN MCP) to meet the index. A tucked thumb can
    # never pinch, even if its tip happens to sit near a curled index.
    THUMB_OUT_RATIO = 0.45
    # Margin decisions: below MIN_SCORE nothing is claimed; a winner must
    # beat the runner-up by AMBIGUITY_MARGIN or the frame is UNKNOWN.
    MIN_SCORE = 0.5
    AMBIGUITY_MARGIN = 0.15

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
        self._last_raw = "NONE"  # pre-debounce classification (debug)
        self._last_thumb = {"reach": 0.0, "alignment": 0.0}  # debug
        # Instance copies so calibration can override without touching code.
        self.straight_deg = self.STRAIGHT_DEG
        self.folded_deg = self.FOLDED_DEG
        self.thumb_extend = self.THUMB_EXTEND_RATIO
        self.thumb_fold = self.THUMB_FOLD_RATIO

    def apply_calib(self, calib):
        """Override thresholds from a calibration dict (see calib.py)."""
        for key in ("pinch_threshold", "straight_deg", "folded_deg",
                    "thumb_extend", "thumb_fold"):
            if key in calib:
                setattr(self, key, float(calib[key]))

    def reset(self):
        """Clear debounce state (call when the hand is lost)."""
        self._current = "NONE"
        self._same_count = 0
        self._diff_count = 0
        self._diff_gesture = None
        self._was_pinch = False
        self._conf = None

    def detect(self, landmarks, handedness=None):
        """Detect gesture from landmarks.

        Returns dict with gesture, confidence, stable, plus debug fields:
        state (CONFIRMED/CANDIDATE/UNKNOWN), reason (clear/weak-pattern/
        no-match/empty/truncated), and per-finger extension scores.
        Truncated landmark lists are rejected, never classified.
        """
        if not landmarks:
            self.reset()
            return {"gesture": "NONE", "confidence": 0.0, "stable": False,
                    "state": "UNKNOWN", "reason": "empty", "raw": "NONE",
                    "scores": {}, "thumb": {}}
        if len(landmarks) < 21:
            self.reset()
            return {"gesture": "NONE", "confidence": 0.0, "stable": False,
                    "state": "UNKNOWN", "reason": "truncated", "raw": "NONE",
                    "scores": {}, "thumb": {}}

        extended, scores = self._get_extended_fingers(landmarks)
        pinch_dist = self._calc_pinch(landmarks)
        self._last_thumb = {
            "reach": round(self._dist(
                landmarks[FINGER_JOINTS["thumb"]["tip"]],
                landmarks[FINGER_JOINTS["middle"]["mcp"]])
                / self._hand_scale(landmarks), 4),
            "alignment": round(self._thumb_alignment(landmarks), 4),
        }
        # Hysteresis band: entering PINCH needs dist < threshold, but once
        # pinching it takes dist > threshold*1.5 to leave — kills flicker
        # when the fingers hover right at the boundary.
        enter_thr = self.pinch_threshold
        exit_thr = self.pinch_threshold * 1.5
        is_pinch = pinch_dist < (exit_thr if self._was_pinch else enter_thr)

        gesture, reason, patterns = self._classify(extended, scores, is_pinch,
                                                     pinch_dist, landmarks,
                                                     touch_thr=(exit_thr if self._was_pinch
                                                                else enter_thr) * 0.5)
        self._was_pinch = (gesture == "PINCH")
        self._last_raw = gesture
        self._current, is_stable = self._stabilize(gesture)

        # Confidence must describe the returned (debounced) gesture,
        # not the raw frame classification which may still be in transition.
        # It is a heuristic pattern-strength blend, NOT a calibrated
        # probability — treat it as quality, not certainty.
        raw_conf = self._calc_confidence(self._current, extended, pinch_dist,
                                         landmarks, patterns)
        if self._conf is None:
            self._conf = raw_conf
        else:
            a = self.confidence_alpha
            self._conf = a * raw_conf + (1 - a) * self._conf
        confidence = self._conf

        if is_stable:
            state = "CONFIRMED"
        elif self._current != "NONE" or self._diff_gesture is not None:
            state = "CANDIDATE"
        else:
            state = "UNKNOWN"

        return {
            "gesture": self._current,
            "confidence": round(confidence, 4),
            "stable": is_stable,
            "state": state,
            "reason": reason,
            "raw": self._last_raw,
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "thumb": self._last_thumb,
        }

    @staticmethod
    def _dist(a, b):
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)

    @classmethod
    def _hand_scale(cls, landmarks):
        """Use the larger palm axis so ratios are stable across camera distance."""
        wrist_to_middle = cls._dist(landmarks[0], landmarks[9])
        palm_width = cls._dist(landmarks[5], landmarks[17])
        return max(wrist_to_middle, palm_width, 1e-6)

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

    def _score_extension(self, ang, tip_mcp, pip_mcp, tip_pip=None):
        """Continuous extension score in [0,1].

        Linear between folded_deg (0) and straight_deg (1). Guards first:
        a tip parked near its PIP (ratio < 0.5 of bone length) is a curled
        finger — its joint angle is pure noise direction, so it reads
        folded without consulting the angle. A degenerate angle
        (overlapping joints) falls back to reach comparison instead of
        freezing any previous state.
        """
        if tip_pip is not None and pip_mcp > 1e-9 \
                and tip_pip < 0.5 * pip_mcp:
            return 0.0
        if ang is None:
            return 1.0 if tip_mcp > pip_mcp else 0.0
        if ang >= self.straight_deg:
            return 1.0
        if ang <= self.folded_deg:
            return 0.0
        return (ang - self.folded_deg) / (self.straight_deg - self.folded_deg)

    def _thumb_score(self, reach):
        if reach >= self.thumb_extend:
            return 1.0
        if reach <= self.thumb_fold:
            return 0.0
        return (reach - self.thumb_fold) / (self.thumb_extend - self.thumb_fold)

    def _get_extended_fingers(self, landmarks):
        """Return ({name: bool}, {name: score}) extension state.

        Index/middle/ring/pinky: PIP joint angle in 3D (rotation and scale
        invariant). Thumb: MCP-to-tip reach relative to palm size.
        Binary uses a 0.5 cutoff; scores grade pattern strength.
        """
        extended, scores = {}, {}

        for name in ["index", "middle", "ring", "pinky"]:
            joints = FINGER_JOINTS[name]
            ang = self._angle(landmarks[joints["mcp"]],
                              landmarks[joints["pip"]],
                              landmarks[joints["tip"]])
            tip_mcp = self._dist(landmarks[joints["tip"]],
                                 landmarks[joints["mcp"]])
            pip_mcp = self._dist(landmarks[joints["pip"]],
                                 landmarks[joints["mcp"]])
            tip_pip = self._dist(landmarks[joints["tip"]],
                                 landmarks[joints["pip"]])
            s = self._score_extension(ang, tip_mcp, pip_mcp, tip_pip)
            scores[name] = s
            extended[name] = s >= 0.5

        # Thumb: extended when the tip is far from the PALM (middle
        # MCP), i.e. sticking out — not merely far from its own base,
        # which stays long even when folded across the palm.
        thumb = FINGER_JOINTS["thumb"]
        scale = self._hand_scale(landmarks)
        reach = self._dist(landmarks[thumb["tip"]],
                           landmarks[FINGER_JOINTS["middle"]["mcp"]]) / scale
        scores["thumb"] = self._thumb_score(reach)
        extended["thumb"] = scores["thumb"] >= 0.5

        return extended, scores

    def _thumb_out(self, landmarks):
        """True when the thumb left its base (for PINCH gating)."""
        thumb = FINGER_JOINTS["thumb"]
        scale = self._hand_scale(landmarks)
        return (self._dist(landmarks[thumb["tip"]],
                           landmarks[thumb["mcp"]]) / scale
                > self.THUMB_OUT_RATIO)

    def _calc_pinch(self, landmarks):
        thumb_tip = landmarks[FINGER_JOINTS["thumb"]["tip"]]
        index_tip = landmarks[FINGER_JOINTS["index"]["tip"]]
        # Keep the public threshold near its old value for an average hand
        # (about 0.15 normalized units), while compensating for hand size.
        return self._dist(thumb_tip, index_tip) / self._hand_scale(landmarks) * 0.15

    def _pattern_scores(self, s, pinch_dist):
        """Graded strength of each gesture pattern in [0,1]."""
        band = self.pinch_threshold * 1.5
        pinch_score = max(0.0, 1.0 - pinch_dist / band)
        straight = {f: s[f] for f in ("index", "middle", "ring", "pinky")}
        folded = {f: 1.0 - s[f] for f in straight}
        return {
            "POINT": s["index"] * folded["middle"] * folded["ring"] * folded["pinky"],
            "PINCH": pinch_score * s["thumb"] * folded["index"],
            "FIST": (folded["index"] + folded["middle"] + folded["ring"]
                     + folded["pinky"] + (1.0 - s["thumb"])) / 5.0,
            "OPEN_PALM": (s["index"] + s["middle"] + s["ring"] + s["pinky"]
                          + s["thumb"]) / 5.0,
            "PEACE": s["index"] * s["middle"] * folded["ring"] * folded["pinky"],
        }

    def _thumb_alignment(self, landmarks):
        """Cosine between thumb axis and the palm's stable up-axis, or 0.0.

        The reference is wrist->middle-MCP (the palm itself), NOT a finger:
        a genuinely curled middle finger points toward the wrist, so using
        it as reference inverts THUMBS_UP into THUMBS_DOWN on real hands.
        The palm axis is always defined and rotates with the hand, so the
        reading stays hand-relative under roll.
        """
        thumb_tip = landmarks[FINGER_JOINTS["thumb"]["tip"]]
        thumb_mcp = landmarks[FINGER_JOINTS["thumb"]["mcp"]]
        wrist = landmarks[FINGER_JOINTS["index"]["wrist"]]
        middle_mcp = landmarks[FINGER_JOINTS["middle"]["mcp"]]
        thumb_axis = tuple(thumb_tip[k] - thumb_mcp[k] for k in "xyz")
        palm_axis = tuple(middle_mcp[k] - wrist[k] for k in "xyz")
        thumb_len = math.sqrt(sum(v * v for v in thumb_axis))
        palm_len = math.sqrt(sum(v * v for v in palm_axis))
        if thumb_len < 1e-6 or palm_len < 1e-6:
            return 0.0
        return (sum(a * b for a, b in zip(thumb_axis, palm_axis))
                / (thumb_len * palm_len))

    def _classify(self, extended, scores, is_pinch, pinch_dist, landmarks,
                    touch_thr=None):
        """Priority rules + pattern-strength gate.

        Returns (gesture, reason). A matched pattern weaker than MIN_SCORE
        is rejected as UNKNOWN ("weak-pattern") instead of being forced
        into a gesture — borderline poses must not fire actions.
        PINCH keeps its hysteresis band as its margin (its boundary case
        is product-defined as PINCH, not UNKNOWN).
        """
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        ext_count = sum(1 for f in fingers if extended[f])
        patterns = self._pattern_scores(scores, pinch_dist)

        def strong(name):
            return patterns[name] >= self.MIN_SCORE

        # POINT takes precedence when the index is straight. This prevents
        # a nearby thumb from turning a normal pointing pose into PINCH.
        if (extended["index"] and not extended["middle"]
                and not extended["ring"] and not extended["pinky"]):
            if strong("POINT"):
                return "POINT", "clear", patterns
            return "NONE", "weak-pattern", patterns

        # PINCH needs the thumb reaching out (left its base) and a bent
        # index finger close together. The bent-index requirement separates
        # it from a straight POINT; the thumb-out requirement separates it
        # from a plain FIST whose curled tips happen to sit near each other.
        # Escape hatch from real logs: a foreshortened thumb (pointing at
        # the camera) collapses its 2D reach below any sane thumb_out
        # threshold, so truly TOUCHING tips always count as PINCH — a
        # touching pair cannot be a plain fist by product definition.
        touching = (touch_thr is not None and pinch_dist < touch_thr)
        if (is_pinch and not extended["index"]
                and (touching or self._thumb_out(landmarks))):
            return "PINCH", "clear", patterns

        # THUMBS_UP / THUMBS_DOWN: thumb extended, others folded. Direction
        # is read against the palm's own up-axis (stable under roll);
        # sideways thumbs are rejected as ambiguous.
        if extended["thumb"] and ext_count == 1:
            alignment = self._thumb_alignment(landmarks)
            if alignment > 0.35:
                gesture = "THUMBS_UP"
            elif alignment < -0.35:
                gesture = "THUMBS_DOWN"
            else:
                return "NONE", "ambiguous-thumb", patterns
            # Thumb-only strength folds direction alignment in, so a
            # sideways or half-curled thumb is rejected, not snapped.
            if abs(alignment) * scores["thumb"] >= self.MIN_SCORE:
                return gesture, "clear", patterns
            return "NONE", "weak-pattern", patterns

        # FIST: no fingers extended
        if ext_count == 0:
            if strong("FIST"):
                return "FIST", "clear", patterns
            return "NONE", "weak-pattern", patterns

        # OPEN_PALM: 4+ fingers extended
        if ext_count >= 4:
            if strong("OPEN_PALM"):
                return "OPEN_PALM", "clear", patterns
            return "NONE", "weak-pattern", patterns

        # PEACE: index + middle extended, others folded (thumb ignored).
        if (extended["index"] and extended["middle"]
                and not extended["ring"] and not extended["pinky"]):
            if strong("PEACE"):
                return "PEACE", "clear", patterns
            return "NONE", "weak-pattern", patterns

        return "NONE", "no-match", patterns

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

    def _calc_confidence(self, gesture, extended, pinch_dist, landmarks, patterns):
        """Heuristic pattern-strength quality in [0,1].

        NOT a calibrated probability — report it as match quality.
        """
        if gesture == "PINCH":
            # Same hysteresis band as detection, so confidence stays
            # meaningful across the whole pinch region (not 0 at boundary).
            band = self.pinch_threshold * 1.5
            return max(0.0, 1.0 - pinch_dist / band)

        if gesture == "FIST":
            return patterns["FIST"]

        if gesture == "OPEN_PALM":
            return patterns["OPEN_PALM"]

        if gesture == "POINT":
            return 0.5 + 0.5 * patterns["POINT"]

        if gesture == "PEACE":
            return 0.5 + 0.5 * patterns["PEACE"]

        if gesture in ("THUMBS_UP", "THUMBS_DOWN"):
            return 0.5 + 0.5 * min(1.0, abs(self._thumb_alignment(landmarks)))

        return 0.5
