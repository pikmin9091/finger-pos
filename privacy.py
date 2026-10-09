"""Smart Privacy Mode controller.

Modes:
  OFF         — video shown normally.
  FACE_BLUR   — blur every detected face box (+padding).
  STRICT      — blur a larger head region with stronger obfuscation.
                (No body-segmentation model exists in this project, so the
                whole head is approximated by expanding the face box.
                Honest limitation: silhouettes/body stay visible.)
  AUTO        — blur governed by configurable rules (see PrivacyRules).

PINCH gesture (via the internal BlurToggle edge detector) toggles
between OFF and the last active blur mode without disturbing the
existing gesture mapping. All decisions use the single detection
result already produced per frame — no extra inference.
"""

from blur import BlurToggle, FaceBoxMemory

MODES = ("OFF", "FACE_BLUR", "STRICT", "AUTO")
BLUR_KINDS = ("GAUSSIAN", "PIXELATE")

FACE_EXPAND = 0.15   # padding around the face box in FACE_BLUR/AUTO
HEAD_EXPAND = 0.60   # larger head-region padding in STRICT mode


class PrivacyRules:
    """AUTO-mode rules. All fields user-configurable."""

    def __init__(self, auto_on_face=True, min_faces=1, blur_kind="GAUSSIAN",
                 strength=5, grace=0.75):
        self.auto_on_face = auto_on_face  # blur whenever a face is seen
        self.min_faces = max(1, min_faces)  # ...or when faces >= this count
        self.blur_kind = blur_kind if blur_kind in BLUR_KINDS else "GAUSSIAN"
        self.strength = max(1, min(10, strength))
        self.grace = max(0.0, grace)  # box-memory grace in seconds


class PrivacyController:
    """Owns the privacy mode, rules, gesture toggle, and box memory."""

    def __init__(self, mode="OFF", rules=None, cooldown=1.0):
        self.mode = mode if mode in MODES else "OFF"
        self.rules = rules or PrivacyRules()
        self.last_blur_mode = "FACE_BLUR"  # PINCH toggles OFF <-> this
        self.toggle = BlurToggle(cooldown=cooldown)
        self.memory = FaceBoxMemory(grace=self.rules.grace)

    def toggle_mode(self):
        """PINCH-equivalent toggle: OFF <-> last active blur mode."""
        if self.mode == "OFF":
            self.mode = self.last_blur_mode
        else:
            self.last_blur_mode = self.mode
            self.mode = "OFF"
        return self.mode

    def cycle_mode(self):
        """OFF → FACE_BLUR → STRICT → AUTO → OFF."""
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        if self.mode != "OFF":
            self.last_blur_mode = self.mode
        return self.mode

    def update_pinch(self, is_pinch, timestamp):
        """Feed stable-PINCH state; flips OFF <-> blur mode on rising edge."""
        was = self.toggle.enabled
        if self.toggle.update(is_pinch, timestamp) != was:
            self.toggle_mode()
        return self.mode

    def evaluate(self, faces, timestamp):
        """Decide blurring for this frame.

        Args:
            faces: list of face entry dicts (with "box").
            timestamp: seconds since epoch.

        Returns dict {active, boxes, expand, kind, strength, reason}.
        Boxes fall back to recent memory inside the grace period so a
        dropped frame never flashes raw faces.
        """
        boxes = [f["box"] for f in (faces or [])]
        self.memory.grace = self.rules.grace
        self.memory.update(boxes, timestamp)
        mem_boxes = self.memory.boxes(timestamp)

        if self.mode == "OFF":
            return {"active": False, "boxes": [], "expand": 0.0,
                    "kind": self.rules.blur_kind, "strength": self.rules.strength,
                    "reason": "off"}
        if self.mode == "FACE_BLUR":
            use = boxes or mem_boxes
            return {"active": bool(use), "boxes": use, "expand": FACE_EXPAND,
                    "kind": self.rules.blur_kind, "strength": self.rules.strength,
                    "reason": "mode" if boxes else "memory"}
        if self.mode == "STRICT":
            use = boxes or mem_boxes
            return {"active": bool(use), "boxes": use, "expand": HEAD_EXPAND,
                    "kind": self.rules.blur_kind,
                    "strength": max(self.rules.strength, 7),
                    "reason": "strict" if boxes else "strict-memory"}
        # AUTO
        auto_on = ((self.rules.auto_on_face and bool(boxes)) or
                   len(boxes) >= self.rules.min_faces)
        if auto_on:
            self._was_auto_on = True
            use, reason = boxes, "auto"
        elif mem_boxes and self._was_auto_on:
            use, reason = mem_boxes, "auto-memory"  # brief drop while engaged
        else:
            self._was_auto_on = False
            use, reason = [], "auto-idle"
        return {"active": bool(use), "boxes": use, "expand": FACE_EXPAND,
                "kind": self.rules.blur_kind, "strength": self.rules.strength,
                "reason": reason}

    _was_auto_on = False

    def status(self):
        return {"mode": self.mode, "kind": self.rules.blur_kind,
                "strength": self.rules.strength}
