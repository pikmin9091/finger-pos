"""Gesture-controlled face blur.

PINCH (thumb tip + index tip together) toggles face blurring on/off.
The toggle fires only on the rising edge of a *stable* PINCH with a
cooldown and release requirement, so holding the pinch never retriggers.
Blur state persists even when no hand is visible.
All processing is local (OpenCV GaussianBlur on face ROIs) — nothing is
uploaded, recorded, or saved by this module.
"""

import time

import cv2


class BlurToggle:
    """Edge-triggered on/off toggle driven by a stable PINCH gesture."""

    def __init__(self, cooldown=1.0):
        self.cooldown = cooldown  # min seconds between toggles
        self.enabled = False
        self._was_pinch = False
        self._last_toggle = float("-inf")

    def update(self, is_pinch, timestamp=None):
        """Feed the current stable-PINCH state. Returns blur enabled.

        Toggles only on a rising edge (no pinch → pinch) once the
        cooldown since the last toggle has elapsed. Holding the pinch
        or hiding the hand changes nothing.
        """
        if timestamp is None:
            timestamp = time.time()
        if is_pinch and not self._was_pinch:
            if timestamp - self._last_toggle >= self.cooldown:
                self.enabled = not self.enabled
                self._last_toggle = timestamp
        self._was_pinch = bool(is_pinch)
        return self.enabled

    def reset(self):
        self.enabled = False
        self._was_pinch = False
        self._last_toggle = float("-inf")


class FaceBoxMemory:
    """Remembers last seen face boxes for a short grace period.

    When blur is on but the detector drops a frame, blurring the last
    known boxes is safer than flashing raw faces. Boxes older than the
    grace period are discarded (faces may have moved away).
    """

    def __init__(self, grace=0.75):
        self.grace = grace
        self._boxes = []
        self._time = float("-inf")

    def update(self, boxes, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        if boxes:
            self._boxes = [dict(b) for b in boxes]
            self._time = timestamp

    def boxes(self, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        if self._boxes and timestamp - self._time <= self.grace:
            return [dict(b) for b in self._boxes]
        return []


def blur_faces(frame, boxes, expand=0.15, kind="GAUSSIAN", strength=5):
    """Blur face regions in place. Returns number of faces blurred.

    Args:
        frame: BGR image (modified in place).
        boxes: list of {"x_min","y_min","x_max","y_max"} in 0.0-1.0.
        expand: grow each box by this fraction to cover the whole face.
        kind: "GAUSSIAN" or "PIXELATE".
        strength: 1-10, higher = stronger obfuscation.
    """
    strength = max(1, min(10, strength))
    h, w = frame.shape[:2]
    count = 0
    for b in boxes or []:
        bw = (b["x_max"] - b["x_min"]) * w
        bh = (b["y_max"] - b["y_min"]) * h
        if bw < 4 or bh < 4:
            continue
        ex, ey = bw * expand, bh * expand
        x1 = max(0, int((b["x_min"] * w) - ex))
        y1 = max(0, int((b["y_min"] * h) - ey))
        x2 = min(w, int((b["x_max"] * w) + ex))
        y2 = min(h, int((b["y_max"] * h) + ey))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue
        roi = frame[y1:y2, x1:x2]
        if kind == "PIXELATE":
            px = max(2, strength * 2)  # mosaic block size in px
            small = cv2.resize(roi, (max(1, (x2 - x1) // px),
                                     max(1, (y2 - y1) // px)),
                               interpolation=cv2.INTER_LINEAR)
            frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                             interpolation=cv2.INTER_NEAREST)
        else:
            k = max(3, (min(x2 - x1, y2 - y1) * strength // 25) | 1)
            k = min(k, 151)
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        count += 1
    return count
