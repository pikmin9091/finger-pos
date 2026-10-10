"""Multi-hand / multi-face slot management.

Each visible hand gets its own FingerTracker + GestureDetector (and each
face its own FaceTracker) so smoothing and debounce state never leak
between subjects. Detections are matched to slots by spatial proximity
(wrist anchor for hands, centroid for faces) — NOT by list order — so a
reordered detector output can never swap two subjects' smoothing state.
Same-handedness slots are preferred as a tiebreak.
Slots for temporarily lost subjects keep counting missed frames
(detector/debounce resets after a prolonged loss, mirroring
FingerTracker.reset_after_lost); long-gone slots are pruned.
"""

import math

from tracker import FingerTracker
from gesture import GestureDetector
from face_tracker import FaceTracker

MAX_MATCH_DIST = 0.3   # max anchor jump (normalized units) to reuse a slot
MAX_SLOTS = 4          # hard cap so slots can never leak memory
PRUNE_AFTER_MISSED = 90  # drop hand slots unseen for this many updates


def _dist2d(a, b):
    if a is None or b is None:
        return float("inf")
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _anchor_hand(landmarks):
    """Wrist position as the per-hand identity anchor."""
    if landmarks:
        w = landmarks[0]
        return (w["x"], w["y"])
    return None


def _anchor_face(landmarks):
    """Centroid as the per-face identity anchor."""
    if not landmarks:
        return None
    n = len(landmarks)
    return (sum(l["x"] for l in landmarks) / n,
            sum(l["y"] for l in landmarks) / n)


def _unique_key(base, taken):
    key, n = base, 2
    while key in taken:
        key = f"{base}#{n}"
        n += 1
    taken.add(key)
    return key


class HandSlots:
    """Maps detected hands to persistent per-hand tracker slots."""

    def __init__(self, smoothing=0.2, pinch_threshold=0.06, debounce_frames=3,
                 calib=None, min_velocity=0.05):
        self.smoothing = smoothing
        self.pinch_threshold = pinch_threshold
        self.debounce_frames = debounce_frames
        self.calib = dict(calib or {})
        self.min_velocity = min_velocity
        self._slots = []
        self._frame = 0

    def set_smoothing(self, value):
        """Live-tune EMA alpha on stored default + existing slots."""
        value = max(0.0, min(1.0, value))
        self.smoothing = value
        for s in self._slots:
            s["tracker"].smoothing = value

    def set_debounce(self, frames):
        """Live-tune debounce frames on stored default + existing slots."""
        frames = max(1, int(frames))
        self.debounce_frames = frames
        for s in self._slots:
            s["detector"].debounce_frames = frames

    def _make_slot(self, key):
        detector = GestureDetector(pinch_threshold=self.pinch_threshold,
                                   debounce_frames=self.debounce_frames)
        if self.calib:
            detector.apply_calib(self.calib)
        return {
            "key": key,
            "label": key,
            "votes": [],
            "tracker": FingerTracker(smoothing=self.smoothing,
                                       min_velocity=self.min_velocity),
            "detector": detector,
            "missed": 0,
            "anchor": None,
            "seen": 0,
        }

    @staticmethod
    def _vote(slot, label):
        """Majority vote over recent labels; MediaPipe handedness flickers
        frame-to-frame (seen in real logs: slot id vs reported hand
        disagreeing), so the raw per-frame label is never trusted alone."""
        if label is not None:
            slot["votes"].append(label)
            del slot["votes"][:-8]
        if not slot["votes"]:
            return label
        counts = {}
        for v in slot["votes"]:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=lambda k: (counts[k], -slot["votes"].index(k)))

    def _match(self, anchor, label, used):
        # Any previously seen slot is a candidate (not just last frame),
        # so a briefly dropped detection rejoins its slot instead of
        # restarting smoothing/debounce. Stale slots carry reset state
        # (see update), so rejoining them is always safe.
        cands = [s for s in self._slots
                 if s["seen"] < self._frame and id(s) not in used]
        if label is not None:
            same = [s for s in cands if s["label"] == label]
            if same:
                # Never steal another hand's slot when a same-label
                # slot exists — prefer it even if it is slightly farther.
                cands = same
        ranked = sorted(cands, key=lambda s: (0 if s["key"] == label else 1,
                                              _dist2d(anchor, s["anchor"]),
                                              self._slots.index(s)))
        if ranked and _dist2d(anchor, ranked[0]["anchor"]) <= MAX_MATCH_DIST:
            return ranked[0]
        return None

    def update(self, hands, timestamp=None):
        """Update slots from a detect_hands() list.

        Args:
            hands: list of {landmarks, handedness, confidence}
            timestamp: seconds since epoch (float)

        Returns:
            list of per-hand data dicts (same shape as
            FingerTracker.update plus a "gesture" key), in input order.
        """
        self._frame += 1
        entries = []
        used, taken = set(), {s["key"] for s in self._slots}
        for i, h in enumerate(hands or []):
            label = h.get("handedness")
            anchor = _anchor_hand(h.get("landmarks") or [])
            slot = self._match(anchor, label, used)
            if slot is None:
                if len(self._slots) >= MAX_SLOTS:  # evict stalest
                    self._slots.sort(key=lambda s: s["missed"])
                    self._slots.pop()
                slot = self._make_slot(_unique_key(label or f"hand{i}", taken))
                self._slots.append(slot)
            used.add(id(slot))
            slot["seen"] = self._frame
            slot["missed"] = 0
            voted = self._vote(slot, label)
            if voted is not None:
                slot["label"] = voted
            if anchor is not None:
                slot["anchor"] = anchor
            landmarks = h.get("landmarks") or []
            entry = slot["tracker"].update(
                landmarks,
                handedness=voted,
                confidence=h.get("confidence", 0.0),
                timestamp=timestamp,
            )
            entry["gesture"] = slot["detector"].detect(
                landmarks, handedness=voted)
            entry["id"] = slot["key"]  # stable tracking id for analytics
            entries.append(entry)

        for slot in self._slots:
            if id(slot) not in used:
                slot["tracker"].update([], timestamp=timestamp)
                slot["missed"] += 1
                if slot["missed"] >= slot["tracker"].reset_after_lost:
                    slot["detector"].reset()
        self._slots = [s for s in self._slots if s["missed"] <= PRUNE_AFTER_MISSED]
        return entries

    def __len__(self):
        return len(self._slots)


class FaceSlots:
    """Maps detected faces to persistent per-face tracker slots."""

    def __init__(self, smoothing=0.4):
        self.smoothing = smoothing
        self._slots = []
        self._frame = 0
        self._next_id = 0

    def update(self, faces):
        """Update slots from a detect_faces() list.

        Returns per-face data dicts ordered left-to-right by bbox.
        """
        self._frame += 1
        entries = []
        used = set()
        for f in faces or []:
            landmarks = f.get("landmarks") or []
            anchor = _anchor_face(landmarks)
            best, best_d = None, MAX_MATCH_DIST
            for s in self._slots:
                if s["seen"] < self._frame and id(s) not in used:
                    d = _dist2d(anchor, s["anchor"])
                    if d <= best_d:
                        best, best_d = s, d
            if best is None:
                if len(self._slots) >= MAX_SLOTS:
                    self._slots.sort(key=lambda s: s["missed"])
                    self._slots.pop()
                self._next_id += 1
                best = {"id": f"face_{self._next_id}",
                        "tracker": FaceTracker(smoothing=self.smoothing),
                        "missed": 0, "anchor": None, "seen": 0}
                self._slots.append(best)
            used.add(id(best))
            best["seen"] = self._frame
            best["missed"] = 0
            if anchor is not None:
                best["anchor"] = anchor
            entry = best["tracker"].update(landmarks,
                                           confidence=f.get("confidence", 0.0))
            if entry is not None:
                entry["id"] = best["id"]  # stable tracking id for analytics
                entries.append(entry)

        for s in self._slots:
            if id(s) not in used:
                s["missed"] += 1
                if s["missed"] > 10:  # occluded a while → reseed smoothing on return
                    s["tracker"].reset()
        self._slots = [s for s in self._slots if s["missed"] <= 30]
        entries.sort(key=lambda e: e["box"]["x_min"])
        return entries

    def __len__(self):
        return len(self._slots)
