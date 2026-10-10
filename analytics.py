"""Real-time object analytics from actual detection output.

Every number comes from the live pipeline (hand/face entries, measured
inference latency, FPS counter). Unavailable metrics are reported as
"N/A" — never fabricated. Unique-object tracking reuses the stable
slot ids from multi.py ("hand:Right", "face:1", ...), so one subject is
never double-counted across frames. History and event logs are bounded
so RAM usage stays flat. Storage is opt-in (export_*); nothing is
recorded automatically and no images ever touch this module.
"""

import csv
import json
import time
from collections import Counter, deque

try:
    import psutil as _psutil
    _PROC = _psutil.Process()
except Exception:  # optional dependency — resource stats become "N/A"
    _psutil, _PROC = None, None


def _rss_mb():
    if _PROC is None:
        return None
    try:
        return round(_PROC.memory_info().rss / 1e6, 1)
    except Exception:
        return None


class AnalyticsSession:
    """Accumulates per-frame detection stats for one camera session."""

    def __init__(self, history_len=120, event_limit=200, exit_grace=5):
        self.history_len = history_len
        self.event_limit = event_limit
        # Consecutive missed frames before an exit is emitted. Single-frame
        # detector gaps must not spam enter/exit pairs (seen in real logs:
        # 13 pairs in 13.6 s for one face).
        self.exit_grace = max(1, exit_grace)
        self.reset()

    def reset(self):
        self.frames = 0
        self.t0 = None
        self.active = {}            # sid -> {kind, first, last, frames}
        self._ever = {}             # sid -> None, insertion-ordered, capped
        self.unique_hands = 0
        self.unique_faces = 0
        self.entered = 0
        self.exited = 0
        self.gestures = Counter()
        self.looking = Counter()
        self.conf_sum = 0.0
        self.conf_n = 0
        self.history = deque(maxlen=self.history_len)  # (t, hands, faces)
        self.events = deque(maxlen=self.event_limit)   # dicts
        self.infer_ms = 0.0
        self.fps = 0.0

    def update(self, hands, faces, infer_ms, fps, timestamp=None):
        """Fold one frame of detection output into the session."""
        if timestamp is None:
            timestamp = time.time()
        if self.t0 is None:
            self.t0 = timestamp
        hands = hands or []
        faces = faces or []
        self.frames += 1
        self.infer_ms = infer_ms
        self.fps = fps
        self.history.append((round(timestamp, 3), len(hands), len(faces)))

        for h in hands:
            self.conf_sum += h.get("confidence", 0.0)
            self.conf_n += 1
            g = (h.get("gesture") or {}).get("gesture", "NONE")
            self.gestures[g] += 1
        for f in faces:
            self.conf_sum += f.get("confidence", 0.0)
            self.conf_n += 1
            self.looking[f.get("looking", "CENTER")] += 1

        seen = {}
        for h in hands:
            seen[f"hand:{h.get('id', '?')}"] = ("hand", h.get("hand"))
        for f in faces:
            seen[f"face:{f.get('id', '?')}"] = ("face", f.get("looking"))

        for sid, (kind, label) in seen.items():
            if sid not in self.active:
                self.active[sid] = {"kind": kind, "first": timestamp,
                                    "frames": 0, "missed": 0}
                if sid not in self._ever:  # re-entries are not new uniques
                    self._ever[sid] = None
                    if len(self._ever) > 2000:  # flat RAM: forget oldest
                        self._ever.pop(next(iter(self._ever)))
                    if kind == "hand":
                        self.unique_hands += 1
                    else:
                        self.unique_faces += 1
                self.entered += 1
                self.events.append({"t": round(timestamp, 3), "event": "enter",
                                    "id": sid, "label": label})
            self.active[sid]["frames"] += 1
            self.active[sid]["last"] = timestamp
            self.active[sid]["missed"] = 0

        for sid in [s for s in self.active if s not in seen]:
            info = self.active[sid]
            info["missed"] += 1
            if info["missed"] < self.exit_grace:
                continue  # probably a brief detection gap, not a real exit
            self.active.pop(sid)
            self.exited += 1
            self.events.append({"t": round(timestamp, 3), "event": "exit",
                                "id": sid,
                                "duration_s": round(info["last"] - info["first"], 2),
                                "frames": info["frames"]})

    def snapshot(self):
        """Small JSON-serializable summary for JSON output."""
        avg_conf = round(self.conf_sum / self.conf_n, 4) if self.conf_n else 0.0
        return {
            "frames": self.frames,
            "session_s": round(time.time() - self.t0, 1) if self.t0 else 0.0,
            "unique_hands": self.unique_hands,
            "unique_faces": self.unique_faces,
            "active": len(self.active),
            "entered": self.entered,
            "exited": self.exited,
            "avg_confidence": avg_conf,
            "gestures": dict(self.gestures),
            "looking": dict(self.looking),
            "infer_ms": round(self.infer_ms, 2),
            "fps": round(self.fps, 1),
            "rss_mb": _rss_mb() if _rss_mb() is not None else "N/A",
            "recent_events": list(self.events)[-5:],
        }

    def export_json(self, path):
        with open(path, "w") as fh:
            json.dump({"snapshot": self.snapshot(),
                       "events": list(self.events)}, fh, indent=2)
        return path

    def export_csv(self, path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp", "hands", "faces"])
            w.writerows(self.history)
        return path
