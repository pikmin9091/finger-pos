"""Gesture Control engine: stable gestures/motion → allowlisted actions.

Detection (gesture strings + movement direction in) is fully separated
from execution (action names out); main.py maps names to actions.py.
Firing rules per action kind:
  edge     — fire once on transition into the trigger (gesture change or
             movement-direction change). Holding never refires.
  repeat   — fire on transition, then at most every `interval` while the
             trigger persists (volume ramps).
State machine per trigger: IDLE → CANDIDATE (dwell hold) → TRIGGERED →
COOLDOWN → WAIT_FOR_NEUTRAL (the trigger must disappear before it can
arm again). Repeat-kind actions are exempt from neutral-reset by design.
PINCH (stable) toggles a volume-adjust sub-mode: while active, pinch
distance (0.0-1.0 calibrated ratio) drives discrete volume_set targets
with deadzone + smoothing + throttle; other triggers are suppressed
except the master toggle. update() returns an action name, a
("volume_set", pct) / ("volume_mode", on) tuple, or None.
Every action has its own cooldown. A master toggle gates everything.
Gesture→action mapping is user-configurable but validated against the
allowlist — unknown actions are ignored, never executed.
"""

# pinch-ratio (thumb-index distance / hand scale) mapping + stability
VOLUME_RATIO_LO = 0.15
VOLUME_RATIO_HI = 0.90
VOLUME_DEADZONE = 0.03   # ignore ratio changes below this (jitter)
VOLUME_SMOOTH = 0.35     # EMA alpha on the ratio
VOLUME_SEND_EVERY = 0.4  # min seconds between volume_set targets
VOLUME_SEND_MIN = 2      # min percent change worth sending

# action name -> (kind, default cooldown s, notification label)
ALLOWLIST = {
    "media_play_pause": ("edge", 1.0, "PLAY/PAUSE"),
    "media_next": ("edge", 1.2, "NEXT TRACK"),
    "media_previous": ("edge", 1.2, "PREV TRACK"),
    "volume_up": ("repeat", 0.5, "VOLUME +"),
    "volume_down": ("repeat", 0.5, "VOLUME -"),
    "mute_toggle": ("edge", 1.0, "MUTE"),
    "workspace_next": ("edge", 1.5, "WORKSPACE +"),
    "workspace_prev": ("edge", 1.5, "WORKSPACE -"),
    "control_toggle": ("edge", 1.0, "CONTROL"),
}

# default trigger -> action (PINCH reserved: volume-mode when master is
# ON, privacy-blur toggle otherwise — never both at once)
DEFAULT_MAP = {
    "OPEN_PALM": "control_toggle",   # master on/off, hold to confirm
    "THUMBS_UP": "media_play_pause",
    "PEACE": "media_next",
    "FIST": "media_previous",
    "POINT": "mute_toggle",
    "THUMBS_DOWN": "volume_down",    # spare, user-remappable
}


class ControlEngine:
    """Stateful gesture→action mapper with cooldowns and master toggle."""

    PRESETS = {
        # decreasing sensitivity: calm needs confident, held gestures and
        # ignores hand motion entirely (motion triggers cause most accidents)
        "default": {"min_confidence": 0.5, "volume_interval": 0.5,
                    "dwell": 0.0, "motion": True},
        "calm": {"min_confidence": 0.8, "volume_interval": 1.0,
                 "dwell": 1.0, "motion": False},
    }

    def __init__(self, action_map=None, enabled=True, min_confidence=0.5,
                 volume_interval=0.5, dwell=0.0, motion=True, preset=None):
        if preset in self.PRESETS:
            p = self.PRESETS[preset]
            min_confidence = p["min_confidence"]
            volume_interval = p["volume_interval"]
            dwell = p["dwell"]
            motion = p["motion"]
        self.map = dict(DEFAULT_MAP)
        if action_map:
            for trigger, action in action_map.items():
                if action in ALLOWLIST:
                    self.map[trigger] = action
                # unknown actions are silently ignored (never executed)
        self.enabled = enabled
        self.min_confidence = min_confidence
        self.volume_interval = volume_interval
        self.dwell = max(0.0, dwell)  # seconds a trigger must persist before firing
        self.motion = motion          # False disables all MOVE_* triggers
        self._last_gesture = None
        self._last_move = "STATIONARY"
        self._last_fire = {}  # action -> timestamp
        self._pending = None  # (trigger, since) awaiting dwell
        self._need_neutral = set()  # fired edge triggers awaiting neutral
        self._pinch_since = None  # pinch hold start (dwell confirm)
        self._pinch_held = False
        self.volume_mode = False
        self._vol_smooth = None
        self._vol_sent = None
        self._vol_last = float("-inf")

    def _ready(self, action, timestamp, cooldown):
        if timestamp - self._last_fire.get(action, float("-inf")) < cooldown:
            return False
        self._last_fire[action] = timestamp
        return True

    def update(self, gesture, confidence, movement, timestamp,
                 pinch=False, pinch_ratio=None):
        """Fold one frame in.

        gesture: stabilized gesture string (or None when no hand).
        confidence: gesture confidence 0.0-1.0.
        movement: movement direction string.
        pinch: stable-PINCH currently held (volume-mode toggle).
        pinch_ratio: thumb-index distance / hand scale, or None.

        Returns an action name, a ("volume_set", pct) /
        ("volume_mode", on) tuple, or None.
        """
        pinch = bool(pinch)
        if pinch and not self._pinch_held:
            if self._pinch_since is None:
                self._pinch_since = timestamp
            if timestamp - self._pinch_since >= self.dwell:
                self._pinch_held = True
                if self.enabled:
                    self.volume_mode = not self.volume_mode
                    self._vol_smooth = pinch_ratio
                    self._vol_sent = None
                    self._vol_last = float("-inf")
                    return ("volume_mode", self.volume_mode)
        if not pinch:
            self._pinch_since = None
            self._pinch_held = False
        if not self.enabled:
            self.volume_mode = False

        if self.volume_mode and self.enabled:
            discrete = self._decide(gesture, confidence, movement, timestamp)
            if discrete == "control_toggle":
                self.volume_mode = False
                self.enabled = not self.enabled
                return discrete
            if pinch_ratio is not None:  # suppressed otherwise, state kept fresh
                return self._volume_adjust(pinch_ratio, timestamp)
            return None

        action = self._decide(gesture, confidence, movement, timestamp)
        if action == "control_toggle":
            self.volume_mode = False
            self.enabled = not self.enabled
        if isinstance(action, str):
            return action if self.enabled or action == "control_toggle" else None
        return action

    def _volume_adjust(self, ratio, timestamp):
        """Map pinch ratio to a throttled volume_set target. May return None."""
        if self._vol_smooth is None:
            self._vol_smooth = ratio
        self._vol_smooth = (VOLUME_SMOOTH * ratio
                            + (1 - VOLUME_SMOOTH) * self._vol_smooth)
        span = VOLUME_RATIO_HI - VOLUME_RATIO_LO
        pct = int(round(max(0.0, min(1.0, (self._vol_smooth - VOLUME_RATIO_LO)
                                             / span)) * 100))
        if ((self._vol_sent is None or abs(pct - self._vol_sent) >= VOLUME_SEND_MIN)
                and timestamp - self._vol_last >= VOLUME_SEND_EVERY):
            self._vol_sent = pct
            self._vol_last = timestamp
            return ("volume_set", pct)
        return None

    def _decide(self, gesture, confidence, movement, timestamp):
        cur_g = (gesture if gesture and gesture != "NONE"
                 and confidence >= self.min_confidence else None)
        # WAIT_FOR_NEUTRAL: fired edge triggers re-arm only after the
        # triggering gesture/motion visibly disappears.
        for trig in list(self._need_neutral):
            if trig.startswith("MOVE_"):
                if movement != trig[5:]:
                    self._need_neutral.discard(trig)
            elif cur_g != trig:
                self._need_neutral.discard(trig)
        new_g = cur_g if cur_g != self._last_gesture else None
        self._last_gesture = cur_g
        new_m = None
        if self.motion and movement and movement != "STATIONARY" \
                and movement != self._last_move:
            new_m = "MOVE_" + movement
        sustained = (self.motion and movement and movement != "STATIONARY"
                     and movement == self._last_move)
        self._last_move = movement or "STATIONARY"

        cands = [t for t in (new_g, new_m)
                 if t and self.map.get(t) in ALLOWLIST
                 and t not in self._need_neutral]
        if cands and (self._pending is None or self._pending[0] != cands[0]):
            self._pending = (cands[0], timestamp)
        if self._pending is not None:
            trig, since = self._pending
            present = (cur_g == trig) or (
                self.motion and movement and ("MOVE_" + movement) == trig)
            if not present:
                self._pending = None  # changed mind mid-dwell: cancel
            elif self.dwell <= 0 or timestamp - since >= self.dwell:
                self._pending = None
                action = self.map[trig]
                _, cd, _ = ALLOWLIST[action]
                if self._ready(action, timestamp, cd):
                    self._need_neutral.add(trig)  # repeat kinds exempt below
                    if ALLOWLIST[action][0] == "repeat":
                        self._need_neutral.discard(trig)
                    return action
        # sustained-motion volume ramps (repeat kind only)
        if sustained and movement in ("UP", "DOWN"):
            action = self.map.get("MOVE_" + movement)
            if action and ALLOWLIST.get(action, ("",))[0] == "repeat":
                if self._ready(action, timestamp, self.volume_interval):
                    return action
        return None

    def label(self, action):
        return ALLOWLIST.get(action, ("", 0, action))[2]
