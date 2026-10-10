"""Gesture Control engine: stable gestures/motion → allowlisted actions.

Detection (gesture strings + movement direction in) is fully separated
from execution (action names out); main.py maps names to actions.py.
Firing rules per action kind:
  edge     — fire once on transition into the trigger; re-arm only after
             a neutral pose or a stationary movement is observed.
State machine per trigger: IDLE → CANDIDATE (dwell hold) → TRIGGERED →
WAIT_FOR_NEUTRAL (the trigger must disappear before it can arm again).
PINCH (stable) toggles a volume-adjust sub-mode: while active, pinch
distance (0.0-1.0 calibrated ratio) drives discrete volume_set targets
with deadzone + smoothing + throttle; other triggers are suppressed
except the master toggle. update() returns an action name, a
("volume_set", pct) / ("volume_mode", on) tuple, or None.
Every action has its own cooldown. A master toggle gates everything.
Gesture→action mapping is user-configurable but validated against the
allowlist — unknown actions are ignored, never executed.
"""

# Volume-by-hand-height mapping + stability. y is normalized frame
# coords (0 top). Calibratable bounds; deadband + smoothing + throttle
# so landmark jitter never spams the audio server.
VOLUME_TOP = 0.25       # hand y mapping to 100%
VOLUME_BOTTOM = 0.75    # hand y mapping to 0%
VOLUME_Y_DEADBAND = 0.008  # ignore smaller raw moves (~4px @ 480p)
VOLUME_SMOOTH = 0.4     # EMA alpha on hand height
VOLUME_SEND_EVERY = 0.4  # min seconds between volume_set targets
VOLUME_SEND_MIN = 2      # min percent change worth sending
MOTION_NEUTRAL_SECONDS = 0.25  # require a deliberate pause before re-arming

# action name -> (kind, default cooldown s, notification label)
ALLOWLIST = {
    "media_play_pause": ("edge", 1.0, "PLAY/PAUSE"),
    "media_next": ("edge", 1.2, "NEXT TRACK"),
    "media_previous": ("edge", 1.2, "PREV TRACK"),
    "volume_up": ("edge", 0.5, "VOLUME +"),
    "volume_down": ("edge", 0.5, "VOLUME -"),
    "mute_toggle": ("edge", 1.0, "MUTE"),
    "workspace_next": ("edge", 1.5, "WORKSPACE +"),
    "workspace_prev": ("edge", 1.5, "WORKSPACE -"),
    "control_toggle": ("edge", 1.0, "CONTROL"),
}

# default trigger -> action. OPEN_PALM is contextual (see update):
# master off -> enables control; master on -> enters/exits volume mode.
# THUMBS_DOWN toggles the master off/on. PINCH is NOT mapped here — with
# master on it does nothing (privacy owns it only while master is off),
# so volume never depends on pinch reliability.
DEFAULT_MAP = {
    "OPEN_PALM": "control_toggle",   # contextual master/volume switch
    "THUMBS_UP": "media_play_pause",
    "PEACE": "media_next",
    "FIST": "media_previous",
    "POINT": "mute_toggle",
    "THUMBS_DOWN": "control_toggle",  # master off ("no" metaphor) / back on
}


class ControlEngine:
    """Stateful gesture→action mapper with cooldowns and master toggle."""

    PRESETS = {
        # decreasing sensitivity: calm needs confident, held gestures and
        # ignores hand motion entirely (motion triggers cause most accidents)
        "default": {"min_confidence": 0.5, "dwell": 0.0, "motion": True},
        "calm": {"min_confidence": 0.8, "dwell": 1.0, "motion": False},
    }

    def __init__(self, action_map=None, enabled=True, min_confidence=0.5,
                 dwell=0.0, motion=True, preset=None,
                 vol_top=None, vol_bottom=None):
        if preset in self.PRESETS:
            p = self.PRESETS[preset]
            min_confidence = p["min_confidence"]
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
        self.dwell = max(0.0, dwell)  # seconds a trigger must persist before firing
        self.motion = motion          # False disables all MOVE_* triggers
        self._last_gesture = None
        self._last_move = "STATIONARY"
        self._last_fire = {}  # action -> timestamp
        self._pending = None  # (trigger, since) awaiting dwell
        self._need_neutral = set()  # fired triggers awaiting neutral
        self._motion_latched = False  # one command per continuous movement
        self._stationary_since = None
        self.volume_mode = False
        self.volume_state = "INACTIVE"  # INACTIVE/ENTERING/ACTIVE/LOCKING/TRACKING_LOST
        self.vol_top = VOLUME_TOP if vol_top is None else vol_top
        self.vol_bottom = VOLUME_BOTTOM if vol_bottom is None else vol_bottom
        self._vol_smooth = None
        self._vol_sent = None
        self._vol_last = float("-inf")
        self._vol_reseed = False

    def _ready(self, action, timestamp, cooldown):
        if timestamp - self._last_fire.get(action, float("-inf")) < cooldown:
            return False
        self._last_fire[action] = timestamp
        return True

    def update(self, gesture, confidence, movement, timestamp,
                 hand_y=None, hand_present=True, gesture_stable=True):
        """Fold one frame in.

        gesture: stabilized gesture string (or None when no hand).
        confidence: gesture confidence 0.0-1.0.
        movement: movement direction string.
        hand_y: normalized hand height 0.0 (top) - 1.0, or None.
        hand_present: False when tracking lost (pauses volume, never zero).
        gesture_stable: debounced gesture reached its stability threshold.

        OPEN_PALM is contextual: master off -> enables control; master on
        -> enters/exits volume mode. THUMBS_DOWN toggles the master.
        Returns an action name, a ("volume_set", pct) /
        ("volume_mode", on) tuple, or None.
        """
        if self.volume_state == "LOCKING":
            # Exit edge was last frame; settle to INACTIVE now so HUD
            # shows the transition instead of skipping it.
            self._volume_off()
        if not hand_present:
            # Tracking lost: freeze edges, pause volume. A vanished hand
            # is NEVER interpreted as a position or a zero volume.
            self._pending = None
            self._last_gesture = None
            self._last_move = "STATIONARY"
            if self.volume_mode:
                self.volume_state = "TRACKING_LOST"
                self._vol_reseed = True  # re-anchor smoothing on return
            return None

        action, trig = self._decide(gesture, confidence, movement, timestamp,
                                    gesture_stable)
        if trig == "OPEN_PALM" and action is not None:
            if not self.enabled:
                self.enabled = True
                self._volume_off()
                return "control_toggle"
            if not self.volume_mode:
                self._enter_volume(hand_y, timestamp)
                return ("volume_mode", True)
            self._exit_volume()
            return ("volume_mode", False)
        if action == "control_toggle":
            # Any other mapped toggle (default: THUMBS_DOWN).
            self.enabled = not self.enabled
            if not self.enabled:
                self._volume_off()
            return "control_toggle"
        if not self.enabled:
            return None
        if self.volume_mode:
            # Volume ACTIVE suppresses discrete triggers; state already
            # advanced above so nothing queues up behind the adjustment.
            if hand_y is None:
                return None
            self.volume_state = "ACTIVE"
            return self._volume_height(hand_y, timestamp)
        return action

    def _volume_off(self):
        self.volume_mode = False
        self.volume_state = "INACTIVE"
        self._vol_smooth = None
        self._vol_sent = None

    def _enter_volume(self, hand_y, timestamp):
        self.volume_mode = True
        self.volume_state = "ACTIVE"
        self._vol_smooth = hand_y  # seed: no jump on entry
        self._vol_sent = None
        self._vol_reseed = False
        self._vol_last = float("-inf")

    def _exit_volume(self):
        self.volume_state = "LOCKING"
        self.volume_mode = False
        self._vol_smooth = None
        self._vol_sent = None

    def get_volume_status(self):
        """HUD/debug status: state, last sent target, bounds, deadband."""
        return {"state": self.volume_state,
                "target": self._vol_sent,
                "top": self.vol_top, "bottom": self.vol_bottom,
                "deadband": VOLUME_Y_DEADBAND}

    def _volume_height(self, y, timestamp):
        """Map hand height to a throttled volume_set target. May be None."""
        if self._vol_smooth is None or self._vol_reseed:
            self._vol_smooth = y  # entry/resume reseed: never jump
            self._vol_reseed = False
            self._vol_last = timestamp
            return None
        if abs(y - self._vol_smooth) >= VOLUME_Y_DEADBAND:
            self._vol_smooth = (VOLUME_SMOOTH * y
                                + (1 - VOLUME_SMOOTH) * self._vol_smooth)
        span = self.vol_bottom - self.vol_top
        pct = 50 if span <= 1e-9 else int(round(
            max(0.0, min(1.0, (self.vol_bottom - self._vol_smooth) / span))
            * 100))
        if ((self._vol_sent is None or abs(pct - self._vol_sent) >= VOLUME_SEND_MIN)
                and timestamp - self._vol_last >= VOLUME_SEND_EVERY):
            self._vol_sent = pct
            self._vol_last = timestamp
            return ("volume_set", pct)
        return None

    def _decide(self, gesture, confidence, movement, timestamp,
                gesture_stable=True):
        cur_g = (gesture if gesture_stable and gesture and gesture != "NONE"
                 and confidence >= self.min_confidence else None)
        if movement == "STATIONARY":
            if self._stationary_since is None:
                self._stationary_since = timestamp
            elif timestamp - self._stationary_since >= MOTION_NEUTRAL_SECONDS:
                self._motion_latched = False
        else:
            # A genuine direction change re-arms motion immediately — only
            # holding the SAME direction stays latched (one command per
            # continuous movement).
            if movement != self._last_move:
                self._motion_latched = False
            self._stationary_since = None
        # WAIT_FOR_NEUTRAL: fired edge triggers re-arm only after the
        # triggering gesture/motion visibly disappears.
        for trig in list(self._need_neutral):
            if trig.startswith("MOVE_"):
                if movement != trig[5:]:
                    self._need_neutral.discard(trig)
            elif gesture_stable and cur_g != trig:
                self._need_neutral.discard(trig)
        new_g = cur_g if cur_g != self._last_gesture else None
        if gesture_stable:
            self._last_gesture = cur_g
        new_m = None
        if (self.motion and not self._motion_latched and movement
                and movement != "STATIONARY"
                and movement != self._last_move):
            new_m = "MOVE_" + movement
        self._last_move = movement or "STATIONARY"

        cands = [t for t in (new_g, new_m)
                 if t and self.map.get(t) in ALLOWLIST
                 and t not in self._need_neutral]
        if cands and (self._pending is None or self._pending[0] != cands[0]):
            self._pending = (cands[0], timestamp)
        if self._pending is not None:
            trig, since = self._pending
            present = (gesture_stable and cur_g == trig) or (
                self.motion and movement and ("MOVE_" + movement) == trig)
            if not present:
                self._pending = None  # changed mind mid-dwell: cancel
            elif self.dwell <= 0 or timestamp - since >= self.dwell:
                self._pending = None
                action = self.map[trig]
                _, cd, _ = ALLOWLIST[action]
                if self._ready(action, timestamp, cd):
                    self._need_neutral.add(trig)
                    if trig.startswith("MOVE_"):
                        self._motion_latched = True
                    return action, trig
        return None, None

    def label(self, action):
        return ALLOWLIST.get(action, ("", 0, action))[2]

    def cooldown_remaining(self, timestamp):
        """Max seconds left on any action cooldown (0 when all clear)."""
        best = 0.0
        for action, (_, cd, _) in ALLOWLIST.items():
            left = cd - (timestamp - self._last_fire.get(action, float("-inf")))
            best = max(best, left)
        return max(0.0, round(best, 2))
