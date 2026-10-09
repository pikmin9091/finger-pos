"""Allowlisted system-action executors for gesture control.

Media  : MPRIS over D-Bus via `busctl` (any player: Spotify, browsers…).
Volume : `wpctl` (WirePlumber/PipeWire).
Windows: `hyprctl` (Hyprland IPC).

No arbitrary shell: every action maps to a fixed argv template, run
without a shell, with a timeout, as the normal user (never root).
Missing tools/players degrade to (False, reason) — never an exception.
The `runner` is injectable so the whole layer unit-tests without hardware.
"""

import json
import re
import shutil
import subprocess

BUSCTL = "busctl"
WPCTL = "wpctl"
HYPRCTL = "hyprctl"
SINK = "@DEFAULT_AUDIO_SINK@"
VOLUME_STEP = "5%"
VOLUME_MAX = 1.0


class Runner:
    """Thin subprocess wrapper with timeout. Returns (rc, stdout, stderr)."""

    def __init__(self, timeout=3.0):
        self.timeout = timeout

    def run(self, argv):
        try:
            p = subprocess.run(argv, capture_output=True, text=True,
                               timeout=self.timeout)
            return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
        except Exception as e:  # timeout, missing binary, D-Bus down, …
            return 127, "", f"{type(e).__name__}: {e}"


def _safe(runner, argv):
    """runner.run() that can never raise (custom runners included)."""
    try:
        return runner.run(argv)
    except Exception as e:
        return 127, "", f"{type(e).__name__}: {e}"


def available(tool):
    return shutil.which(tool) is not None


class MediaPlayer:
    """MPRIS media control. Picks a player, prefers Spotify-likes."""

    def __init__(self, runner=None):
        self.runner = runner or Runner()

    def players(self):
        """MPRIS bus names currently owned, Spotify-likes first."""
        if not available(BUSCTL):
            return []
        rc, out, _ =  _safe(self.runner, 
            [BUSCTL, "--user", "list", "--no-pager", "--no-legend"])
        if rc != 0:
            return []
        names = sorted({line.split()[0] for line in out.splitlines()
                        if line.startswith("org.mpris.MediaPlayer2.")})
        return sorted(names, key=lambda n: (0 if "spotify" in n.lower() else 1, n))

    def _player(self):
        ps = self.players()
        return ps[0] if ps else None

    def _call(self, method):
        player = self._player()
        if player is None:
            return False, "no MPRIS player running"
        rc, _, err =  _safe(self.runner, 
            [BUSCTL, "--user", "call", player, "/org/mpris/MediaPlayer2",
             "org.mpris.MediaPlayer2.Player", method])
        return (True, "") if rc == 0 else (False, err or f"{method} failed")

    def play_pause(self):
        return self._call("PlayPause")

    def next(self):
        return self._call("Next")

    def previous(self):
        return self._call("Previous")

    def status(self):
        """PlaybackStatus string, or (False, reason)."""
        player = self._player()
        if player is None:
            return False, "no MPRIS player running"
        rc, out, err =  _safe(self.runner, 
            [BUSCTL, "--user", "get-property", player, "/org/mpris/MediaPlayer2",
             "org.mpris.MediaPlayer2.Player", "PlaybackStatus"])
        m = re.search(r'"([^"]+)"\s*$', out) if rc == 0 else None
        return (True, m.group(1)) if m else (False, err or "no status")

    def now_playing(self):
        """(True, 'artist - title') or (False, reason). Never raises."""
        player = self._player()
        if player is None:
            return False, "no MPRIS player running"
        rc, out, err =  _safe(self.runner, 
            [BUSCTL, "--user", "--json=short", "get-property", player,
             "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player",
             "Metadata"])
        if rc != 0:
            return False, err or "no metadata"
        try:
            data = json.loads(out).get("data", {})
            artist = data.get("xesam:artist", {}).get("data", ["?"])
            artist = ", ".join(artist) if isinstance(artist, list) else str(artist)
            title = data.get("xesam:title", {}).get("data", "?")
            return True, f"{artist} - {title}".strip(" -")
        except Exception as e:
            return False, f"metadata parse: {e}"


class Volume:
    """PipeWire volume via wpctl. Range clamped to 0.0–VOLUME_MAX."""

    def __init__(self, runner=None):
        self.runner = runner or Runner()

    def get(self):
        """(True, (level 0.0+, muted)) or (False, reason)."""
        if not available(WPCTL):
            return False, "wpctl not installed"
        rc, out, err =  _safe(self.runner, [WPCTL, "get-volume", SINK])
        if rc != 0:
            return False, err or "wpctl get-volume failed"
        m = re.search(r"Volume:\s*([0-9.]+)", out)
        if not m:
            return False, f"unparsable: {out!r}"
        return True, (float(m.group(1)), "[MUTED]" in out)

    def _set(self, arg):
        if not available(WPCTL):
            return False, "wpctl not installed"
        rc, _, err =  _safe(self.runner, [WPCTL, "set-volume", SINK, arg])
        return (True, "") if rc == 0 else (False, err or "set-volume failed")

    def up(self):
        ok, info = self.get()
        if not ok:
            return ok, info
        level, _ = info
        if level >= VOLUME_MAX:
            return True, "already max"
        return self._set(VOLUME_STEP + "+")

    def down(self):
        ok, info = self.get()
        if not ok:
            return ok, info
        level, _ = info
        if level <= 0.0:
            return True, "already min"
        return self._set(VOLUME_STEP + "-")

    def mute_toggle(self):
        if not available(WPCTL):
            return False, "wpctl not installed"
        rc, _, err =  _safe(self.runner, [WPCTL, "set-mute", SINK, "toggle"])
        return (True, "") if rc == 0 else (False, err or "set-mute failed")

    def set_absolute(self, pct):
        """Set volume to 0-100 (clamped). Used by pinch-distance control."""
        if not available(WPCTL):
            return False, "wpctl not installed"
        pct = max(0, min(100, int(pct)))
        rc, _, err =  _safe(self.runner, [WPCTL, "set-volume", SINK, f"{pct}%"])
        return (True, f"{pct}%") if rc == 0 else (False, err or "set-volume failed")

    def percent(self):
        """(True, (pct_int, muted)) for display, clamped 0–100+."""
        ok, info = self.get()
        if not ok:
            return ok, info
        level, muted = info
        return True, (int(round(level * 100)), muted)



class Hyprland:
    """Hyprland window control via hyprctl (probed, read-only first)."""

    def __init__(self, runner=None):
        self.runner = runner or Runner()

    def ok(self):
        if not available(HYPRCTL):
            return False
        rc, _, _ =  _safe(self.runner, [HYPRCTL, "version"])
        return rc == 0

    def active_workspace(self):
        """(True, workspace_id) or (False, reason). Read-only."""
        if not available(HYPRCTL):
            return False, "hyprctl not installed"
        rc, out, err =  _safe(self.runner, [HYPRCTL, "activeworkspace", "-j"])
        if rc != 0:
            return False, err or "activeworkspace failed"
        try:
            return True, int(json.loads(out)["id"])
        except Exception as e:
            return False, f"parse: {e}"

    def workspace_delta(self, delta):
        """Move to workspace (current + delta). Only explicit ±int."""
        if delta not in (1, -1):
            return False, "delta must be +1/-1"
        if not available(HYPRCTL):
            return False, "hyprctl not installed"
        ok, cur = self.active_workspace()
        if not ok:
            return False, cur
        target = max(1, cur + delta)
        rc, _, err =  _safe(self.runner, 
            [HYPRCTL, "dispatch", "workspace", str(target)])
        return (True, f"workspace {target}") if rc == 0 else (False, err or "dispatch failed")
