"""Tests for allowlisted system actions — fake runner, no hardware."""

import pytest
import actions
from actions import MediaPlayer, Volume, Hyprland


class Fake:
    """Scripted runner: argv-tuple -> (rc, stdout, stderr)."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def run(self, argv):
        self.calls.append(list(argv))
        key = tuple(argv)
        if key in self.script:
            return self.script[key]
        for prefix, res in self.script.items():
            if isinstance(prefix, str) and key[0].endswith(prefix):
                return res
        raise AssertionError(f"unexpected call: {argv}")


BUS = ("busctl", "--user", "list", "--no-pager", "--no-legend")
MPRIS = "org.mpris.MediaPlayer2.brave.instance1"
LIST = (BUS, 0, f"{MPRIS}  1  u :1.1  x -\n", "")


def test_media_calls_exact_mpris():
    call = ("busctl", "--user", "call", MPRIS, "/org/mpris/MediaPlayer2",
            "org.mpris.MediaPlayer2.Player", "PlayPause")
    f = Fake({BUS: (0, f"{MPRIS}  1  u :1.1  x -\n", ""),
              call: (0, "", "")})
    m = MediaPlayer(runner=f)
    assert m.play_pause() == (True, "")
    assert f.calls[-1] == list(call)


def test_media_no_player_graceful():
    f = Fake({BUS: (0, "", "")})
    m = MediaPlayer(runner=f)
    assert m.next() == (False, "no MPRIS player running")
    assert m.now_playing() == (False, "no MPRIS player running")


def test_media_now_playing_parses_json():
    meta = ('{"data":{"xesam:artist":{"type":"as","data":["A","B"]},'
            '"xesam:title":{"type":"s","data":"Song"}}}')

    class Sub(Fake):
        def run(self, argv):
            self.calls.append(list(argv))
            if argv[1:3] == ["--user", "list"]:
                return 0, f"{MPRIS}  1  u :1.1  x -\n", ""
            return 0, meta, ""
    m = MediaPlayer(runner=Sub({}))
    assert m.now_playing() == (True, "A, B - Song")


def test_volume_parse_and_clamp(monkeypatch):
    monkeypatch.setattr(actions, "available", lambda t: True)
    f = Fake({("wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"): (0, "Volume: 1.00", "")})
    v = Volume(runner=f)
    assert v.up() == (True, "already max")  # clamped, no set call
    assert v.percent() == (True, (100, False))


def test_volume_muted_parse_and_steps(monkeypatch):
    monkeypatch.setattr(actions, "available", lambda t: True)
    f = Fake({
        ("wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"): (0, "Volume: 0.45 [MUTED]", ""),
        ("wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%-"): (0, "", ""),
        ("wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"): (0, "", ""),
    })
    v = Volume(runner=f)
    assert v.percent() == (True, (45, True))
    assert v.down() == (True, "")
    assert v.mute_toggle() == (True, "")


def test_volume_tool_missing(monkeypatch):
    monkeypatch.setattr(actions, "available", lambda t: False)
    v = Volume(runner=Fake({}))
    assert v.get() == (False, "wpctl not installed")


def test_set_absolute_clamps(monkeypatch):
    monkeypatch.setattr(actions, "available", lambda t: True)
    f = Fake({("wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "55%"): (0, "", ""),
              ("wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "100%"): (0, "", ""),
              ("wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "0%"): (0, "", "")})
    v = Volume(runner=f)
    assert v.set_absolute(55) == (True, "55%")
    assert v.set_absolute(999) == (True, "100%")
    assert v.set_absolute(-3) == (True, "0%")


def test_runner_exception_never_raises():
    class Boom:
        def run(self, argv):
            raise OSError("nope")
    m = MediaPlayer(runner=Boom())
    assert m.play_pause()[0] is False


def test_hypr_workspace_math(monkeypatch):
    monkeypatch.setattr(actions, "available", lambda t: True)
    import json as J
    f = Fake({
        ("hyprctl", "activeworkspace", "-j"): (0, J.dumps({"id": 2}), ""),
        ("hyprctl", "dispatch", "workspace", "3"): (0, "", ""),
    })
    h = Hyprland(runner=f)
    assert h.workspace_delta(1) == (True, "workspace 3")
    assert h.workspace_delta(0) == (False, "delta must be +1/-1")


def test_hypr_missing_tool(monkeypatch):
    monkeypatch.setattr(actions, "available", lambda t: False)
    assert Hyprland(runner=Fake({})).workspace_delta(1)[0] is False
