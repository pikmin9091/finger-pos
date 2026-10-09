"""Tests for Real-Time Object Analytics — synthetic only, no camera."""

import json
import csv
import pytest
from analytics import AnalyticsSession


def hand(i="Right", gesture="POINT", conf=0.9):
    return {"id": i, "hand": i, "confidence": conf,
            "gesture": {"gesture": gesture, "confidence": 0.9, "stable": True}}


def face(i="face_1", looking="CENTER", conf=1.0):
    return {"id": i, "confidence": conf, "looking": looking,
            "box": {"x_min": 0.3, "y_min": 0.2, "x_max": 0.6, "y_max": 0.6}}


def test_empty_session():
    s = AnalyticsSession()
    s.update([], [], 5.0, 30.0, timestamp=100.0)
    snap = s.snapshot()
    assert snap["frames"] == 1
    assert snap["unique_hands"] == 0 and snap["active"] == 0
    assert snap["avg_confidence"] == 0.0
    assert snap["gestures"] == {} and snap["recent_events"] == []


def test_counts_and_confidence_from_real_entries():
    s = AnalyticsSession()
    s.update([hand("Right", "POINT", 0.8), hand("Left", "FIST", 1.0)],
             [face(conf=0.6)], 10.0, 30.0, timestamp=0.0)
    snap = s.snapshot()
    assert snap["unique_hands"] == 2 and snap["unique_faces"] == 1
    assert snap["active"] == 3
    assert snap["avg_confidence"] == pytest.approx((0.8 + 1.0 + 0.6) / 3)
    assert snap["gestures"] == {"POINT": 1, "FIST": 1}
    assert snap["looking"] == {"CENTER": 1}
    assert snap["infer_ms"] == 10.0 and snap["fps"] == 30.0


def test_enter_exit_events_with_duration():
    s = AnalyticsSession()
    s.update([hand("Right")], [], 1.0, 30.0, timestamp=10.0)
    s.update([hand("Right")], [], 1.0, 30.0, timestamp=11.0)
    s.update([], [], 1.0, 30.0, timestamp=13.0)  # exit
    kinds = [e["event"] for e in s.events]
    assert kinds == ["enter", "exit"]
    assert s.events[-1]["duration_s"] == pytest.approx(3.0)
    assert s.events[-1]["frames"] == 2
    assert s.snapshot()["active"] == 0


def test_reentry_not_double_unique_but_counts_entered():
    s = AnalyticsSession()
    s.update([hand("Right")], [], 1.0, 30.0, timestamp=0.0)
    s.update([], [], 1.0, 30.0, timestamp=1.0)
    s.update([hand("Right")], [], 1.0, 30.0, timestamp=2.0)
    snap = s.snapshot()
    assert snap["unique_hands"] == 1   # same id → not new
    assert snap["entered"] == 2 and snap["exited"] == 1


def test_history_and_events_bounded():
    s = AnalyticsSession(history_len=10, event_limit=10)
    for i in range(30):
        s.update([hand(f"R{i}")], [], 1.0, 30.0, timestamp=float(i))
    assert len(s.history) == 10
    assert len(s.events) == 10


def test_reset_clears():
    s = AnalyticsSession()
    s.update([hand()], [face()], 1.0, 30.0, timestamp=0.0)
    s.reset()
    snap = s.snapshot()
    assert (snap["frames"], snap["unique_hands"], snap["active"],
            snap["entered"]) == (0, 0, 0, 0)


def test_export_roundtrip(tmp_path):
    s = AnalyticsSession()
    s.update([hand()], [face()], 4.0, 25.0, timestamp=7.0)
    s.update([], [], 4.0, 25.0, timestamp=8.0)
    jp = s.export_json(str(tmp_path / "s.json"))
    cp = s.export_csv(str(tmp_path / "s.csv"))
    data = json.loads(open(jp).read())
    assert data["snapshot"]["frames"] == 2
    assert len(data["events"]) == 4  # 2 enters + 2 exits
    rows = list(csv.reader(open(cp)))
    assert rows[0] == ["timestamp", "hands", "faces"]
    assert len(rows) == 3  # header + 2 frames
    assert rows[1][1:] == ["1", "1"]
