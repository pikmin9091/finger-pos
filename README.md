# Finger POS

Real-time hand tracking with MediaPipe — fingertip positions, movement tracking, and gesture recognition. Outputs normalized coordinates as JSON for integration into POS / touchless interfaces.

## Features

- **Fingertip tracking** — normalized `x, y, z` (0.0–1.0) for all 5 fingertips
- **Position classification** — `LEFT/CENTER/RIGHT` and `TOP/CENTER/BOTTOM` per finger
- **Movement tracking** — direction (`UP/DOWN/LEFT/RIGHT/STATIONARY`) + velocity (units/sec)
- **Gesture recognition** — `POINT`, `PEACE`, `THUMBS_UP`, `THUMBS_DOWN`, `OPEN_PALM`, `FIST`, `PINCH` with confidence and stability flag
- **Smoothing** — exponential moving average (EMA) with automatic reset after hand loss
- **Debouncing** — gestures must persist several frames before switching (anti-flicker)
- **JSON output** — one line per frame, easy to pipe into other tools

## Requirements

- Python 3.10+
- `opencv-python-headless`, `mediapipe`, `numpy` (see `requirements.txt`)
- `hand_landmarker.task` model (included; or download from [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker))

## Install

```bash
git clone https://github.com/pikmin9091/finger-pos
cd finger-pos
pip install -r requirements.txt
```

## Usage

### Live camera

```bash
python3 main.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | camera index |
| `--smoothing` | `0.2` | EMA alpha (lower = smoother, more lag) |
| `--debounce` | `3` | frames a new gesture must persist |
| `--pinch-threshold` | `0.06` | thumb–index distance for PINCH |

A preview window shows the hand skeleton, current gesture, and FPS counter. Press `q` to quit. Without a display (`DISPLAY` unset), it outputs JSON only.

### Static image

```bash
python3 cli_static.py path/to/image.jpg
```

### JSON output format

```json
{
  "hand": "Right",
  "confidence": 0.9876,
  "fingers": {
    "index_tip": {"x": 0.512, "y": 0.301, "z": -0.02, "horizontal": "CENTER", "vertical": "TOP"}
  },
  "hand_center": {"x": 0.5, "y": 0.45, "z": 0.0},
  "movement": {"direction": "RIGHT", "velocity_x": 1.24, "velocity_y": 0.03},
  "gesture": {"gesture": "POINT", "confidence": 0.9, "stable": true}
}
```

## Tests

```bash
python3 -m pytest
```

52 unit tests using synthetic landmarks (no webcam needed).

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | live camera loop (tracking + gesture + window/JSON) |
| `detect.py` | MediaPipe HandLandmarker wrapper |
| `tracker.py` | fingertip positions, smoothing, movement |
| `gesture.py` | deterministic gesture classification |
| `positions.py` | position classification + OpenCV drawing helpers |
| `capture.py` | camera abstraction |
| `cli_static.py` | single-image CLI |

## Notes

- Performance is CPU-bound by MediaPipe inference (~20 FPS at 640×480 on a typical CPU).
- Handedness follows MediaPipe convention (mirrored for selfie cameras).
