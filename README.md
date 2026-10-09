# Finger POS

Real-time hand + face tracking with MediaPipe — fingertip positions, movement tracking, gesture recognition, and face position. Outputs normalized coordinates as JSON for integration into POS / touchless interfaces.

## Features

- **Multi-hand tracking** — up to 2 hands, each with its own smoothing + gesture debounce slot (matched by wrist-anchor proximity across frames)
- **Multi-face tracking** — up to 4 faces, ordered left-to-right
- **Gesture-controlled face blur** — stable PINCH toggles Gaussian blur on all visible faces (edge-triggered + cooldown, state persists with no hands); last-known boxes cover brief detection drops
- **Smart Privacy Mode** — `OFF / FACE_BLUR / STRICT / AUTO` with configurable rules, Gaussian or pixelate obfuscation, PINCH or `M`/`P` keys to switch
- **Real-time analytics** — live counts, stable tracking IDs, enter/exit events with durations, gesture/looking distributions, inference latency, session export to JSON/CSV
- **Text-only overlay** — one thin box per detected object plus plain status lines (gesture, face, blur, FPS), no distracting graphics
- **Gesture Control Center** (`--control`) — hand gestures drive media (MPRIS), volume (`wpctl`), and Hyprland workspaces (`hyprctl`) via an allowlist; per-action cooldowns, master toggle, text status lines + notifications

- **Fingertip tracking** — normalized `x, y, z` (0.0–1.0) for all 5 fingertips
- **Position classification** — `LEFT/CENTER/RIGHT` and `TOP/CENTER/BOTTOM` per finger
- **Movement tracking** — direction (`UP/DOWN/LEFT/RIGHT/STATIONARY`) + velocity (units/sec)
- **Gesture recognition** — `POINT`, `PEACE`, `THUMBS_UP`, `THUMBS_DOWN`, `OPEN_PALM`, `FIST`, `PINCH` with confidence and stability flag
- **Face position** — face center, bounding box, screen zone, proximity (`CLOSE/MEDIUM/FAR`), and head orientation (`yaw/pitch` + looking `LEFT/RIGHT/UP/DOWN/CENTER`, viewer perspective)
- **Smoothing** — exponential moving average (EMA) with automatic reset after hand loss
- **Debouncing** — gestures must persist several frames before switching (anti-flicker)
- **JSON output** — one line per frame, easy to pipe into other tools

## Requirements

- Python 3.10+
- `opencv-python-headless`, `mediapipe`, `numpy` (see `requirements.txt`)
- `hand_landmarker.task` model (included; or download from [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker))
- `face_landmarker.task` model (included; or download from [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker))

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
| `--no-face` | off | disable face detection (hand tracking only) |
| `--no-blur` | off | disable gesture-controlled face blur |
| `--blur-cooldown` | `1.0` | min seconds between blur on/off toggles |
| `--privacy` | `OFF` | initial privacy mode: `OFF/FACE_BLUR/STRICT/AUTO` |
| `--blur-kind` | `GAUSSIAN` | obfuscation style: `GAUSSIAN/PIXELATE` |
| `--blur-strength` | `5` | obfuscation strength 1–10 |
| `--auto-faces` | `1` | AUTO privacy engages at this face count |
| `--control` | off | enable gesture control (media/volume/Hyprland) |
| `--control-profile` | `default` | `default` or `calm` (confident held gestures only, no motion) |
| `--no-motion` | off | disable motion/swipe triggers in gesture control |

Sensitivity ranking (most → least false triggers): motion swipes >
quick static gestures > held static gestures (`calm`). Recommended daily
driver: `--control` (confidence ≥0.8, 0.7 s hold, neutral-reset). If still
twitchy: `--control --control-profile calm` (1.0 s hold, swipes off) and/or
raise `--debounce` (e.g. `5`).

### Gesture control (`--control`)

Opt-in only. Primary hand drives allowlisted actions. Overlay shows
`CONTROL: ON/OFF`, `VOL:`, `NOW:` lines plus transient notifications
(`NEXT TRACK`, `VOLUME 60%`, `workspace 3`).

| Gesture (hold) | Action |
|---|---|
| Open palm ~1 s ✋ | master control on/off |
| Thumbs up 👍 | play/pause |
| Peace ✌️ | next track |
| Fist ✊ | previous track |
| Index up ☝️ | mute toggle |
| Pinch (master ON) | enter/exit volume mode — pinch spread sets volume |
| Pinch (master OFF / no `--control`) | privacy-blur toggle (legacy, unchanged) |

PINCH never does two things at once: with master ON it owns volume
mode; otherwise it owns the privacy toggle. Thumbs-down is a spare
(volume step, remappable in `control.py:DEFAULT_MAP`).

**Volume mode:** pinch in → `VOLUME MODE` line appears → spread fingers
to raise, close to lower (calibrated to hand size, 3% deadzone, smoothed,
sent at most every 0.4 s) → pinch again to lock. Open palm exits to
master toggle as usual.

**Anti-misfire rules (defaults):** confidence ≥0.8, hold-to-fire 0.7 s
(`--control-profile calm`: 1.0 s + no motion), per-action cooldowns
1–1.5 s, neutral-reset (a trigger must disappear before re-arming),
debounce + hysteresis upstream, master gate. Tune via `--debounce`,
`--control-profile calm`, `--no-motion`.

Per-action cooldowns + gesture-transition edges prevent repeats while held;
low-confidence frames never fire. No player/tool → clear `(failed)` notice,
no crash. JSON gains a `control` block
(`enabled, volume_mode, last_action, volume_pct, muted, now_playing`).

CachyOS needs no extra packages: `busctl` (MPRIS, any player incl.
Spotify/browser), `wpctl` (volume), `hyprctl` (workspaces) ship with the
system. Mapping gesture→action is validated against the allowlist in
`control.py` (`DEFAULT_MAP`); arbitrary shell commands cannot be triggered.

A text-only preview window shows one thin box per detected object
(green = hand with handedness + gesture label, blue = face with looking
label), hand skeleton, gesture/face/blur status lines, control status,
and expiring toast messages for every event (mode change, action,
export, reset) — the camera window is self-sufficient, no terminal
needed. FPS counter included. Press `q` to quit. Without a display
(`DISPLAY` unset), it outputs JSON only.

> CachyOS/Hyprland notes: the app runs native Wayland (no `QT_QPA_PLATFORM`
> override needed; XWayland/X11 works too). This repo's venv uses
> `opencv-python-headless`, which has **no window support** — for the camera
> window (boxes, status lines, toast notifications, keyboard keys) install
> the GUI build in your venv instead:
> `./myenv/bin/pip uninstall -y opencv-python-headless && ./myenv/bin/pip install opencv-python`
> (same API, enables `imshow`). Headless is fine
> for JSON-only piping. All inference is CPU-local.

### Phone camera via v4l2loopback

OpenCV opens `/dev/videoN` by its number, so pass it to `--camera`:

```bash
# terminal 1: feed the phone stream into the loopback device (keep running)
ffmpeg -i http://192.168.1.71:8080/video \
        -vf format=yuv420p \
        -f v4l2 /dev/video10
# terminal 2: phone cam
python3 main.py --camera 10 [--control]
# USB webcam instead (skip /dev/video1 — usually a metadata node, not video)
python3 main.py --camera 0
```

Quick device check: `python3 -c "from capture import Camera; c=Camera(10); print(c.read()[0]); c.release()"` → `True` means readable.

### Static image

```bash
python3 cli_static.py path/to/image.jpg
python3 cli_static.py path/to/image.jpg --blur --out preview.jpg  # save blurred preview
```

### Controls (preview window)

| Key | Action |
|-----|--------|
| `M` | cycle privacy mode OFF→FACE_BLUR→STRICT→AUTO |
| `P` | PINCH-equivalent privacy toggle OFF↔blur mode |
| `S` | export session to `analytics_session_<ts>.json` + `.csv` |
| `R` | reset analytics session |
| `Q` | quit |
| `C` | gesture control master on/off (only with `--control`) |

PINCH gesture (thumb + index together until `[STABLE]`) behaves like `P`
when gesture-control master is OFF (or without `--control`); with master ON
it owns volume mode instead — never both at once.

### Gesture-controlled face blur

PINCH (thumb tip + index tip together, held ~0.5 s until `[STABLE]`)
toggles face blur on/off — unless gesture-control master is ON, in which
case PINCH owns volume mode and privacy is switched via `M`/`P` keys.
The toggle fires once per pinch (rising edge +
1 s cooldown), so holding the pinch never retriggers; the on/off state
persists even when no hand is visible. The preview shows
`BLUR: ON (N face(s))` in green (grey when off). Briefly dropped face
detections fall back to last-known boxes (<0.75 s) so raw faces never
flash; with `--no-face` there is nothing to blur (count stays 0).

To test the gesture without a camera, photograph your own pinch and run
`cli_static.py` — a PINCH line ends with `<-- blur toggle gesture`.

## Privacy & limitations

- 100% local processing (MediaPipe + OpenCV). No video is uploaded,
  recorded, or saved — `main.py` never writes image files; `cli_static.py`
  only saves when you pass `--out` explicitly.
- Face detection is not perfect: extreme angles, occlusions, and tiny
  faces can be missed, and missed faces cannot be blurred. The on-screen
  face count tells you what is actually covered. Face blur reduces
  identifiability but does not make video fully anonymous (clothing,
  gait, context remain).
- STRICT mode widens blur to an approximated head region (no
  body-segmentation model in this project) — silhouettes stay visible.
- Hand/face slot matching uses position continuity; very fast crossings
  may briefly swap identities.

## Performance (measured, blank 640×480, CPU)

- `detect_hands`: ~15 ms/frame, `detect_faces`: ~6 ms/frame (detection-only
  path; real subjects cost more). Single-threaded loop, no frame queue to
  pile up; overlay drawing is plain OpenCV (~1–2 ms).
- No GPU backend is forced; MediaPipe runs on CPU/XNNPACK here.
- Tune down with `--no-face` or lower camera resolution if
  the CPU saturates.

### JSON output format

One object per frame with `hands` (0–2 entries) and `faces` (0–4 entries):

```json
{
  "hands": [
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
  ],
  "faces": [
    {
      "present": true, "confidence": 1.0,
      "center": {"x": 0.5, "y": 0.4},
      "box": {"x_min": 0.35, "y_min": 0.2, "x_max": 0.65, "y_max": 0.6},
      "size": 0.12, "proximity": "MEDIUM",
      "horizontal": "CENTER", "vertical": "CENTER",
      "yaw": 0.02, "pitch": -0.01, "looking": "CENTER"
    }
  ],
  "blur": {"enabled": true, "faces_blurred": 1},
  "privacy": {"mode": "FACE_BLUR", "active": true, "kind": "GAUSSIAN", "reason": "mode"},
  "control": {"enabled": true, "volume_mode": false, "last_action": "media_next",
              "volume_pct": 60, "muted": false, "now_playing": "Artist - Title"},
  "analytics": {"frames": 120, "unique_hands": 1, "unique_faces": 1, "active": 2,
                "avg_confidence": 0.95, "infer_ms": 21.4, "fps": 29.8, "rss_mb": "N/A"}
}
```

`rss_mb` is `"N/A"` unless the optional `psutil` package is installed
(`./myenv/bin/pip install psutil` — pure-Python wheels, no system changes).
All analytics numbers come from the live pipeline; unavailable metrics
report `"N/A"`, never fabricated values.

## Stability

Six layers keep the output steady on live video:

| Layer | Where | What |
|-------|-------|------|
| Temporal model tracking | `detect.py`, `face.py` | `VIDEO` running mode propagates landmarks frame-to-frame instead of redetecting from scratch |
| Position-based slot matching | `multi.py` | hands/faces matched to tracker slots by wrist anchor / centroid proximity, never by detector list order — reordered output can't swap identities; brief gaps rejoin the same slot |
| Teleport guard | `tracker.py` | per-frame tip displacement clamped to `max_jump` (default 0.5): real fast motion converges in a few frames, one-frame spikes barely register |
| EMA smoothing | `tracker.py`, `face_tracker.py` | exponential moving average on tips and on face center/box/yaw/pitch; per-subject state, auto-reset after hand/face loss |
| Gesture debounce + hysteresis | `gesture.py` | new gestures must persist `debounce_frames`; PINCH exits at 1.5× the enter threshold; confidence is EMA-smoothed and always describes the returned (debounced) gesture |
| Control state machine | `control.py` | hold-to-fire dwell, per-action cooldowns, WAIT_FOR_NEUTRAL re-arm, master gate; pinch-hold enters volume mode with deadzone + throttled targets |
| Slot hygiene | `multi.py` | missed-frame counters, debounce reset after prolonged loss, stale-slot pruning, hard slot cap |

## Tests

```bash
python3 -m pytest
```

157 unit tests (synthetic landmarks + mocked action runner, no webcam/hardware needed).

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | live camera loop (tracking + gesture + window/JSON) |
| `detect.py` | MediaPipe HandLandmarker wrapper (up to 2 hands) |
| `face.py` | MediaPipe FaceLandmarker wrapper (up to 4 faces) |
| `multi.py` | per-hand tracker/gesture slots + per-face slots, position-matched |
| `blur.py` | PINCH toggle state machine, face-box memory, Gaussian/pixelate blur |
| `privacy.py` | OFF/FACE_BLUR/STRICT/AUTO controller + auto rules |
| `analytics.py` | session stats, tracking IDs, events, JSON/CSV export |
| `actions.py` | allowlisted media/volume/Hyprland executors (busctl/wpctl/hyprctl) |
| `control.py` | gesture→action engine: edges, cooldowns, master toggle, mapping |
| `tracker.py` | fingertip positions, smoothing, movement |
| `face_tracker.py` | face center/box, zone, proximity, head orientation |
| `gesture.py` | deterministic gesture classification |
| `positions.py` | position classification + OpenCV drawing helpers |
| `capture.py` | camera abstraction |
| `cli_static.py` | single-image CLI |

## Notes

- Performance is CPU-bound by MediaPipe inference (~20 FPS at 640×480 on a typical CPU).
- Handedness follows MediaPipe convention (mirrored for selfie cameras).
