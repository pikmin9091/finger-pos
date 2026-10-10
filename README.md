# Finger POS

Real-time hand + face tracking with MediaPipe — fingertip positions, movement tracking, gesture recognition, and face position. Outputs normalized coordinates as JSON for integration into POS / touchless interfaces.

## Features

- **Multi-hand tracking** — up to 2 hands, each with its own smoothing + gesture debounce slot (matched by wrist-anchor proximity across frames)
- **Multi-face tracking** — up to 4 faces, ordered left-to-right
- **Gesture-controlled face blur** — stable PINCH toggles Gaussian blur on all visible faces (edge-triggered + cooldown, state persists with no hands); last-known boxes cover brief detection drops
- **Smart Privacy Mode** — `OFF / FACE_BLUR / STRICT / AUTO` with configurable rules, Gaussian or pixelate obfuscation, PINCH or `M`/`P` keys to switch
- **Real-time analytics** — live counts, stable tracking IDs, enter/exit events with durations (exits need 5 consecutive missed frames, so brief detection gaps don't spam pairs), gesture/looking distributions, inference latency, session export to JSON/CSV
- **Text-only overlay** — one thin box per detected object plus plain status lines (gesture, face, blur, FPS), no distracting graphics
- **Gesture Control Center** (`--control`) — hand gestures drive media (MPRIS), volume (`wpctl`), and Hyprland workspaces (`hyprctl`) via an allowlist; per-action cooldowns, master toggle, text status lines + notifications

- **Fingertip tracking** — normalized `x, y, z` (0.0–1.0) for all 5 fingertips
- **Position classification** — `LEFT/CENTER/RIGHT` and `TOP/CENTER/BOTTOM` per finger
- **Movement tracking** — direction (`UP/DOWN/LEFT/RIGHT/STATIONARY`) + velocity (units/sec)
- **Gesture recognition** — `POINT`, `PEACE`, `THUMBS_UP`, `THUMBS_DOWN`, `OPEN_PALM`, `FIST`, `PINCH` with confidence and stability flag
- **Thumb-vs-fist disambiguation** — thumb extension reads tip-to-palm reach (not bone length), direction reads against the palm up-axis; sideways/weak thumbs rejected as UNKNOWN
- **UNKNOWN rejection** — weak or unmatched patterns return `NONE` with a reason (`weak-pattern`/`no-match`/`ambiguous-thumb`) instead of forcing a gesture; truncated landmarks are rejected, never classified
- **Calibration** — `main.py --calibrate` derives your pinch/thumb thresholds from held poses (JSON only, no images saved); auto-loaded when present
- **Debug overlay** — `main.py --debug` / `cli_static.py --debug` show raw vs final gesture, per-finger bits+scores, thumb reach/alignment, hand height, volume state/target/bounds, pinch ratio, cooldown, tracking state — all real pipeline values
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
| `--pinch-threshold` | `0.06` | hand-size-normalized thumb–index gap for PINCH |
| `--no-face` | off | disable face detection (hand tracking only) |
| `--sensitivity` | `normal` | detection sensitivity: `low`=0.65 (fewer false positives), `normal`=0.5, `high`=0.35 (finds distant/small hands, more flicker) |
| `--min-velocity` | `0.05` | units/s below which motion reads STATIONARY (raise if slow drift misfires moves) |
| `--no-blur` | off | disable gesture-controlled face blur |
| `--blur-cooldown` | `1.0` | min seconds between blur on/off toggles |
| `--privacy` | `OFF` | initial privacy mode: `OFF/FACE_BLUR/STRICT/AUTO` |
| `--blur-kind` | `GAUSSIAN` | obfuscation style: `GAUSSIAN/PIXELATE` |
| `--blur-strength` | `5` | obfuscation strength 1–10 |
| `--auto-faces` | `1` | AUTO privacy engages at this face count |
| `--control` | off | enable gesture control (media/volume/Hyprland) |
| `--control-profile` | `default` | `default` or `calm` (confident held gestures only, no motion) |
| `--no-motion` | off | disable motion/swipe triggers in gesture control |
| `--vol-top` | `0.25` | hand height (0=top) mapping to volume 100% |
| `--vol-bottom` | `0.75` | hand height mapping to volume 0% |

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
| Open palm ~1 s ✋ | master OFF→ON; when ON: enter/exit volume mode |
| Thumbs up 👍 | play/pause |
| Peace ✌️ | next track |
| Fist ✊ | previous track |
| Index up ☝️ | mute toggle |
| Thumbs down 👎 | master toggle off/on |
| Pinch (master OFF / no `--control`) | privacy-blur toggle (unchanged) |

PINCH is never used for volume: with master ON it does nothing (privacy
owns it only while master is off). Volume lives entirely in volume mode.

**Volume mode (hand height):** hold OPEN_PALM → `VOLUME MODE: ACTIVE`
appears with 100/0 calibration markers at the frame edge → move the hand
up (louder) / down (softer); position maps to 0–100% between `--vol-top`
(0.25) and `--vol-bottom` (0.75). Smoothing + 0.008 deadband + ≥2% / 0.4 s
throttle keep `wpctl` quiet; still hands send nothing further. Hold
OPEN_PALM again to lock & exit. Losing tracking pauses updates
(`TRACKING_LOST`, never treated as 0); still hands and inactive mode
never change volume. Calibrate the range: note your comfortable top/bottom
`y` from `--debug` (`HAND:y=…`), then pass `--vol-top/--vol-bottom`.

**Anti-misfire rules (defaults):** confidence ≥0.8, hold-to-fire 0.7 s
(`--control-profile calm`: 1.0 s + no motion), per-action cooldowns
1–1.5 s, neutral-reset (a trigger must disappear before re-arming),
debounce + hysteresis upstream, master gate. Tune via `--debounce`,
`--control-profile calm`, `--no-motion`.

Each gesture or continuous movement fires at most one command, then waits
for a neutral pose or a 0.25-second stationary pause before re-arming;
motion-based volume actions no longer repeat while held. Low-confidence or
unstable gestures never fire. No player/tool → clear `(failed)` notice,
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


### Static image

```bash
python3 cli_static.py path/to/image.jpg
python3 cli_static.py path/to/image.jpg --blur --out preview.jpg  # save blurred preview
```

### Semua Kontrol

Bagian ini merangkum **seluruh cara mengendalikan aplikasi** di satu tempat:
keyboard, tombol klik, slider, gesture tangan, dan flag CLI terkait.

#### 1. Keyboard (jendela preview)

| Tombol | Aksi | Syarat |
|--------|------|--------|
| `M` | Ganti mode privasi OFF→FACE_BLUR→STRICT→AUTO | — |
| `P` | Toggle blur ON↔OFF (setara gesture PINCH) | — |
| `B` | Ganti jenis blur GAUSSIAN↔PIXELATE | — |
| `C` | Master gesture-control ON/OFF | hanya dengan `--control` |
| `H` | Tampilkan/sembunyikan panel bantuan | — |
| `S` | Ekspor sesi ke `analytics_session_<ts>.json` + `.csv` | — |
| `R` | Reset statistik sesi analytics | — |
| `Q` | Keluar aplikasi | — |

#### 2. Tombol klik mouse (bottom bar, perlu build OpenCV GUI)

| Tombol | Setara keyboard | Keterangan |
|--------|-----------------|------------|
| PRIV | `M` | Label menunjukkan mode aktif (`PRIV:AUTO`, …) |
| BLUR | `B` | Label menunjukkan jenis aktif (`BLUR:PIXELATE`, …) |
| CTRL | `C` | Hijau saat master ON; nonaktif tanpa `--control` |
| SNAP | `S` | Simpan JSON + CSV sesi |
| RESET | `R` | Nol-kan penghitung sesi |
| HELP | `H` | Panel contekan |
| QUIT | `Q` | Keluar |

Mode/blur yang aktif disorot hijau. Klik di luar tombol diabaikan.

#### 3. Slider live-tuning (perlu build OpenCV GUI)

| Slider | Rentang | Efek langsung |
|--------|---------|---------------|
| `smooth` | 0–100 | EMA tracking semua tangan aktif + default slot baru |
| `debounce` | 1–10 | Frame tahan gesture sebelum berganti |
| `strength` | 1–10 | Kekuatan blur wajah |

Berlaku seketika tanpa restart. Di build headless slider tidak ada
(aplikasi tetap jalan JSON-only) — gunakan flag CLI sebagai gantinya.

#### 4. Gesture tangan (perlu `--control` untuk aksi sistem)

Setiap gesture harus **ditahan** (default 0,7 dtk; profil `calm` 1,0 dtk),
melewati cooldown per aksi, dan kembali netral sebelum bisa menembak lagi.
Satu gesture = satu aksi; menahan tidak mengulang.

| Gesture (tahan) | Aksi | Catatan |
|---|---|---|
| Telapak terbuka ✋ | Master OFF→ON; saat ON: masuk/keluar mode volume | Satu-satunya gesture multifungsi |
| Jempol ke atas 👍 | Play/pause musik (MPRIS) | — |
| Peace ✌️ | Lagu berikutnya | — |
| Kepalan ✊ | Lagu sebelumnya | — |
| Telunjuk ☝️ | Mute/unmute | — |
| Jempol ke bawah 👎 | Master toggle on/off | Metafora "tidak" |
| Cubit/PINCH 🤏 | Toggle blur (master OFF) / tidak dipakai (master ON) | Volume **tidak pernah** pakai PINCH |

**Mode volume (tinggi tangan):** tahan telapak → baris `VOLUME MODE: ACTIVE`
+ garis kalibrasi 100/0 muncul → gerakan tangan naik/turun mengatur volume
0–100% (rentang `--vol-top`/`--vol-bottom`, deadband, throttle ≥2%/0,4 dtk).
Tahan telapak lagi = kunci & keluar. Tracking hilang = pause (bukan nol);
tangan diam = tidak ada perintah baru.

#### 5. Flag CLI terkait kontrol

| Flag | Default | Efek |
|------|---------|------|
| `--control` | off | Aktifkan subsistem gesture-control |
| `--control-profile` | `default` | `calm` = confidence ≥0,8 + tahan 1 dtk + tanpa motion |
| `--no-motion` | off | Matikan pemicu gerakan/swipe |
| `--vol-top` / `--vol-bottom` | `0.25` / `0.75` | Batas kalibrasi tinggi tangan → 100%/0% |
| `--min-velocity` | `0.05` | Di bawah ini gerakan = STATIONARY |
| `--sensitivity` | `normal` | `low` 0,65 / `normal` 0,5 / `high` 0,35 |
| `--smoothing` | `0.2` | Sama dengan slider `smooth` (nilai awal) |
| `--debounce` | `3` | Sama dengan slider `debounce` (nilai awal) |
| `--blur-cooldown` | `1.0` | Jeda minimum antar toggle blur |
| `--privacy` | `OFF` | Mode privasi awal |
| `--auto-faces` | `1` | AUTO aktif pada jumlah wajah ini |

Contoh paling kalem (minim salah tembak):
```bash
./myenv/bin/python main.py --camera 10 --control --control-profile calm --sensitivity normal
```

### Gesture accuracy: what was audited

Landmark topology matches the MediaPipe 21-point spec; BGR→RGB conversion,
VIDEO tracking mode, and the rotation/mirror-invariant classifier were all
verified. Known limits, found by code audit + synthetic probing:

- Curled fingers carry almost no joint-angle signal, so extension uses a
  curl guard (tip parked at PIP ⇒ folded) instead of trusting noisy angles.
- Thumb extension reads tip-to-palm reach (0.55/0.40), NOT tip-to-own-MCP:
  the old bone-length feature with its 0.90 threshold could never fire on
  a real thumb, which is why THUMBS_UP always read FIST. Direction reads
  against the wrist→middle-MCP palm axis (a curled middle finger points
  the wrong way and inverted the old reading).
- `confidence` is a heuristic pattern-strength blend, **not** a calibrated
  probability — treat it as match quality.
- Thresholds are tuned on anatomically proportioned synthetic poses; real
  hands vary. Run `--calibrate` (tunes pinch + thumb reach) and re-check
  the gestures you actually use.

### Calibration

```bash
python3 main.py --camera 10 --calibrate
```

Hold OPEN_PALM, FIST, then PINCH (~4 s each). The app collects landmark
geometry only, derives `pinch_threshold`/`thumb_extend`/`thumb_fold`, and
saves `~/.config/finger-pos/calib.json` (override with nothing to share —
no images). Rejected sessions print reasons and change nothing. Live and
static modes auto-load the file when present.

### Measuring accuracy (honestly)

```bash
./myenv/bin/python eval_gestures.py
```

Runs the real detector over a deterministic synthetic grid (5 rotations ×
3 scales + noise per gesture, plus ambiguous poses) and prints accuracy,
per-class P/R/F1, confusion matrix, UNKNOWN rate, and confirm latency.
Current synthetic result: **90/90 correct, 10/10 ambiguous rejected,
~2 frames to confirm**. This measures geometric robustness, NOT camera
accuracy — do not quote it as product accuracy.

Manual protocol (required before trusting a new gesture mapping): in varied
lighting, distances (0.4–1.2 m), both hands, and ±45° roll, hold each
gesture 3 s and log raw/final from `--debug`; count correct frames. A
mapping is accepted at ≥90% held-frame agreement with no wrong-gesture
fires (UNKNOWN is always acceptable).

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
- MediaPipe handedness flickers frame-to-frame, so each slot majority-votes
  recent labels (window of 8) instead of trusting the raw per-frame label;
  analytics ids therefore stay put while the reported hand may lag a flip
  by a few frames.
- PINCH accepts truly touching tips even when the thumb reads folded
  (foreshortening collapses its 2D reach); separated curled tips still
  need the thumb-out check, so plain fists stay FIST.

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

194 unit tests (synthetic landmarks + mocked action runner, no webcam/hardware needed).

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
| `calib.py` | per-user threshold calibration (JSON, landmarks only) |
| `eval_gestures.py` | synthetic robustness eval: accuracy, confusion, latency |
| `tracker.py` | fingertip positions, smoothing, movement |
| `face_tracker.py` | face center/box, zone, proximity, head orientation |
| `gesture.py` | deterministic gesture classification |
| `positions.py` | position classification + OpenCV drawing helpers |
| `capture.py` | camera abstraction |
| `cli_static.py` | single-image CLI |

## Notes

- Performance is CPU-bound by MediaPipe inference (~20 FPS at 640×480 on a typical CPU).
- Handedness follows MediaPipe convention (mirrored for selfie cameras).
