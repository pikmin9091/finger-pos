import cv2

TIP_IDS = [4, 8, 12, 16, 20]
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20)]

def get_finger_tips(landmarks):
    if not landmarks or len(landmarks) <= max(TIP_IDS):
        return None
    return [{"finger": i, "x": landmarks[i]["x"], "y": landmarks[i]["y"], "conf": landmarks[i].get("visibility", 1.0)} for i in TIP_IDS]

def draw_gesture(frame, gesture_data, y=30):
    """Draw gesture name and confidence on the frame."""
    name = gesture_data.get("gesture", "NONE")
    confidence = gesture_data.get("confidence", 0.0)
    stable = gesture_data.get("stable", False)
    text = f"GESTURE: {name} ({confidence:.0%}) {'[STABLE]' if stable else ''}"
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame

def draw_status(frame, text):
    """Draw a status message on the frame (e.g. NO HAND)."""
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

def draw_landmarks(frame, landmarks):
    if not landmarks or len(landmarks) < max(TIP_IDS) + 1:
        return frame
    h, w = frame.shape[:2]
    for a, b in CONNECTIONS:
        x1, y1 = int(landmarks[a]["x"] * w), int(landmarks[a]["y"] * h)
        x2, y2 = int(landmarks[b]["x"] * w), int(landmarks[b]["y"] * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm["x"] * w), int(lm["y"] * h)
        color = (0, 0, 255) if i in TIP_IDS else (255, 255, 255)
        cv2.circle(frame, (cx, cy), 5, color, -1)
    return frame

def draw_face(frame, face_data, y=90):
    """Draw face bounding box, center dot, and position/looking label."""
    if not face_data:
        return frame
    h, w = frame.shape[:2]
    box = face_data["box"]
    x1, y1 = int(box["x_min"] * w), int(box["y_min"] * h)
    x2, y2 = int(box["x_max"] * w), int(box["y_max"] * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    center = face_data["center"]
    cv2.circle(frame, (int(center["x"] * w), int(center["y"] * h)), 5, (255, 0, 0), -1)
    text = (f"FACE: {face_data['horizontal']}/{face_data['vertical']} "
            f"LOOKING:{face_data['looking']} ({face_data['proximity']})")
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame

def draw_blur_status(frame, enabled, count, y=120):
    """Draw blur on/off status plus blurred-face count."""
    text = f"BLUR: {'ON' if enabled else 'OFF'} ({count} face(s))"
    color = (0, 255, 0) if enabled else (150, 150, 150)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def draw_face_text(frame, face_data, y=90):
    """Text-only face status line (no boxes/graphics)."""
    if not face_data:
        return frame
    text = (f"FACE: {face_data['horizontal']}/{face_data['vertical']} "
            f"LOOKING:{face_data['looking']} ({face_data['proximity']})")
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

class UiToasts:
    """Expiring on-screen text messages for the camera window.

    Every important runtime event (mode change, action, export, reset)
    goes here so the camera UI is self-sufficient without terminal logs.
    """

    def __init__(self, ttl=3.0, max_show=4):
        self.ttl = ttl
        self.max_show = max_show
        self._items = []  # (text, color, until)

    def push(self, text, color=(255, 255, 255), timestamp=None):
        import time as _t
        now = timestamp if timestamp is not None else _t.time()
        self._items.append((str(text)[:64], color, now + self.ttl))
        self._items = self._items[-8:]

    def visible(self, timestamp=None):
        import time as _t
        now = timestamp if timestamp is not None else _t.time()
        self._items = [it for it in self._items if it[2] >= now]
        return [(t, c) for (t, c, _) in self._items[-self.max_show:]]

    def draw(self, frame, timestamp=None, x=10, y0=220, dy=25):
        for i, (text, color) in enumerate(self.visible(timestamp)):
            cv2.putText(frame, text, (x, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

def landmarks_box(landmarks, pad=0.02):
    """Tight normalized box around landmarks, or None when invalid."""
    if not landmarks:
        return None
    xs = [lm["x"] for lm in landmarks]
    ys = [lm["y"] for lm in landmarks]
    return {"x_min": max(0.0, min(xs) - pad), "y_min": max(0.0, min(ys) - pad),
            "x_max": min(1.0, max(xs) + pad), "y_max": min(1.0, max(ys) + pad)}

def draw_box(frame, box, color=(0, 255, 0), label=None):
    """Plain thin box + small label. Coordinates follow the real frame size."""
    if not box:
        return frame
    h, w = frame.shape[:2]
    x1 = int(max(0.0, box["x_min"]) * w)
    y1 = int(max(0.0, box["y_min"]) * h)
    x2 = int(min(1.0, box["x_max"]) * w)
    y2 = int(min(1.0, box["y_max"]) * h)
    if x2 <= x1 or y2 <= y1:
        return frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    if label:
        cv2.putText(frame, str(label)[:32], (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame