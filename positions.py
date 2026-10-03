import cv2

TIP_IDS = [4, 8, 12, 16, 20]
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20)]

def get_finger_tips(landmarks):
    if not landmarks:
        return None
    return [{"finger": i, "x": landmarks[i]["x"], "y": landmarks[i]["y"], "conf": landmarks[i]["visibility"]} for i in TIP_IDS]

def draw_gesture(frame, gesture_data):
    """Draw gesture name and confidence on the frame."""
    name = gesture_data.get("gesture", "NONE")
    confidence = gesture_data.get("confidence", 0.0)
    stable = gesture_data.get("stable", False)
    text = f"GESTURE: {name} ({confidence:.0%}) {'[STABLE]' if stable else ''}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame

def draw_status(frame, text):
    """Draw a status message on the frame (e.g. NO HAND)."""
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

def draw_landmarks(frame, landmarks):
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