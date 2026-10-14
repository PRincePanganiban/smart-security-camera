"""
visualization.py
----------------
Drawing helpers: bounding boxes, labels, trails, status banner, HUD, mask view.
"""

import cv2
import numpy as np

GREEN = (0, 200, 0)
YELLOW = (0, 215, 255)
RED = (0, 0, 255)
BLUE = (255, 160, 0)
GRAY = (160, 160, 160)
WHITE = (255, 255, 255)


def draw_moving_objects(frame, tracked, trails):
    """Yellow boxes + ID labels for moving objects, with a blue movement trail."""
    for object_id, (x, y, w, h) in tracked:
        cv2.rectangle(frame, (x, y), (x + w, y + h), YELLOW, 2)
        cv2.putText(frame, f"Object {object_id}", (x, max(15, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
        points = list(trails.get(object_id, []))
        for p1, p2 in zip(points, points[1:]):
            cv2.line(frame, p1, p2, BLUE, 2)


def draw_faces(frame, faces):
    """Green boxes + label for each detected face."""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(frame, "Face", (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)


def draw_status_banner(frame, status):
    """Colored banner on top: CLEAR / MOTION / INTRUDER / DISARMED / LEARNING."""
    text, color = {
        "clear": ("AREA CLEAR", GREEN),
        "motion": ("MOTION DETECTED", YELLOW),
        "intruder": ("!! INTRUDER DETECTED !!", RED),
        "disarmed": ("SYSTEM DISARMED", GRAY),
        "learning": ("LEARNING BACKGROUND...", GRAY),
        "adapting": ("LIGHTING CHANGE - ADAPTING...", GRAY),
    }[status]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 36), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    text_color = (0, 0, 0) if status in ("motion", "clear") else WHITE
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                text_color, 2, cv2.LINE_AA)


def draw_hud(frame, fps, frame_idx, alert_count, armed, show_keys=True):
    """Small info line at the bottom of the frame."""
    info = (f"FPS {fps:4.1f} | Frame {frame_idx} | Alerts {alert_count} | "
            f"{'ARMED' if armed else 'DISARMED'}")
    if show_keys:
        info += "  [q] quit [a] arm [space] pause [p] screenshot"
    cv2.rectangle(frame, (0, frame.shape[0] - 24), (frame.shape[1], frame.shape[0]),
                  (0, 0, 0), -1)
    cv2.putText(frame, info, (6, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, WHITE, 1, cv2.LINE_AA)


def side_by_side_with_mask(frame, mask):
    """Show the processed frame next to the motion mask (segmentation result)."""
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_bgr, "Motion mask (segmentation)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    return np.hstack([frame, mask_bgr])
