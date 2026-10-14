"""
Generates a synthetic test video so the project can be demonstrated without a camera.

Timeline (20 FPS, 640x480):
  0-3 s   empty room (background learning)
  3-6 s   a plain box moves across      -> MOTION only (no face)
  6-11 s  a person's photo walks across -> INTRUDER (motion + face)
  11-13 s empty room again

Run:  python tools/make_sample_video.py
"""

import cv2
import numpy as np

W, H, FPS = 640, 480, 20
rng = np.random.default_rng(1)


def make_background():
    """A simple 'room': gradient wall, floor, and a door-like rectangle."""
    bg = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        v = 150 + int(40 * y / H)
        bg[y, :] = (v, v - 10, v - 25)
    cv2.rectangle(bg, (0, 360), (W, H), (90, 100, 110), -1)        # floor
    cv2.rectangle(bg, (480, 120), (590, 360), (70, 90, 120), -1)   # door
    cv2.rectangle(bg, (60, 110), (200, 220), (200, 220, 235), -1)  # window
    cv2.rectangle(bg, (60, 110), (200, 220), (80, 80, 80), 3)
    return bg


def paste(frame, sprite, x, y):
    """Paste a sprite onto the frame, clipping at the edges."""
    sh, sw = sprite.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + sw), min(H, y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    frame[y0:y1, x0:x1] = sprite[y0 - y:y1 - y, x0 - x:x1 - x]


def main():
    background = make_background()
    person = cv2.imread("sample_inputs/face_sample.jpg")
    person = cv2.resize(person, (260, 260))
    box = np.full((90, 110, 3), (40, 120, 200), np.uint8)
    cv2.rectangle(box, (0, 0), (109, 89), (20, 60, 110), 3)

    path = "sample_inputs/demo_intruder.mp4"
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    total = 13 * FPS

    for i in range(total):
        t = i / FPS
        frame = background.copy()
        if 3 <= t < 6:      # plain box crossing the room
            x = int(-110 + (t - 3) / 3 * (W + 110))
            paste(frame, box, x, 300)
        elif 6 <= t < 11:   # person crossing the room, slow enough to detect
            x = int(-260 + (t - 6) / 5 * (W + 260) * 0.9)
            paste(frame, person, x, 150)
        noise = rng.normal(0, 3, frame.shape)  # mild camera noise
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
