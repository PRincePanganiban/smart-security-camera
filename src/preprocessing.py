"""
preprocessing.py
----------------
TOPIC A - IMAGE OPERATIONS

Basic image operations applied to every frame before detection:
  * resizing       -> faster, consistent processing size
  * grayscale      -> motion and faces do not need color
  * Gaussian blur  -> removes sensor noise so it is not mistaken for motion
  * histogram equalization -> makes face detection more robust to lighting
"""

import cv2


def resize_keep_aspect(frame, width=640):
    """Resize a frame to the given width while keeping the aspect ratio."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def prepare_for_motion(frame, ksize=(5, 5)):
    """BGR frame -> blurred grayscale image used by the motion detector."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, ksize, 0)


def prepare_for_faces(frame):
    """BGR frame -> contrast-equalized grayscale image used by the Haar cascade."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)
