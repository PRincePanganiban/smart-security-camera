"""
face_detector.py
----------------
TOPIC C - HAAR CASCADE CLASSIFIER

Uses OpenCV's pre-trained frontal-face Haar cascade.
To save time and avoid false alarms, faces are searched ONLY inside the
regions where motion was found (plus a small padding), not the whole frame.
A moving object that contains a face is treated as a person.
"""

import os

import cv2

from src.paths import resource_path


class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=3, min_size=(30, 30)):
        """
        scale_factor  : how much the search window grows each pass (smaller = slower, finer)
        min_neighbors : overlapping detections needed to accept a face (higher = fewer false positives)
        min_size      : smallest face (w, h) to look for
        """
        # Use the copy shipped in assets/ (works inside the .exe); otherwise
        # fall back to the file that comes with the opencv-python package.
        cascade_path = resource_path(os.path.join("assets", "haarcascade_frontalface_default.xml"))
        if not os.path.isfile(cascade_path):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect_in_regions(self, gray, regions, padding=60):
        """
        Search for faces inside each (x, y, w, h) region of the gray image.
        Returns a list of face boxes in full-frame coordinates.
        """
        img_h, img_w = gray.shape[:2]
        faces = []
        for (x, y, w, h) in regions:
            # Expand the region a little, but stay inside the image
            x0, y0 = max(0, x - padding), max(0, y - padding)
            x1, y1 = min(img_w, x + w + padding), min(img_h, y + h + padding)
            roi = gray[y0:y1, x0:x1]

            # Skip regions that are too small to contain a face
            if roi.shape[1] < self.min_size[0] or roi.shape[0] < self.min_size[1]:
                continue

            found = self.cascade.detectMultiScale(
                roi, scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors, minSize=self.min_size
            )
            for (fx, fy, fw, fh) in found:
                faces.append((fx + x0, fy + y0, fw, fh))
        return self._remove_duplicates(faces)

    @staticmethod
    def _remove_duplicates(faces):
        """Overlapping motion regions can find the same face twice; keep one."""
        unique = []
        for (x, y, w, h) in faces:
            cx, cy = x + w // 2, y + h // 2
            duplicate = any(ux <= cx <= ux + uw and uy <= cy <= uy + uh
                            for (ux, uy, uw, uh) in unique)
            if not duplicate:
                unique.append((x, y, w, h))
        return unique