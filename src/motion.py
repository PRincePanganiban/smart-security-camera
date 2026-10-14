"""
motion.py
---------
TOPIC B - IMAGE SEGMENTATION  (foreground / background separation)
TOPIC E - MOTION AND OBJECT TRACKING (finding moving objects)

Pipeline for every frame:
  1. Background subtraction (MOG2) learns the static scene and marks
     everything that differs from it as "foreground".
  2. Thresholding turns the foreground into a clean black/white mask
     (shadows, which MOG2 marks as gray, are removed).
  3. Morphology (open + dilate) removes specks and fills small holes.
  4. Contours are extracted from the mask; small ones are ignored.
  5. Each remaining contour becomes a bounding box = one moving object.
"""

import cv2


class MotionDetector:
    def __init__(self, min_area=1500, history=300, var_threshold=32,
                 warmup_frames=30):
        """
        min_area       : ignore moving blobs smaller than this many pixels
        history        : how many frames MOG2 uses to learn the background
        var_threshold  : higher = less sensitive to small changes
        warmup_frames  : frames used only to learn the background (no detections)
        """
        self.min_area = min_area
        self.warmup_frames = warmup_frames
        self.frame_count = 0
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, blurred_gray):
        """
        Returns (mask, boxes)
          mask  : binary image, white = moving pixels
          boxes : list of (x, y, w, h) for each moving object
        """
        self.frame_count += 1

        # 1. Background subtraction: 255 = foreground, 127 = shadow, 0 = background
        foreground = self.subtractor.apply(blurred_gray)

        # 2. Threshold at 200 so shadows (127) are dropped
        _, mask = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)

        # 3. Morphology: opening removes noise, dilation joins broken parts
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        # During warm-up the model is still learning, so report nothing
        if self.frame_count <= self.warmup_frames:
            return mask, []

        # 4-5. Contours -> bounding boxes of sufficiently large moving objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours
                 if cv2.contourArea(c) >= self.min_area]
        return mask, boxes

    @property
    def learning(self):
        """True while the detector is still in its warm-up period."""
        return self.frame_count <= self.warmup_frames
