"""
pipeline.py
-----------
Connects all the techniques into one object that is used by BOTH the
desktop window (app.py) and the command-line version (main.py).

For every frame:
  A. Image operations  -> preprocessing.py
  B. Segmentation      -> motion.py (background subtraction -> mask)
  E. Tracking          -> motion.py boxes + tracker.py (IDs, trails)
  C. Haar cascade      -> face_detector.py (only inside moving regions)
  -> decide status, save evidence, draw the results
"""

import os
from dataclasses import dataclass, field

from src.alerts import AlertManager
from src.face_detector import FaceDetector
from src.motion import MotionDetector
from src.paths import app_dir
from src.preprocessing import prepare_for_faces, prepare_for_motion, resize_keep_aspect
from src.tracker import CentroidTracker
from src.visualization import draw_faces, draw_moving_objects, draw_status_banner


@dataclass
class FrameResult:
    frame: object                      # annotated BGR frame (boxes, labels, banner)
    mask: object                       # binary motion mask
    status: str                        # clear / motion / intruder / disarmed / learning / adapting
    n_objects: int = 0
    n_faces: int = 0
    alert_path: str = None             # snapshot path if an alert was saved this frame
    tracked: list = field(default_factory=list)


class SecurityPipeline:
    def __init__(self, fps=25.0, width=640, min_area=1500, var_threshold=32,
                 cooldown_sec=3.0, snapshot_dir=None, log_path=None):
        base = app_dir()
        snapshot_dir = snapshot_dir or os.path.join(base, "outputs", "snapshots")
        log_path = log_path or os.path.join(base, "outputs", "logs", "events.csv")

        self.width = width
        self.armed = True
        self.frame_idx = 0
        self.motion = MotionDetector(min_area=min_area, var_threshold=var_threshold)
        self.tracker = CentroidTracker()
        self.face_detector = FaceDetector()
        self.alerts = AlertManager(snapshot_dir, log_path, fps, cooldown_sec)

    def process(self, frame):
        """Run every technique on one BGR frame and return a FrameResult."""
        self.frame_idx += 1

        # ---- Topic A: image operations ----
        frame = resize_keep_aspect(frame, self.width)
        blurred = prepare_for_motion(frame)
        equalized = prepare_for_faces(frame)

        # ---- Topic B + E: segmentation and motion detection ----
        mask, boxes = self.motion.detect(blurred)

        # ---- Topic E: object tracking ----
        tracked = self.tracker.update(boxes)

        # ---- Topic C: Haar cascade inside the moving regions only ----
        faces = self.face_detector.detect_in_regions(equalized, boxes) if boxes else []

        # ---- Decide the system status ----
        if not self.armed:
            status = "disarmed"
        elif self.motion.learning:
            status = "learning"
        elif self.motion.global_change:
            status = "adapting"
        elif faces:
            status = "intruder"
        elif boxes:
            status = "motion"
        else:
            status = "clear"

        alert_path = None
        if status == "intruder":
            alert_path = self.alerts.trigger(frame, self.frame_idx, len(boxes), len(faces))

        # ---- Draw the results ----
        output = frame.copy()
        draw_moving_objects(output, tracked, self.tracker.trails)
        draw_faces(output, faces)
        draw_status_banner(output, status)

        return FrameResult(output, mask, status, len(boxes), len(faces), alert_path, tracked)
