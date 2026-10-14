"""
alerts.py
---------
Saves evidence when an intruder (moving object + face) is detected:
  * a snapshot image of the frame
  * a row in a CSV event log
A cooldown prevents saving hundreds of near-identical snapshots.
"""

import csv
import os
from datetime import datetime

import cv2


class AlertManager:
    def __init__(self, snapshot_dir="outputs/snapshots",
                 log_path="outputs/logs/events.csv", fps=25.0, cooldown_sec=3.0):
        self.snapshot_dir = snapshot_dir
        self.log_path = log_path
        self.cooldown_frames = max(1, int(cooldown_sec * fps))
        self.fps = fps
        self.last_alert_frame = -10**9
        self.alert_count = 0

        os.makedirs(snapshot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Start a fresh log file with a header row
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "frame", "video_time_sec", "event", "moving_objects", "faces"])

    def trigger(self, frame, frame_idx, n_motion, n_faces):
        """Save a snapshot + log row if the cooldown has passed. Returns the snapshot path, or None."""
        if frame_idx - self.last_alert_frame < self.cooldown_frames:
            return None
        self.last_alert_frame = frame_idx
        self.alert_count += 1

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intruder_{stamp}_frame{frame_idx}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, filename)
        cv2.imwrite(snapshot_path, frame)

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"), frame_idx,
                round(frame_idx / self.fps, 2), "INTRUDER_DETECTED", n_motion, n_faces])
        return snapshot_path
