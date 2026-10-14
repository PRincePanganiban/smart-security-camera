"""
Smart Security Camera
CSC 126 - Computer Vision Midterm Project

Detects motion in a video, checks whether the moving object contains a face,
and raises an INTRUDER alert (snapshot + log entry) when it does.

Techniques used (lab topics):
  A. Image Operations ......... src/preprocessing.py
  B. Image Segmentation ....... src/motion.py  (background subtraction + threshold)
  C. Haar Cascade Classifier .. src/face_detector.py
  E. Motion & Object Tracking . src/motion.py + src/tracker.py

Usage examples:
  python main.py --source sample_inputs/demo_intruder.mp4
  python main.py --source my_video.mp4 --save outputs/result.mp4
  python main.py --source webcam            # optional, if a webcam exists
"""

import argparse
import os
import sys
import time

import cv2

from src.alerts import AlertManager
from src.face_detector import FaceDetector
from src.motion import MotionDetector
from src.preprocessing import prepare_for_faces, prepare_for_motion, resize_keep_aspect
from src.tracker import CentroidTracker
from src.visualization import (draw_faces, draw_hud, draw_moving_objects,
                               draw_status_banner, side_by_side_with_mask)


def parse_args():
    p = argparse.ArgumentParser(description="Smart Security Camera (OpenCV)")
    p.add_argument("--source", default="sample_inputs/demo_intruder.mp4",
                   help="video file path, 'webcam', or a camera index (default: sample video)")
    p.add_argument("--width", type=int, default=640, help="processing width in pixels")
    p.add_argument("--min-area", type=int, default=1500,
                   help="smallest moving blob (pixels) to count as motion")
    p.add_argument("--cooldown", type=float, default=3.0,
                   help="seconds between saved snapshots")
    p.add_argument("--save", default=None, help="optional path to save the processed video (.mp4)")
    p.add_argument("--no-mask", action="store_true", help="hide the motion-mask panel")
    p.add_argument("--no-display", action="store_true",
                   help="run without windows (for testing / servers)")
    return p.parse_args()


def open_source(source):
    """Open a video file, a webcam, or a camera index."""
    if source.lower() == "webcam":
        return cv2.VideoCapture(0)
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    if not os.path.isfile(source):
        sys.exit(f"[ERROR] Video file not found: {source}")
    return cv2.VideoCapture(source)


def main():
    args = parse_args()

    cap = open_source(args.source)
    if not cap.isOpened():
        sys.exit("[ERROR] Could not open the video source.")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps < 1:  # missing or NaN
        src_fps = 25.0

    # Create the processing components
    motion = MotionDetector(min_area=args.min_area)
    tracker = CentroidTracker()
    faces_detector = FaceDetector()
    alerts = AlertManager(fps=src_fps, cooldown_sec=args.cooldown)

    writer = None
    armed = True
    paused = False
    frame_idx = 0
    fps = 0.0
    output = None
    shots_taken = 0
    os.makedirs("screenshots", exist_ok=True)

    print(f"[INFO] Source: {args.source}  |  {src_fps:.1f} FPS")
    print("[INFO] Keys: q = quit, a = arm/disarm, space = pause, p = screenshot")

    while True:
        loop_start = time.time()

        if not paused:
            ok, frame = cap.read()
            if not ok:
                break  # end of video
            frame_idx += 1

            # ---- Topic A: image operations ----
            frame = resize_keep_aspect(frame, args.width)
            blurred = prepare_for_motion(frame)
            equalized = prepare_for_faces(frame)

            # ---- Topic B + E: segmentation and motion detection ----
            mask, boxes = motion.detect(blurred)

            # ---- Topic E: object tracking ----
            tracked = tracker.update(boxes)

            # ---- Topic C: Haar cascade, only inside moving regions ----
            faces = faces_detector.detect_in_regions(equalized, boxes) if boxes else []

            # ---- Decide the system status ----
            if not armed:
                status = "disarmed"
            elif motion.learning:
                status = "learning"
            elif faces:
                status = "intruder"
            elif boxes:
                status = "motion"
            else:
                status = "clear"

            if status == "intruder":
                alerts.trigger(frame, frame_idx, len(boxes), len(faces))

            # ---- Draw the results ----
            output = frame.copy()
            draw_moving_objects(output, tracked, tracker.trails)
            draw_faces(output, faces)
            draw_status_banner(output, status)
            draw_hud(output, fps, frame_idx, alerts.alert_count, armed)
            if not args.no_mask:
                output = side_by_side_with_mask(output, mask)

            # Optional: write the processed video to disk
            if args.save:
                if writer is None:
                    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                    h, w = output.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps, (w, h))
                writer.write(output)

        # ---- Show the window and handle keys ----
        if not args.no_display and output is not None:
            cv2.imshow("Smart Security Camera", output)
            elapsed_ms = int((time.time() - loop_start) * 1000)
            delay = max(1, int(1000 / src_fps) - elapsed_ms)  # play at natural speed
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                armed = not armed
                print(f"[INFO] System {'ARMED' if armed else 'DISARMED'}")
            elif key == ord(" "):
                paused = not paused
            elif key == ord("p"):
                shots_taken += 1
                path = f"screenshots/screenshot_{shots_taken:02d}.png"
                cv2.imwrite(path, output)
                print(f"[INFO] Screenshot saved: {path}")

        total = time.time() - loop_start
        if total > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / total) if fps else 1.0 / total

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("\n===== SUMMARY =====")
    print(f"Frames processed : {frame_idx}")
    print(f"Intruder alerts  : {alerts.alert_count}")
    print(f"Snapshots folder : {alerts.snapshot_dir}")
    print(f"Event log        : {alerts.log_path}")


if __name__ == "__main__":
    main()
