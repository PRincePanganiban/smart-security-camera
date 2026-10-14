"""
Smart Security Camera - COMMAND-LINE version
CSC 126 - Computer Vision Midterm Project

(The desktop application with buttons is app.py. This version is handy for
quick tests and for saving a processed video.)

Usage examples:
  python main.py --source sample_inputs/demo_intruder.mp4
  python main.py --source my_video.mp4 --save outputs/result.mp4
  python main.py --source webcam
"""

import argparse
import os
import sys
import time

import cv2

from src.pipeline import SecurityPipeline
from src.visualization import draw_hud, side_by_side_with_mask


def parse_args():
    p = argparse.ArgumentParser(description="Smart Security Camera (OpenCV)")
    p.add_argument("--source", default="sample_inputs/demo_intruder.mp4",
                   help="video file path, 'webcam', or a camera index")
    p.add_argument("--width", type=int, default=640, help="processing width in pixels")
    p.add_argument("--min-area", type=int, default=1500,
                   help="smallest moving blob (pixels) to count as motion")
    p.add_argument("--sensitivity", type=float, default=32,
                   help="background-subtraction threshold (higher = less sensitive)")
    p.add_argument("--cooldown", type=float, default=3.0,
                   help="seconds between saved snapshots")
    p.add_argument("--save", default=None, help="optional path to save the processed video (.mp4)")
    p.add_argument("--no-mask", action="store_true", help="hide the motion-mask panel")
    p.add_argument("--no-display", action="store_true",
                   help="run without windows (for testing)")
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

    pipeline = SecurityPipeline(fps=src_fps, width=args.width, min_area=args.min_area,
                                var_threshold=args.sensitivity, cooldown_sec=args.cooldown)
    writer, paused, fps, output, shots = None, False, 0.0, None, 0
    os.makedirs("screenshots", exist_ok=True)

    print(f"[INFO] Source: {args.source}  |  {src_fps:.1f} FPS")
    print("[INFO] Keys: q = quit, a = arm/disarm, space = pause, p = screenshot")

    while True:
        loop_start = time.time()

        if not paused:
            ok, frame = cap.read()
            if not ok:
                break  # end of video
            result = pipeline.process(frame)
            output = result.frame
            draw_hud(output, fps, pipeline.frame_idx, pipeline.alerts.alert_count,
                     pipeline.armed)
            if not args.no_mask:
                output = side_by_side_with_mask(output, result.mask)

            if args.save:
                if writer is None:
                    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                    h, w = output.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps, (w, h))
                writer.write(output)

        if not args.no_display and output is not None:
            cv2.imshow("Smart Security Camera", output)
            elapsed_ms = int((time.time() - loop_start) * 1000)
            key = cv2.waitKey(max(1, int(1000 / src_fps) - elapsed_ms)) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                pipeline.armed = not pipeline.armed
                print(f"[INFO] System {'ARMED' if pipeline.armed else 'DISARMED'}")
            elif key == ord(" "):
                paused = not paused
            elif key == ord("p"):
                shots += 1
                path = f"screenshots/screenshot_{shots:02d}.png"
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
    print(f"Frames processed : {pipeline.frame_idx}")
    print(f"Intruder alerts  : {pipeline.alerts.alert_count}")
    print(f"Snapshots folder : {pipeline.alerts.snapshot_dir}")
    print(f"Event log        : {pipeline.alerts.log_path}")


if __name__ == "__main__":
    main()
