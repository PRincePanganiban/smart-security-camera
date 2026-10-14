"""
Smart Security Camera - DESKTOP APPLICATION
CSC 126 - Computer Vision Midterm Project

A window with buttons and sliders around the OpenCV pipeline:
  * Open a video file (or use a webcam if one is available)
  * Watch the processed video and the motion mask side by side
  * See the live status: AREA CLEAR / MOTION DETECTED / INTRUDER DETECTED
  * Evidence (snapshots + CSV log) is saved automatically

Techniques used (lab topics):
  A. Image Operations ......... src/preprocessing.py
  B. Image Segmentation ....... src/motion.py  (background subtraction + threshold)
  C. Haar Cascade Classifier .. src/face_detector.py
  E. Motion & Object Tracking . src/motion.py + src/tracker.py

Run from Python:   python app.py
Build the .exe:    build_exe.bat
"""

import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from src.paths import app_dir
from src.pipeline import SecurityPipeline

DISPLAY_WIDTH = 460          # width of each preview panel on screen
DEFAULT_SENSITIVITY = 32     # MOG2 threshold (higher = less sensitive)
DEFAULT_MIN_AREA = 1500      # smallest moving blob that counts as motion

STATUS_STYLE = {             # status -> (banner text, background, text color)
    "idle": ("Open a video to begin", "#555555", "white"),
    "learning": ("LEARNING BACKGROUND...", "#6b6b6b", "white"),
    "adapting": ("LIGHTING CHANGE - ADAPTING...", "#6b6b6b", "white"),
    "clear": ("AREA CLEAR", "#2e9e44", "white"),
    "motion": ("MOTION DETECTED", "#f0b400", "black"),
    "intruder": ("!! INTRUDER DETECTED !!", "#d32f2f", "white"),
    "disarmed": ("SYSTEM DISARMED", "#777777", "white"),
    "finished": ("VIDEO FINISHED", "#37474f", "white"),
}


def to_photo(bgr, width=DISPLAY_WIDTH):
    """Convert an OpenCV image to a Tk image (PPM format, no Pillow needed)."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    if w != width:
        bgr = cv2.resize(bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    header = f"P6 {w} {h} 255\n".encode("ascii")
    return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")


class SmartSecurityApp:
    def __init__(self, root):
        self.root = root
        root.title("Smart Security Camera  |  CSC 126 Computer Vision")
        root.geometry("1030x800")
        root.minsize(1030, 760)

        # Playback state
        self.cap = None
        self.pipeline = None
        self.source = None
        self.is_webcam = False
        self.src_fps = 25.0
        self.playing = False
        self.after_id = None
        self.fps = 0.0
        self.last_frame = None
        self.screenshot_count = 0

        self._build_ui()
        self._show_placeholder()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # Button row
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", **pad)
        ttk.Button(bar, text="Open Video...", command=self.open_video).pack(side="left", padx=2)
        ttk.Button(bar, text="Use Webcam", command=self.open_webcam).pack(side="left", padx=2)
        self.play_btn = ttk.Button(bar, text="Pause", command=self.toggle_play, state="disabled")
        self.play_btn.pack(side="left", padx=(14, 2))
        self.restart_btn = ttk.Button(bar, text="Restart", command=self.restart, state="disabled")
        self.restart_btn.pack(side="left", padx=2)
        self.arm_btn = ttk.Button(bar, text="Disarm", command=self.toggle_arm, state="disabled")
        self.arm_btn.pack(side="left", padx=2)
        self.shot_btn = ttk.Button(bar, text="Save Screenshot", command=self.save_screenshot,
                                   state="disabled")
        self.shot_btn.pack(side="left", padx=(14, 2))
        ttk.Button(bar, text="Open Output Folder", command=self.open_output_folder).pack(
            side="right", padx=2)

        # Status banner
        self.banner = tk.Label(self.root, text="", font=("Segoe UI", 18, "bold"), pady=8)
        self.banner.pack(fill="x", **pad)

        # Two preview panels: processed video and motion mask
        views = ttk.Frame(self.root)
        views.pack(**pad)
        left = ttk.LabelFrame(views, text="Processed video (boxes, labels, trails)")
        left.grid(row=0, column=0, padx=6)
        right = ttk.LabelFrame(views, text="Motion mask (segmentation result)")
        right.grid(row=0, column=1, padx=6)
        self.video_label = tk.Label(left)
        self.video_label.pack(padx=4, pady=4)
        self.mask_label = tk.Label(right)
        self.mask_label.pack(padx=4, pady=4)

        # Settings sliders
        settings = ttk.LabelFrame(self.root, text="Settings (adjust while the video plays)")
        settings.pack(fill="x", **pad)
        settings.columnconfigure(1, weight=1)

        self.sens_var = tk.IntVar(value=DEFAULT_SENSITIVITY)
        ttk.Label(settings, text="Motion threshold\n(higher = less sensitive)").grid(
            row=0, column=0, sticky="w", padx=8, pady=2)
        ttk.Scale(settings, from_=10, to=80, variable=self.sens_var,
                  command=self._on_settings_change).grid(row=0, column=1, sticky="ew", padx=8)
        self.sens_text = ttk.Label(settings, width=6)
        self.sens_text.grid(row=0, column=2, padx=8)

        self.area_var = tk.IntVar(value=DEFAULT_MIN_AREA)
        ttk.Label(settings, text="Minimum object size\n(pixels)").grid(
            row=1, column=0, sticky="w", padx=8, pady=2)
        ttk.Scale(settings, from_=300, to=10000, variable=self.area_var,
                  command=self._on_settings_change).grid(row=1, column=1, sticky="ew", padx=8)
        self.area_text = ttk.Label(settings, width=6)
        self.area_text.grid(row=1, column=2, padx=8)
        self._on_settings_change()

        # Alert log
        log_frame = ttk.LabelFrame(self.root, text="Alert log (double-click a row to open the snapshot)")
        log_frame.pack(fill="both", expand=True, **pad)
        cols = ("no", "time", "snapshot")
        self.log = ttk.Treeview(log_frame, columns=cols, show="headings", height=4)
        self.log.heading("no", text="#")
        self.log.heading("time", text="Video time")
        self.log.heading("snapshot", text="Snapshot file")
        self.log.column("no", width=50, anchor="center")
        self.log.column("time", width=110, anchor="center")
        self.log.column("snapshot", width=700)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        scroll.pack(side="right", fill="y", pady=4)
        self.log.bind("<Double-1>", self._open_snapshot)

        # Status bar
        self.info = ttk.Label(self.root, text="", anchor="w")
        self.info.pack(fill="x", padx=10, pady=(0, 8))
        self._set_banner("idle")

    def _show_placeholder(self):
        blank = np.full((int(DISPLAY_WIDTH * 9 / 16), DISPLAY_WIDTH, 3), 35, np.uint8)
        self._video_photo = to_photo(blank)
        self._mask_photo = to_photo(blank)
        self.video_label.configure(image=self._video_photo)
        self.mask_label.configure(image=self._mask_photo)

    def _set_banner(self, status):
        text, bg, fg = STATUS_STYLE[status]
        self.banner.configure(text=text, bg=bg, fg=fg)

    def _on_settings_change(self, _=None):
        sens, area = int(self.sens_var.get()), int(self.area_var.get())
        self.sens_text.configure(text=str(sens))
        self.area_text.configure(text=str(area))
        if self.pipeline:  # apply immediately to the running detector
            self.pipeline.motion.set_var_threshold(sens)
            self.pipeline.motion.min_area = area

    # --------------------------------------------------------- source control
    def open_video(self):
        start_dir = os.path.join(app_dir(), "sample_inputs")
        if not os.path.isdir(start_dir):
            start_dir = app_dir()
        path = filedialog.askopenfilename(
            title="Choose a video", initialdir=start_dir,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")])
        if path:
            self.load_source(path)

    def open_webcam(self):
        self.load_source(0)

    def load_source(self, source):
        """Open a video file path or a camera index and start playing."""
        self._stop()
        if isinstance(source, int) and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Cannot open source",
                                 "Could not open the camera." if isinstance(source, int)
                                 else f"Could not open this video:\n{source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.src_fps = fps if fps and fps == fps and fps >= 1 else 25.0
        self.cap, self.source = cap, source
        self.is_webcam = isinstance(source, int)

        self.pipeline = SecurityPipeline(
            fps=self.src_fps, min_area=int(self.area_var.get()),
            var_threshold=int(self.sens_var.get()))
        for row in self.log.get_children():
            self.log.delete(row)

        self.play_btn.configure(state="normal", text="Pause")
        self.restart_btn.configure(state="normal")
        self.arm_btn.configure(state="normal", text="Disarm")
        self.shot_btn.configure(state="normal")
        self.playing = True
        self.fps = 0.0
        self._tick()

    def restart(self):
        if self.source is not None:
            self.load_source(self.source)

    def toggle_play(self):
        if not self.cap:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._tick()

    def toggle_arm(self):
        if not self.pipeline:
            return
        self.pipeline.armed = not self.pipeline.armed
        self.arm_btn.configure(text="Disarm" if self.pipeline.armed else "Arm")

    def _stop(self):
        self.playing = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ------------------------------------------------------------ main loop
    def _tick(self):
        """Process one frame, update the window, schedule the next frame."""
        if not self.playing or self.cap is None:
            return
        start = time.time()

        ok, frame = self.cap.read()
        if not ok:
            self._finish()
            return

        result = self.pipeline.process(frame)
        self.last_frame = result.frame

        self._video_photo = to_photo(result.frame)
        self._mask_photo = to_photo(result.mask)
        self.video_label.configure(image=self._video_photo)
        self.mask_label.configure(image=self._mask_photo)
        self._set_banner(result.status)

        if result.alert_path:
            seconds = self.pipeline.frame_idx / self.src_fps
            self.log.insert("", "end", iid=result.alert_path,
                            values=(self.pipeline.alerts.alert_count, f"{seconds:0.2f} s",
                                    os.path.basename(result.alert_path)))
            self.log.yview_moveto(1)

        elapsed = time.time() - start
        if elapsed > 0:
            self.fps = 0.9 * self.fps + 0.1 / elapsed if self.fps else 1.0 / elapsed
        self.info.configure(
            text=f"Frame {self.pipeline.frame_idx}   |   {self.fps:4.1f} FPS   |   "
                 f"Moving objects: {result.n_objects}   |   Faces: {result.n_faces}   |   "
                 f"Alerts: {self.pipeline.alerts.alert_count}   |   "
                 f"{'ARMED' if self.pipeline.armed else 'DISARMED'}")

        # Play video files at their natural speed; webcams as fast as possible
        delay = 1 if self.is_webcam else max(1, int(1000 / self.src_fps - elapsed * 1000))
        self.after_id = self.root.after(delay, self._tick)

    def _finish(self):
        self.playing = False
        self.play_btn.configure(text="Play", state="disabled")
        self._set_banner("finished")
        alerts = self.pipeline.alerts.alert_count if self.pipeline else 0
        self.info.configure(
            text=f"Finished: {self.pipeline.frame_idx} frames processed, {alerts} alert(s) saved. "
                 f"Press Restart to run again.")

    # ------------------------------------------------------------- extras
    def save_screenshot(self):
        if self.last_frame is None:
            return
        folder = os.path.join(app_dir(), "screenshots")
        os.makedirs(folder, exist_ok=True)
        self.screenshot_count += 1
        stamp = time.strftime("%H%M%S")
        path = os.path.join(folder, f"app_screenshot_{stamp}_{self.screenshot_count:02d}.png")
        cv2.imwrite(path, self.last_frame)
        messagebox.showinfo("Screenshot saved", f"Saved to:\n{path}")

    def open_output_folder(self):
        folder = os.path.join(app_dir(), "outputs")
        os.makedirs(folder, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(folder)  # Windows
        else:
            messagebox.showinfo("Output folder", folder)

    def _open_snapshot(self, _event):
        selected = self.log.selection()
        if selected and hasattr(os, "startfile"):
            os.startfile(selected[0])  # the row id is the full snapshot path

    def on_close(self):
        self._stop()
        self.root.destroy()


def selftest():
    """Headless check used to verify a build: loads the cascade and runs the pipeline."""
    report = os.path.join(app_dir(), "selftest.txt")
    try:
        pipeline = SecurityPipeline(fps=20)
        sample = os.path.join(app_dir(), "sample_inputs", "demo_intruder.mp4")
        cap = cv2.VideoCapture(sample)
        frames = 0
        while frames < 200:
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((360, 640, 3), np.uint8)
            pipeline.process(frame)
            frames += 1
        cap.release()
        text = f"SELFTEST OK - {frames} frames, {pipeline.alerts.alert_count} alert(s)"
    except Exception as error:  # report any failure in the text file
        text = f"SELFTEST FAILED - {error!r}"
    with open(report, "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    root = tk.Tk()
    SmartSecurityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
