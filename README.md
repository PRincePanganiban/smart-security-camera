# Smart Security Camera (Motion + Face Detection)

**Download the app (Windows .exe):** [Google Drive](https://drive.google.com/file/d/15Nz4rlfiWQhm1IxfPafBQjyVSFGpET6J/view?usp=sharing)
Unzip the folder and double-click `SmartSecurityCamera.exe`. If Windows warns that the app is unrecognized, click **More info, then Run anyway**. Use **Open Video...** and choose a file from the `sample_inputs` folder.

**CSC 126 – Computer Vision Midterm Project** · Caraga State University, CCIS

A **desktop application** (Python + OpenCV, packaged as a Windows `.exe`) that watches a video,
detects anything that moves, checks whether the moving object contains a **face**, and raises an
**INTRUDER DETECTED** alert (on-screen banner + saved snapshot + CSV log entry).

> Moving object without a face → `MOTION DETECTED` (yellow)  
> Moving object with a face → `INTRUDER DETECTED` (red, evidence saved)

## Real-world problem
Basic security cameras either record everything (wasting storage) or alert on any movement
(curtains, pets, shadows). This project filters motion by checking for a person's face, then
stores only the useful evidence.

## Techniques used (lab topics)
| Topic | Where | What it does |
|---|---|---|
| **A. Image Operations** | `src/preprocessing.py` | resize, grayscale, Gaussian blur, histogram equalization |
| **B. Image Segmentation** | `src/motion.py` | MOG2 background subtraction, thresholding, morphology → motion mask |
| **C. Haar Cascade Classifier** | `src/face_detector.py` | frontal-face detection, searched only inside moving regions |
| **E. Motion & Object Tracking** | `src/motion.py`, `src/tracker.py` | contours → bounding boxes, centroid tracker with IDs and movement trails |

## The application
Open `SmartSecurityCamera.exe` (or run `python app.py`):

| Control | What it does |
|---|---|
| **Open Video...** | choose a video file to analyze |
| **Use Webcam** | optional, only if the computer has a camera |
| **Pause / Play, Restart** | control playback |
| **Disarm / Arm** | turn alerts off and on |
| **Save Screenshot** | saves the processed frame to `screenshots/` |
| **Open Output Folder** | opens `outputs/` (snapshots + event log) |
| **Motion threshold** slider | higher = less sensitive to small changes |
| **Minimum object size** slider | ignores moving blobs smaller than this many pixels |

The window shows the processed video (boxes, labels, trails), the motion mask (segmentation),
a live status banner, counters, and an alert log (double-click a row to open its snapshot).

## Project structure
```
smart-security-camera/
├── app.py                   # DESKTOP APPLICATION (window, buttons, sliders)
├── main.py                  # command-line version (optional, can save a processed video)
├── build_exe.bat            # builds SmartSecurityCamera.exe
├── requirements.txt
├── assets/                  # Haar cascade file (bundled into the .exe)
├── src/
│   ├── pipeline.py          # connects all techniques; used by app.py and main.py
│   ├── preprocessing.py     # Topic A
│   ├── motion.py            # Topic B + E
│   ├── tracker.py           # Topic E
│   ├── face_detector.py     # Topic C
│   ├── alerts.py            # snapshots + CSV event log
│   ├── visualization.py     # boxes, labels, banner
│   └── paths.py             # file locations (script and .exe)
├── tools/make_sample_video.py
├── sample_inputs/           # demo videos
├── screenshots/
├── outputs/                 # snapshots/ and logs/events.csv are written here
└── docs/DEMO_SCRIPT.md
```

## Setup to run from source (Windows + VS Code)
1. **Python 3.11 – 3.14** (tested with OpenCV 4.x). Check with `python --version`. If missing, install
   from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"**
   (keep **tcl/tk and IDLE** ticked, because the window needs it).
2. Open this folder in VS Code → **Terminal → New Terminal**, then:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   python app.py
   ```
   If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Build the .exe
```
build_exe.bat
```
This installs PyInstaller, builds `dist\SmartSecurityCamera.exe`, copies `sample_inputs` next to it,
and runs a self-test (it should print `SELFTEST OK`). Give people the whole `dist` folder.
Windows SmartScreen or antivirus may warn about an unsigned `.exe` the first time: choose
**More info → Run anyway**.

> The `.exe` is about 100 MB, which is above GitHub's 100 MB file limit, so do **not** commit it.
> Upload it as a **GitHub Release** asset (Releases → Draft a new release → attach the zipped `dist` folder)
> or share a Google Drive link.

## Command-line version (optional)
```
python main.py --source sample_inputs\demo_intruder.mp4
python main.py --source your_video.mp4 --save outputs\result.mp4
python main.py --source your_video.mp4 --sensitivity 45 --min-area 3000
```
Keys: `q` quit · `a` arm/disarm · `space` pause · `p` screenshot.

## Recording your own test video
Place the phone on a steady surface (**do not move it**), record ~10 s of the empty scene first
(lets the background model learn), then walk in facing the camera. Even lighting works best.
Put the video in `sample_inputs/`.

## Output
* Live window with boxes/labels/banner beside the motion mask
* `outputs/snapshots/intruder_*.jpg` – evidence images
* `outputs/logs/events.csv` – timestamp, frame, video time, event, counts

## Limitations
* Needs a **static camera**; camera shake is treated as motion.
* If most of the picture changes at once (e.g. a phone camera changes exposure), the app shows
  `LIGHTING CHANGE - ADAPTING` and ignores that frame. A person filling over half the frame can trigger this too.
* Haar cascades detect **frontal, reasonably lit faces** only (no profile views, masks, or tiny faces).
* A person who stops moving is slowly absorbed into the background and is no longer detected.

## Possible improvements
Profile-face / body cascades, facial landmarks (eye-blink / drowsiness), a DNN face detector,
e-mail/SMS alerts, region-of-interest zones.
