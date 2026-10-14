# Smart Security Camera (Motion + Face Detection)

**CSC 126 – Computer Vision Midterm Project** · Caraga State University, CCIS

A classical OpenCV application that watches a video, detects anything that moves,
checks whether the moving object contains a **face**, and raises an
**INTRUDER DETECTED** alert (on-screen banner + saved snapshot + CSV log entry).

> Moving object without a face → `MOTION DETECTED` (yellow)  
> Moving object with a face → `INTRUDER DETECTED` (red, evidence saved)

## Real-world problem
Basic security cameras either record everything (wasting storage) or alert on any
movement (curtains, pets, shadows). This project filters motion by checking for a
person's face, then stores only the useful evidence.

## Techniques used (lab topics)
| Topic | Where | What it does |
|---|---|---|
| **A. Image Operations** | `src/preprocessing.py` | resize, grayscale, Gaussian blur, histogram equalization |
| **B. Image Segmentation** | `src/motion.py` | MOG2 background subtraction, thresholding, morphology → motion mask |
| **C. Haar Cascade Classifier** | `src/face_detector.py` | frontal-face detection, searched only inside moving regions |
| **E. Motion & Object Tracking** | `src/motion.py`, `src/tracker.py` | contours → bounding boxes, centroid tracker with IDs and movement trails |

## Project structure
```
smart-security-camera/
├── main.py                  # entry point: runs the whole pipeline
├── requirements.txt
├── src/
│   ├── preprocessing.py     # Topic A
│   ├── motion.py            # Topic B + E
│   ├── tracker.py           # Topic E
│   ├── face_detector.py     # Topic C
│   ├── alerts.py            # snapshots + CSV event log
│   └── visualization.py     # boxes, labels, banner, mask panel
├── tools/make_sample_video.py   # builds the sample test video
├── sample_inputs/           # demo_intruder.mp4 + face_sample.jpg
├── screenshots/             # output screenshots
├── outputs/                 # snapshots/ and logs/events.csv are written here
└── docs/DEMO_SCRIPT.md      # demo plan, member roles, Q&A prep
```

## Setup (Windows + VS Code)
1. **Python 3.11 – 3.14** (tested with OpenCV 4.13). Check with `python --version`;
   if missing, install from https://www.python.org/downloads/  
   ⚠ On the first installer screen, tick **"Add python.exe to PATH"**.
2. In VS Code install the **Python** extension (Microsoft).
3. Open this folder in VS Code → **Terminal → New Terminal**, then run:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   If PowerShell blocks activation, run once:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
4. (Optional) In VS Code press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`.

## How to run
```
python main.py                                   # uses the included sample video
python main.py --source path\to\your_video.mp4   # your own video
python main.py --source your_video.mp4 --save outputs/result.mp4
python main.py --source webcam                   # optional, only if a webcam exists
```
Keys in the window: `q` quit · `a` arm/disarm · `space` pause · `p` save screenshot to `screenshots/`

Useful options: `--min-area` (motion sensitivity), `--cooldown` (seconds between snapshots),
`--no-mask` (hide the mask panel), `--width` (processing size).

## Recording your own test video
Use a phone: place it on a steady surface (**do not move the camera**), record 10–15 s of
the empty scene first (lets the background model learn), then walk into view facing the
camera. Good, even lighting works best. Copy the video into `sample_inputs/`.

## Output
* Live window: processed frame with boxes/labels/banner, beside the motion mask
* `outputs/snapshots/intruder_*.jpg` — evidence images
* `outputs/logs/events.csv` — timestamp, frame, video time, event, counts

## Limitations
* Needs a **static camera**; camera shake is treated as motion.
* Haar cascades only detect **frontal, reasonably lit faces** (no profile views, masks, or tiny faces).
* A person who stops moving is slowly absorbed into the background and is no longer detected.
* Sudden lighting changes can trigger false motion.

## Possible improvements
Profile-face / body cascades, facial landmarks (eye-blink / drowsiness), a DNN face detector,
e-mail/SMS alerts, region-of-interest zones.
