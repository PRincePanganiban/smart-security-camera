# Demo Script & Member Roles (15 min demo + 5 min Q&A)

## Suggested split for 3 members
| Member | Owns | Explains in the demo |
|---|---|---|
| **Member 1** | `preprocessing.py`, `motion.py` | Image operations (grayscale, blur, equalize) and segmentation (background subtraction → mask) |
| **Member 2** | `face_detector.py`, `tracker.py` | Haar cascade and why it only searches moving regions; centroid tracking and IDs |
| **Member 3** | `main.py`, `alerts.py`, `visualization.py`, README, PDF | Overall flow, alerts/logs, live run, limitations |

## Timeline
| Min | What | Who |
|---|---|---|
| 0:00–1:30 | Problem: why plain motion cameras give false alarms; our idea | 3 |
| 1:30–3:00 | Pipeline diagram: frame → preprocess → segment → track → face check → alert | 3 |
| 3:00–5:00 | Run `python main.py`. Point out the empty-room "learning" phase, then the box = **MOTION** (yellow) | 3 |
| 5:00–7:00 | Pause (`space`) on the mask. Explain MOG2, threshold, morphology | 1 |
| 7:00–9:30 | Person enters → **INTRUDER**. Explain Haar cascade, scaleFactor / minNeighbors, ROI trick | 2 |
| 9:30–11:00 | Show object IDs and blue trails; explain centroid matching | 2 |
| 11:00–12:30 | Open `outputs/snapshots` and `events.csv`; press `a` to disarm/arm | 3 |
| 12:30–14:00 | Limitations + improvements (live, honest) | all |
| 14:00–15:00 | Wrap-up | 3 |

Tips: run it once before presenting; keep the video in `sample_inputs/`; have `outputs/result.mp4` as a backup.

## Likely Q&A
1. **Why background subtraction instead of simple frame differencing?** MOG2 models the background statistically per pixel, so it tolerates small lighting changes and noise better than subtracting two frames.
2. **Why blur before motion detection?** Camera noise changes pixel values randomly; blurring prevents it from becoming false motion.
3. **Why threshold at 200?** MOG2 marks shadows as gray (127) and real foreground as 255. Thresholding at 200 keeps only true foreground.
4. **What do opening and dilation do?** Opening removes small white specks; dilation joins broken parts of one object.
5. **What is a Haar cascade?** A classifier trained on many face / non-face images using simple light-dark rectangle features, applied in stages so non-faces are rejected quickly.
6. **What do `scaleFactor` and `minNeighbors` do?** scaleFactor sets how the search window grows between scans (smaller = finer but slower); minNeighbors is how many overlapping hits are required (higher = fewer false positives).
7. **Why search only inside moving regions?** Faster and fewer false alarms; it also gives the logic "moving thing + face = person".
8. **How does the tracker keep IDs?** It matches each new centroid to the nearest previous centroid within a distance limit; unmatched ones become new objects.
9. **When does it fail?** Camera shake, profile/covered faces, lighting changes, a person standing still (absorbed into the background).
10. **How would you improve it?** DNN face detector, landmarks for eye/blink analysis, zones, e-mail alerts.
