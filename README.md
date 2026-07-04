# DriveGuard

Edge AI-Based Real-Time Driver Drowsiness and Distraction Detection System.

DriveGuard is a Python computer vision project that monitors a driver's face in
real time and detects:

- Eye closure for drowsiness
- Yawning using mouth opening
- Head turning left, right, up, or down for distraction
- Alert events saved into a CSV log
- Dashboard metrics and charts using Streamlit

## Tech Stack

- Python 3.11+
- OpenCV
- MediaPipe Face Mesh
- NumPy
- Pygame
- Streamlit

## Project Structure

```text
DriveGuard/
|-- assets/
|   `-- sounds/
|-- data/
|   |-- logs/
|   `-- recordings/
|-- docs/
|-- src/
|   |-- core/
|   |   |-- webcam.py
|   |   |-- face_mesh_detector.py
|   |   |-- drowsiness_detector.py
|   |   |-- yawn_detector.py
|   |   |-- distraction_detector.py
|   |   `-- alert_manager.py
|   |-- utils/
|   |   |-- logger.py
|   |   `-- constants.py
|   `-- ui/
|       `-- dashboard.py
|-- tests/
|-- requirements.txt
|-- README.md
`-- main.py
```

## File Guide

`main.py`
Starts the webcam, gets face landmarks, runs all detectors, shows the live
OpenCV window, triggers alerts, and exits when `q` is pressed.

`src/core/webcam.py`
Wraps OpenCV camera setup so the main file stays simple.

`src/core/face_mesh_detector.py`
Uses MediaPipe Face Mesh to find face landmarks.

`src/core/drowsiness_detector.py`
Calculates Eye Aspect Ratio (EAR). If EAR stays below the threshold for several
frames, DriveGuard marks the driver as drowsy.

`src/core/yawn_detector.py`
Calculates Mouth Aspect Ratio (MAR). If MAR stays above the threshold for
several frames, DriveGuard marks yawning as detected.

`src/core/distraction_detector.py`
Estimates head pose using OpenCV `solvePnP`. It detects left, right, up, and
down head turns.

`src/core/alert_manager.py`
Controls alert cooldowns, plays an alarm, and writes alert events to CSV.

`src/utils/constants.py`
Stores thresholds, file paths, webcam settings, and MediaPipe landmark indexes.
Tune this file first when improving accuracy.

`src/utils/logger.py`
Creates and updates the CSV alert log at `data/logs/alerts.csv`.

`src/ui/dashboard.py`
Streamlit dashboard that shows total alerts, alert history, and alert count
charts.

`requirements.txt`
Lists all Python dependencies needed by the project.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Real-Time Detection

```bash
python main.py
```

Press `q` to close the webcam window.

Type the command exactly as shown. Do not type `q` in PowerShell. The `q` key
only works after the webcam window opens and that window is selected.

## Run Dashboard

Open a second terminal from the project root and run:

```bash
streamlit run src/ui/dashboard.py
```

The dashboard reads alert history from:

```text
data/logs/alerts.csv
```

## Alarm Sound

Optional: place an alarm file here:

```text
assets/sounds/alarm.wav
```

If no sound file is present, DriveGuard uses a simple fallback beep on Windows.

## Troubleshooting

If you see `can't open file 'main.pypython'`, it means two commands were typed
together. Run only this:

```bash
python main.py
```

If no alert log exists yet, run the detector once. DriveGuard creates:

```text
data/logs/alerts.csv
```

If MediaPipe prints warnings about TensorFlow Lite or Clearcut, those are not
project errors. The app can still run.

## Beginner Tuning Tips

- If drowsiness is detected too easily, lower sensitivity by decreasing
  `EAR_THRESHOLD` or increasing `DROWSY_SECONDS_LIMIT`.
- If yawning is missed, lower `MAR_THRESHOLD`.
- If head movement alerts are too sensitive, increase `HEAD_YAW_THRESHOLD`,
  `HEAD_PITCH_THRESHOLD`, or `DISTRACTION_SECONDS_LIMIT`.

## Current Alert Timing

DriveGuard now uses continuous time instead of only frame counts:

- Eyes closed for 8 seconds: `DROWSINESS ALERT!`
- Mouth open/yawning for 8 seconds: `YAWNING DETECTED!`
- Head left/right/up/down for 8 seconds: `DISTRACTION ALERT!`

For drowsiness and distraction, DriveGuard also speaks:

```text
Please focus on driving
```
