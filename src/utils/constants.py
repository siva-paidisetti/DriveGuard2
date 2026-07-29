"""Project-wide settings for DriveGuard.

Beginners' note:
Keep the numbers in this file easy to find. In real testing, you will tune
thresholds because lighting, camera angle, and face shape can change results.
"""

from pathlib import Path


# Root folder of the project. This file is in src/utils/, so parents[2] is root.
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
RECORDING_DIR = DATA_DIR / "recordings"
ASSETS_DIR = BASE_DIR / "assets"
SOUNDS_DIR = ASSETS_DIR / "sounds"
MODELS_DIR = ASSETS_DIR / "models"
ALARM_SOUND_PATH = SOUNDS_DIR / "alarm.wav"
FACE_LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
ALERT_LOG_PATH = LOG_DIR / "alerts.csv"


# Webcam settings
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540


# Drowsiness settings
EAR_THRESHOLD = 0.23
DROWSY_SECONDS_LIMIT = 4.0


# Yawning settings
MAR_THRESHOLD = 0.60
YAWN_SECONDS_LIMIT = 4.0


# Distraction settings
HEAD_YAW_THRESHOLD = 35.0
HEAD_PITCH_THRESHOLD = 25.0
DISTRACTION_SECONDS_LIMIT = 4.0


# Alert cooldown prevents the same alert being logged every single frame.
ALERT_COOLDOWN_SECONDS = 3.0


# MediaPipe Face Mesh landmark indexes.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_VERTICAL_PAIRS = [(13, 14), (82, 87), (312, 317)]

HEAD_POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}
