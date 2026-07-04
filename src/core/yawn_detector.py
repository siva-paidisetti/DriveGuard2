"""Yawning detection using Mouth Aspect Ratio (MAR)."""

import math
import time

from src.utils.constants import MAR_THRESHOLD, MOUTH_LEFT, MOUTH_RIGHT, MOUTH_VERTICAL_PAIRS, YAWN_SECONDS_LIMIT


def _distance(point_a, point_b) -> float:
    return math.dist(point_a, point_b)


def _landmark_to_point(landmark, image_width: int, image_height: int) -> tuple[int, int]:
    return int(landmark.x * image_width), int(landmark.y * image_height)


class YawnDetector:
    """Calculates MAR and checks if the mouth stays open for several seconds."""

    def __init__(
        self,
        mar_threshold: float = MAR_THRESHOLD,
        seconds_limit: float = YAWN_SECONDS_LIMIT,
    ):
        self.mar_threshold = mar_threshold
        self.seconds_limit = seconds_limit
        self.open_mouth_frames = 0
        self.open_mouth_started_at = None

    def check(self, face_landmarks, frame_shape) -> dict:
        height, width = frame_shape[:2]
        landmarks = face_landmarks.landmark

        mouth_left = _landmark_to_point(landmarks[MOUTH_LEFT], width, height)
        mouth_right = _landmark_to_point(landmarks[MOUTH_RIGHT], width, height)
        horizontal = _distance(mouth_left, mouth_right)

        vertical_distances = []
        for upper_index, lower_index in MOUTH_VERTICAL_PAIRS:
            upper = _landmark_to_point(landmarks[upper_index], width, height)
            lower = _landmark_to_point(landmarks[lower_index], width, height)
            vertical_distances.append(_distance(upper, lower))

        mar = 0.0
        if horizontal != 0:
            mar = sum(vertical_distances) / (len(vertical_distances) * horizontal)

        if mar > self.mar_threshold:
            self.open_mouth_frames += 1
            if self.open_mouth_started_at is None:
                self.open_mouth_started_at = time.time()
        else:
            self.open_mouth_frames = 0
            self.open_mouth_started_at = None

        open_mouth_seconds = 0.0
        if self.open_mouth_started_at is not None:
            open_mouth_seconds = time.time() - self.open_mouth_started_at

        return {
            "mar": mar,
            "open_mouth_frames": self.open_mouth_frames,
            "open_mouth_seconds": open_mouth_seconds,
            "is_yawning": open_mouth_seconds >= self.seconds_limit,
        }
