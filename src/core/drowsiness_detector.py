"""Eye closure detection using Eye Aspect Ratio (EAR)."""

import math
import time

from src.utils.constants import DROWSY_SECONDS_LIMIT, EAR_THRESHOLD, LEFT_EYE, RIGHT_EYE


def _distance(point_a, point_b) -> float:
    return math.dist(point_a, point_b)


def _landmark_to_point(landmark, image_width: int, image_height: int) -> tuple[int, int]:
    return int(landmark.x * image_width), int(landmark.y * image_height)


class DrowsinessDetector:
    """Calculates EAR and checks if eyes stay closed for several seconds."""

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        seconds_limit: float = DROWSY_SECONDS_LIMIT,
    ):
        self.ear_threshold = ear_threshold
        self.seconds_limit = seconds_limit
        self.closed_eye_frames = 0
        self.closed_eye_started_at = None

    def _eye_aspect_ratio(self, landmarks, eye_indexes, width: int, height: int) -> float:
        points = [_landmark_to_point(landmarks[index], width, height) for index in eye_indexes]

        # Formula: EAR = (vertical distance 1 + vertical distance 2) / (2 * horizontal distance)
        vertical_1 = _distance(points[1], points[5])
        vertical_2 = _distance(points[2], points[4])
        horizontal = _distance(points[0], points[3])

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def check(self, face_landmarks, frame_shape) -> dict:
        height, width = frame_shape[:2]
        landmarks = face_landmarks.landmark

        left_ear = self._eye_aspect_ratio(landmarks, LEFT_EYE, width, height)
        right_ear = self._eye_aspect_ratio(landmarks, RIGHT_EYE, width, height)
        ear = (left_ear + right_ear) / 2.0

        if ear < self.ear_threshold:
            self.closed_eye_frames += 1
            if self.closed_eye_started_at is None:
                self.closed_eye_started_at = time.time()
        else:
            self.closed_eye_frames = 0
            self.closed_eye_started_at = None

        closed_eye_seconds = 0.0
        if self.closed_eye_started_at is not None:
            closed_eye_seconds = time.time() - self.closed_eye_started_at

        return {
            "ear": ear,
            "closed_eye_frames": self.closed_eye_frames,
            "closed_eye_seconds": closed_eye_seconds,
            "is_drowsy": closed_eye_seconds >= self.seconds_limit,
        }
