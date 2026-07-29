"""Head pose based distraction detection."""

import time

import cv2
import numpy as np

from src.utils.constants import (
    DISTRACTION_SECONDS_LIMIT,
    HEAD_PITCH_THRESHOLD,
    HEAD_POSE_LANDMARKS,
    HEAD_YAW_THRESHOLD,
)

# How many seconds of frames to average at the start to learn this
# person's own natural "looking forward" angle (their real resting head
# position, e.g. slightly tilted down toward a book/laptop), instead of
# assuming 0 degrees always means "facing the camera dead-on."
CALIBRATION_SECONDS = 2.0


class DistractionDetector:
    """Estimates head direction and checks if looking away lasts several seconds."""

    def __init__(
        self,
        yaw_threshold: float = HEAD_YAW_THRESHOLD,
        pitch_threshold: float = HEAD_PITCH_THRESHOLD,
        seconds_limit: float = DISTRACTION_SECONDS_LIMIT,
    ):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.seconds_limit = seconds_limit
        self.distracted_frames = 0
        self.distracted_started_at = None

        # Calibration state: collect early readings to find this person's
        # natural neutral pitch/yaw, then compare future frames to that.
        self._calibration_started_at = None
        self._calibration_samples: list[tuple[float, float]] = []
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self.is_calibrated = False

    def _image_point(self, landmarks, index: int, width: int, height: int) -> list[float]:
        landmark = landmarks[index]
        return [landmark.x * width, landmark.y * height]

    def check(self, face_landmarks, frame_shape) -> dict:
        height, width = frame_shape[:2]
        landmarks = face_landmarks.landmark

        image_points = np.array(
            [
                self._image_point(landmarks, HEAD_POSE_LANDMARKS["nose_tip"], width, height),
                self._image_point(landmarks, HEAD_POSE_LANDMARKS["chin"], width, height),
                self._image_point(landmarks, HEAD_POSE_LANDMARKS["left_eye_outer"], width, height),
                self._image_point(landmarks, HEAD_POSE_LANDMARKS["right_eye_outer"], width, height),
                self._image_point(landmarks, HEAD_POSE_LANDMARKS["left_mouth"], width, height),
                self._image_point(landmarks, HEAD_POSE_LANDMARKS["right_mouth"], width, height),
            ],
            dtype="double",
        )

        # Approximate 3D model points of a human face in millimeters.
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
            ],
            dtype="double",
        )

        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype="double",
        )
        distortion_coefficients = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return self._result(0.0, 0.0, "CENTER", 0.0, False)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

        raw_pitch = float(euler_angles[0][0])
        raw_yaw = float(euler_angles[1][0])

        # --- Calibration phase ---
        # For the first couple of seconds, just record readings to learn
        # this person's own natural resting head angle (their real
        # "looking forward" pose, e.g. slightly tilted down at a book or
        # laptop). We don't judge distraction yet during this phase.
        if not self.is_calibrated:
            if self._calibration_started_at is None:
                self._calibration_started_at = time.time()

            self._calibration_samples.append((raw_pitch, raw_yaw))

            if time.time() - self._calibration_started_at >= CALIBRATION_SECONDS:
                pitches = [sample[0] for sample in self._calibration_samples]
                yaws = [sample[1] for sample in self._calibration_samples]
                self.baseline_pitch = sum(pitches) / len(pitches)
                self.baseline_yaw = sum(yaws) / len(yaws)
                self.is_calibrated = True

            return self._result(raw_pitch, raw_yaw, "CALIBRATING", 0.0, False)

        # --- Normal detection, relative to this person's own baseline ---
        pitch = raw_pitch - self.baseline_pitch
        yaw = raw_yaw - self.baseline_yaw
        direction = "CENTER"

        if yaw < -self.yaw_threshold:
            direction = "LEFT"
        elif yaw > self.yaw_threshold:
            direction = "RIGHT"
        elif pitch < -self.pitch_threshold:
            direction = "UP"
        elif pitch > self.pitch_threshold:
            direction = "DOWN"

        is_looking_away = direction != "CENTER"

        if is_looking_away:
            self.distracted_frames += 1
            if self.distracted_started_at is None:
                self.distracted_started_at = time.time()
        else:
            self.distracted_frames = 0
            self.distracted_started_at = None

        distracted_seconds = 0.0
        if self.distracted_started_at is not None:
            distracted_seconds = time.time() - self.distracted_started_at

        return self._result(
            pitch=pitch,
            yaw=yaw,
            direction=direction,
            distracted_seconds=distracted_seconds,
            is_distracted=distracted_seconds >= self.seconds_limit,
        )

    def _result(
        self,
        pitch: float,
        yaw: float,
        direction: str,
        distracted_seconds: float,
        is_distracted: bool,
    ) -> dict:
        return {
            "pitch": pitch,
            "yaw": yaw,
            "direction": direction,
            "distracted_frames": self.distracted_frames,
            "distracted_seconds": distracted_seconds,
            "is_distracted": is_distracted,
        }
