"""MediaPipe face landmark wrapper.

This file supports two MediaPipe styles:
1. Older API: mp.solutions.face_mesh
2. Newer API: mediapipe.tasks FaceLandmarker with a .task model file
"""

import cv2
import mediapipe as mp

from src.utils.constants import FACE_LANDMARKER_MODEL_PATH


class _LandmarkList:
    """Makes newer MediaPipe task results look like old Face Mesh results."""

    def __init__(self, landmarks):
        self.landmark = landmarks


class _FaceResults:
    """Simple result object with the same field main.py expects."""

    def __init__(self, face_landmarks):
        self.multi_face_landmarks = face_landmarks


class FaceMeshDetector:
    """Detects facial landmarks using MediaPipe."""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mode = "solutions" if hasattr(mp, "solutions") else "tasks"
        self.face_mesh = None
        self.face_landmarker = None

        if self.mode == "solutions":
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles

            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            if not FACE_LANDMARKER_MODEL_PATH.exists():
                raise FileNotFoundError(
                    "MediaPipe Face Landmarker model is missing. "
                    f"Expected file: {FACE_LANDMARKER_MODEL_PATH}"
                )

            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )

            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
                running_mode=VisionTaskRunningMode.IMAGE,
                num_faces=max_num_faces,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame):
        """Returns face landmark results for a BGR OpenCV frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.mode == "solutions":
            rgb_frame.flags.writeable = False
            results = self.face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True
            return results

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        task_result = self.face_landmarker.detect(image)
        face_landmarks = [_LandmarkList(landmarks) for landmarks in task_result.face_landmarks]
        return _FaceResults(face_landmarks)

    def draw_landmarks(self, frame, face_landmarks) -> None:
        """Draws face mesh contours directly on the frame."""
        if self.mode == "tasks":
            height, width = frame.shape[:2]
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            return

        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style(),
        )

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.face_landmarker is not None:
            self.face_landmarker.close()
