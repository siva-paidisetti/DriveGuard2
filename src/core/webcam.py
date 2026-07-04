"""Small wrapper around OpenCV webcam capture."""

import cv2

from src.utils.constants import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH


class Webcam:
    """Opens, reads, and releases the webcam."""

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def read(self):
        """Returns one frame from the webcam."""
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()
