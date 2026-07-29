"""DriveGuard Live — browser-based version of main.py.

Runs the same detection logic (EAR, MAR, head pose) but takes video from the
visitor's own browser webcam via streamlit-webrtc, instead of a local OpenCV
window. This is the file to point Streamlit Cloud at for a real, shareable
live-detection link.
"""

import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.core.distraction_detector import DistractionDetector
from src.core.drowsiness_detector import DrowsinessDetector
from src.core.face_mesh_detector import FaceMeshDetector
from src.core.yawn_detector import YawnDetector
from src.utils.constants import ALERT_COOLDOWN_SECONDS, ALERT_LOG_PATH
from src.utils.logger import AlertLogger

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class WebAlertManager:
    """Browser-safe alert manager.

    Unlike the desktop AlertManager, this never tries to play a local sound
    or speak through PowerShell — that would run on the SERVER, not on the
    visitor's computer, so it would be pointless. This version only logs
    alerts to CSV and lets the video overlay show the warning text.
    """

    def __init__(self):
        self.logger = AlertLogger(ALERT_LOG_PATH)
        self.last_alert_times: dict[str, float] = {}
        self.active_alerts: set[str] = set()

    def trigger(self, alert_type: str, message: str) -> bool:
        if alert_type in self.active_alerts:
            return False

        now = time.time()
        previous_time = self.last_alert_times.get(alert_type, 0)
        if now - previous_time < ALERT_COOLDOWN_SECONDS:
            return False

        self.last_alert_times[alert_type] = now
        self.active_alerts.add(alert_type)
        self.logger.log(alert_type, message)
        return True

    def clear(self, alert_type: str) -> None:
        self.active_alerts.discard(alert_type)


def _draw_text(frame, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


class DriveGuardProcessor(VideoProcessorBase):
    """Runs one video frame at a time through the same detectors as main.py."""

    def __init__(self):
        self.face_detector = FaceMeshDetector()
        self.drowsiness_detector = DrowsinessDetector()
        self.yawn_detector = YawnDetector()
        self.distraction_detector = DistractionDetector()
        self.alert_manager = WebAlertManager()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        results = self.face_detector.detect(img)

        status_lines = []
        alert_lines = []

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            self.face_detector.draw_landmarks(img, face_landmarks)

            drowsy_result = self.drowsiness_detector.check(face_landmarks, img.shape)
            yawn_result = self.yawn_detector.check(face_landmarks, img.shape)
            distraction_result = self.distraction_detector.check(face_landmarks, img.shape)

            status_lines.extend(
                [
                    f"EAR: {drowsy_result['ear']:.2f}",
                    f"Eyes closed: {drowsy_result['closed_eye_seconds']:.1f}s / 4.0s",
                    f"MAR: {yawn_result['mar']:.2f}",
                    f"Mouth open: {yawn_result['open_mouth_seconds']:.1f}s / 4.0s",
                    f"Head: {distraction_result['direction']}",
                    f"Looking away: {distraction_result['distracted_seconds']:.1f}s / 4.0s",
                ]
            )

            if drowsy_result["is_drowsy"]:
                message = "DROWSINESS ALERT!"
                alert_lines.append(message)
                self.alert_manager.trigger("DROWSINESS", message)
            else:
                self.alert_manager.clear("DROWSINESS")

            if yawn_result["is_yawning"]:
                message = "YAWNING DETECTED!"
                alert_lines.append(message)
                self.alert_manager.trigger("YAWNING", message)
            else:
                self.alert_manager.clear("YAWNING")

            if distraction_result["is_distracted"]:
                message = "DISTRACTION ALERT!"
                alert_lines.append(message)
                self.alert_manager.trigger("DISTRACTION", message)
            else:
                self.alert_manager.clear("DISTRACTION")
        else:
            status_lines.append("No face detected")
            self.alert_manager.clear("DROWSINESS")
            self.alert_manager.clear("YAWNING")
            self.alert_manager.clear("DISTRACTION")

        for index, line in enumerate(status_lines):
            _draw_text(img, line, (20, 35 + index * 30), (255, 255, 255))

        for index, line in enumerate(alert_lines):
            _draw_text(img, line, (20, 260 + index * 40), (0, 0, 255))

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="DriveGuard Live", layout="wide")

st.title("DriveGuard — Live Detection")
st.write(
    "Click **Start** below and allow camera access. Detection runs live, "
    "right here in your browser — eye closure, yawning, and head-turn "
    "distraction are all monitored in real time."
)

webrtc_streamer(
    key="driveguard-live",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=DriveGuardProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.caption(
    "Note: alerts here are visual only (no sound), since this runs in your "
    "browser rather than on your local machine. All alerts are still logged "
    "and visible on the DriveGuard Dashboard."
)
