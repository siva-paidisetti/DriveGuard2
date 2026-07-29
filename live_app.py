"""DriveGuard Live — browser-based version of main.py.

Runs the same detection logic (EAR, MAR, head pose) but takes video from the
visitor's own browser webcam via streamlit-webrtc, instead of a local OpenCV
window.

v2 changes:
- Plays a real alert sound in the visitor's browser (not the server) using a
  small shared, thread-safe flag between the video-processing thread and the
  main Streamlit script.
- Skips per-point face-mesh drawing (very expensive on a CPU-only server) and
  lowers the requested camera resolution, to remove the lag/freeze feeling.
"""

import base64
import threading
import time

import av
import cv2
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.core.distraction_detector import DistractionDetector
from src.core.drowsiness_detector import DrowsinessDetector
from src.core.face_mesh_detector import FaceMeshDetector
from src.core.yawn_detector import YawnDetector
from src.utils.constants import ALERT_COOLDOWN_SECONDS, ALERT_LOG_PATH, ASSETS_DIR
from src.utils.logger import AlertLogger

# How often (in seconds) the beep/voice repeats while an alert condition
# (drowsy / yawning / distracted) stays continuously true.
ALERT_REPEAT_SECONDS = 1.5

# Alert types that should also say "Please focus on driving" out loud,
# matching the desktop app's voice message.
SPOKEN_ALERTS = {"DROWSINESS", "DISTRACTION"}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

BEEP_PATH = ASSETS_DIR / "beep.wav"
BEEP_BASE64 = None
if BEEP_PATH.exists():
    BEEP_BASE64 = base64.b64encode(BEEP_PATH.read_bytes()).decode("ascii")


class SharedAlertState:
    """Thread-safe flag used to pass 'an alert just fired' from the video
    processing thread (no Streamlit access) to the main script thread
    (which can render an autoplay <audio> tag in the browser)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None

    def set_alert(self, alert_type: str) -> None:
        with self._lock:
            self._pending = alert_type

    def pop_alert(self):
        with self._lock:
            alert = self._pending
            self._pending = None
            return alert


class WebAlertManager:
    """Browser-safe alert manager: logs to CSV and marks the shared flag so
    the main script can play a sound. Never tries local sound/PowerShell
    (that would run on the server, not the visitor's computer)."""

    def __init__(self, shared_alert_state: SharedAlertState):
        self.logger = AlertLogger(ALERT_LOG_PATH)
        self.last_alert_times: dict[str, float] = {}
        self.active_alerts: set[str] = set()
        self.shared_alert_state = shared_alert_state

    def trigger(self, alert_type: str, message: str) -> bool:
        now = time.time()
        previous_time = self.last_alert_times.get(alert_type, 0)

        # First time this alert becomes active: log it once to CSV.
        if alert_type not in self.active_alerts:
            self.active_alerts.add(alert_type)
            self.logger.log(alert_type, message)

        # Re-fire the sound/voice repeatedly (every ALERT_REPEAT_SECONDS)
        # for as long as the condition stays true, instead of only once.
        if now - previous_time < ALERT_REPEAT_SECONDS:
            return False

        self.last_alert_times[alert_type] = now
        self.shared_alert_state.set_alert(alert_type)
        return True

    def clear(self, alert_type: str) -> None:
        self.active_alerts.discard(alert_type)


def _draw_text(frame, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


class DriveGuardProcessor(VideoProcessorBase):
    """Runs each video frame through the same detectors as main.py.

    Detection runs on every frame (needed for accurate timing), but the
    expensive per-point face-mesh drawing has been removed to keep this
    fast enough for a shared CPU-only cloud server.
    """

    def __init__(self, shared_alert_state: SharedAlertState):
        self.face_detector = FaceMeshDetector()
        self.drowsiness_detector = DrowsinessDetector()
        self.yawn_detector = YawnDetector()
        self.distraction_detector = DistractionDetector()
        self.alert_manager = WebAlertManager(shared_alert_state)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        results = self.face_detector.detect(img)

        status_lines = []
        alert_lines = []

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # Note: dense landmark drawing intentionally skipped for speed.

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


st.set_page_config(page_title="DriveGuard Live", page_icon="🚗", layout="wide")

BACKGROUND_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 20%, rgba(0,217,192,0.15), transparent 40%),
                radial-gradient(circle at 85% 80%, rgba(0,140,255,0.12), transparent 45%),
                #0B0F19;
}
</style>
"""
st.markdown(BACKGROUND_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div style="padding: 6px 0 0 0;">
        <span style="background-color:#00D9C0; color:#0B0F19; padding:4px 12px;
        border-radius:20px; font-weight:700; font-size:12px; letter-spacing:1px;">
        ● LIVE</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🚗 DriveGuard — Live Detection")
st.write(
    "Real-time driver monitoring, running directly in your browser. "
    "Click **Start**, allow camera access, and DriveGuard will watch for "
    "signs of drowsiness, yawning, and distraction as they happen."
)

feature_1, feature_2, feature_3 = st.columns(3)
with feature_1:
    st.info("👁️ **Drowsiness**\n\nTracks eye closure using Eye Aspect Ratio (EAR).")
with feature_2:
    st.info("🥱 **Yawning**\n\nTracks mouth opening using Mouth Aspect Ratio (MAR).")
with feature_3:
    st.info("↔️ **Distraction**\n\nTracks head turns using head-pose estimation.")

if not BEEP_BASE64:
    st.warning(
        "Alert sound file not found at assets/beep.wav — "
        "alerts will still show visually, just without sound.",
        icon="🔇",
    )

st.divider()

if "alert_state" not in st.session_state:
    st.session_state["alert_state"] = SharedAlertState()
shared_alert_state = st.session_state["alert_state"]


def _make_processor():
    return DriveGuardProcessor(shared_alert_state)


webrtc_ctx = webrtc_streamer(
    key="driveguard-live",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=_make_processor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480},
            "height": {"ideal": 360},
            "frameRate": {"ideal": 15, "max": 20},
        },
        "audio": False,
    },
    async_processing=True,
)

st.caption(
    "Tip: performance depends on your internet connection and the free "
    "server's available CPU — lower resolution keeps it responsive."
)

audio_placeholder = st.empty()

VOICE_MESSAGE = "Please focus on driving"


def _alert_html(alert_type: str, nonce: str) -> str:
    """Builds a tiny standalone HTML page (beep + spoken message) to run
    inside components.html's iframe, where <script> tags actually execute
    (unlike st.markdown, which silently strips/ignores <script> tags)."""
    beep_tag = ""
    if BEEP_BASE64:
        beep_tag = f"""
        <audio autoplay="true">
            <source src="data:audio/wav;base64,{BEEP_BASE64}" type="audio/wav">
        </audio>
        """

    speak_script = ""
    if alert_type in SPOKEN_ALERTS:
        speak_script = f"""
        <script>
            try {{
                var msg = new SpeechSynthesisUtterance({VOICE_MESSAGE!r});
                msg.rate = 1.0;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(msg);
            }} catch (e) {{}}
        </script>
        """

    return f"<!-- {nonce} -->{beep_tag}{speak_script}"


# Browsers require ONE real click on the page before they allow any
# automatic sound/voice later on. This button "unlocks" audio + speech
# for the rest of the session — click it once after pressing START.
if st.button("🔔 Enable Sound Alerts (click once)"):
    with audio_placeholder.container():
        components.html(
            _alert_html("DISTRACTION", "unlock-" + str(time.time())),
            height=0,
        )
    st.success("Sound enabled. Alerts will now beep and speak automatically.")

if webrtc_ctx.state.playing:
    while webrtc_ctx.state.playing:
        alert_type = shared_alert_state.pop_alert()
        if alert_type:
            with audio_placeholder.container():
                components.html(
                    _alert_html(alert_type, str(time.time())),
                    height=0,
                )
        time.sleep(0.3)
