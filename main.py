"""DriveGuard main application.

Run this file to open the webcam and start real-time detection.

Exit key: q
"""

import cv2

from src.core.alert_manager import AlertManager
from src.core.distraction_detector import DistractionDetector
from src.core.drowsiness_detector import DrowsinessDetector
from src.core.face_mesh_detector import FaceMeshDetector
from src.core.webcam import Webcam
from src.core.yawn_detector import YawnDetector


def draw_text(frame, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Draws readable text on the video frame."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    webcam = Webcam()
    alert_manager = AlertManager()

    try:
        face_detector = FaceMeshDetector()
    except Exception as error:
        print("DriveGuard could not start face landmark detection.")
        print(error)
        webcam.release()
        return

    drowsiness_detector = DrowsinessDetector()
    yawn_detector = YawnDetector()
    distraction_detector = DistractionDetector()

    if not webcam.is_opened():
        print("Could not open webcam. Check camera permissions or CAMERA_INDEX.")
        return

    print("DriveGuard started. Press 'q' to exit.")

    try:
        while True:
            success, frame = webcam.read()
            if not success:
                print("Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            results = face_detector.detect(frame)

            status_lines = []
            alert_lines = []

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                face_detector.draw_landmarks(frame, face_landmarks)

                drowsy_result = drowsiness_detector.check(face_landmarks, frame.shape)
                yawn_result = yawn_detector.check(face_landmarks, frame.shape)
                distraction_result = distraction_detector.check(face_landmarks, frame.shape)

                status_lines.extend(
                    [
                        f"EAR: {drowsy_result['ear']:.2f}",
                        f"Eyes closed: {drowsy_result['closed_eye_seconds']:.1f}s / 8.0s",
                        f"MAR: {yawn_result['mar']:.2f}",
                        f"Mouth open: {yawn_result['open_mouth_seconds']:.1f}s / 8.0s",
                        f"Head: {distraction_result['direction']}",
                        f"Looking away: {distraction_result['distracted_seconds']:.1f}s / 8.0s",
                        f"Yaw: {distraction_result['yaw']:.1f} Pitch: {distraction_result['pitch']:.1f}",
                    ]
                )

                if drowsy_result["is_drowsy"]:
                    message = "DROWSINESS ALERT!"
                    alert_lines.append(message)
                    alert_manager.trigger(
                        "DROWSINESS",
                        message,
                        play_sound=True,
                        voice_message="Please focus on driving",
                    )
                else:
                    alert_manager.clear("DROWSINESS")

                if yawn_result["is_yawning"]:
                    message = "YAWNING DETECTED!"
                    alert_lines.append(message)
                    alert_manager.trigger("YAWNING", message, play_sound=False)
                else:
                    alert_manager.clear("YAWNING")

                if distraction_result["is_distracted"]:
                    message = "DISTRACTION ALERT!"
                    alert_lines.append(message)
                    alert_manager.trigger(
                        "DISTRACTION",
                        message,
                        play_sound=True,
                        voice_message="Please focus on driving",
                    )
                else:
                    alert_manager.clear("DISTRACTION")
            else:
                status_lines.append("No face detected")
                alert_manager.clear("DROWSINESS")
                alert_manager.clear("YAWNING")
                alert_manager.clear("DISTRACTION")

            for index, line in enumerate(status_lines):
                draw_text(frame, line, (20, 35 + index * 30), (255, 255, 255))

            for index, line in enumerate(alert_lines):
                draw_text(frame, line, (20, 180 + index * 40), (0, 0, 255))

            cv2.imshow("DriveGuard - Driver Monitoring", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        webcam.release()
        face_detector.close()
        cv2.destroyAllWindows()
        print("DriveGuard stopped.")


if __name__ == "__main__":
    main()
