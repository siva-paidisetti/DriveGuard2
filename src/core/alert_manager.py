"""Alert display, sound, and logging management."""

import subprocess
import threading
import time

from src.utils.constants import ALARM_SOUND_PATH, ALERT_COOLDOWN_SECONDS, ALERT_LOG_PATH
from src.utils.logger import AlertLogger


class AlertManager:
    """Handles alert cooldowns, alarm sound, and CSV logging."""

    def __init__(self):
        self.logger = AlertLogger(ALERT_LOG_PATH)
        self.last_alert_times: dict[str, float] = {}
        self.active_alerts: set[str] = set()
        self.sound_enabled = self._initialize_sound()

    def _initialize_sound(self) -> bool:
        """Initializes Pygame mixer if an alarm file exists."""
        if not ALARM_SOUND_PATH.exists():
            return False

        try:
            import pygame

            pygame.mixer.init()
            pygame.mixer.music.load(str(ALARM_SOUND_PATH))
            return True
        except Exception as error:
            print(f"Sound disabled: {error}")
            return False

    def trigger(
        self,
        alert_type: str,
        message: str,
        play_sound: bool = False,
        voice_message: str | None = None,
    ) -> bool:
        """Logs and optionally plays an alert.

        Returns True only when a new alert was actually triggered after cooldown.
        """
        if alert_type in self.active_alerts:
            return False

        now = time.time()
        previous_time = self.last_alert_times.get(alert_type, 0)

        if now - previous_time < ALERT_COOLDOWN_SECONDS:
            return False

        self.last_alert_times[alert_type] = now
        self.active_alerts.add(alert_type)
        self.logger.log(alert_type, message)

        if play_sound:
            self._play_alarm()

        if voice_message:
            self._speak(voice_message)

        return True

    def clear(self, alert_type: str) -> None:
        """Marks an alert as no longer active."""
        self.active_alerts.discard(alert_type)

    def _play_alarm(self) -> None:
        if self.sound_enabled:
            try:
                import pygame

                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
                return
            except Exception as error:
                print(f"Could not play alarm file: {error}")

        # Windows fallback beep. The app still works if no alarm.wav is present.
        threading.Thread(target=self._beep_pattern, daemon=True).start()

    def _beep_pattern(self) -> None:
        """Plays a short repeated alarm pattern without freezing the camera."""
        try:
            import winsound

            for _ in range(8):
                winsound.Beep(1200, 300)
                time.sleep(0.15)
        except Exception:
            print("ALARM!")

    def _speak(self, text: str) -> None:
        """Speaks a short voice alert on Windows without blocking the webcam."""
        safe_text = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Rate = 0; "
            "$speaker.Volume = 100; "
            f"$speaker.Speak('{safe_text}')"
        )

        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as error:
            print(f"Voice alert unavailable: {error}")
