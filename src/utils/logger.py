"""CSV logging helpers for DriveGuard alerts."""

import csv
from datetime import datetime
from pathlib import Path


class AlertLogger:
    """Writes alert events to a CSV file.

    CSV is beginner-friendly and can be opened in Excel, Google Sheets, or read
    by Streamlit for the dashboard.
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        if self.log_path.exists():
            return

        with self.log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "alert_type", "message"])

    def log(self, alert_type: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, alert_type, message])
