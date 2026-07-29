"""Streamlit dashboard for DriveGuard alert logs."""

import csv
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

# Streamlit runs this file from src/ui, so we add the project root to Python's
# import path before importing our own src package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import ALERT_LOG_PATH


def read_alerts(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []

    with log_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def delete_alerts(log_path: Path) -> None:
    """Clears all logged alerts, keeping just the CSV header row so future
    alerts still get logged correctly."""
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "alert_type", "message"])


def main() -> None:
    st.set_page_config(page_title="DriveGuard Dashboard", page_icon="DG", layout="wide")
    st.title("DriveGuard Dashboard")

    alerts = read_alerts(ALERT_LOG_PATH)
    total_alerts = len(alerts)
    counts = Counter(row["alert_type"] for row in alerts)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts", total_alerts)
    col2.metric("Drowsiness Alerts", counts.get("DROWSINESS", 0))
    col3.metric("Distraction Alerts", counts.get("DISTRACTION", 0))

    st.subheader("Alert Counts")
    if counts:
        chart_data = [{"alert_type": key, "count": value} for key, value in counts.items()]
        st.bar_chart(chart_data, x="alert_type", y="count")
    else:
        st.info("No alerts logged yet. Run main.py to start detection.")

    st.subheader("Alert History")
    if alerts:
        st.dataframe(alerts, use_container_width=True)

        st.divider()
        if "confirm_clear" not in st.session_state:
            st.session_state["confirm_clear"] = False

        if not st.session_state["confirm_clear"]:
            if st.button("🗑️ Clear Alert History"):
                st.session_state["confirm_clear"] = True
                st.rerun()
        else:
            st.warning("This will permanently delete all logged alerts. Are you sure?")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("✅ Yes, delete everything", type="primary"):
                    delete_alerts(ALERT_LOG_PATH)
                    st.session_state["confirm_clear"] = False
                    st.success("Alert history cleared.")
                    st.rerun()
            with cancel_col:
                if st.button("Cancel"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()
    else:
        st.write("Alert history is empty.")


if __name__ == "__main__":
    main()
