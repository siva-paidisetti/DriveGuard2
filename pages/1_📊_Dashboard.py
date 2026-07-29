"""Dashboard page — part of the same deployed app as live_app.py, so it
reads the exact same alerts.csv file that live detection writes to."""

import csv
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import ALERT_LOG_PATH

st.set_page_config(page_title="DriveGuard Dashboard", page_icon="📊", layout="wide")

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


def read_alerts(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    with log_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


st.title("📊 DriveGuard Dashboard")
st.caption("Live alert data from the DriveGuard Live Detection page.")

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
    st.info("No alerts logged yet. Go to the Live Detection page and trigger one.")

st.subheader("Alert History")
if alerts:
    st.dataframe(alerts, use_container_width=True)
else:
    st.write("Alert history is empty.")
