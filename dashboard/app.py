import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Gate Monitor Dashboard", layout="wide")

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def safe_get(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except Exception:
        return None


def safe_post(url, timeout=5):
    try:
        r = requests.post(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def color_risk(val):
    try:
        v = float(val)
    except Exception:
        return ""
    if v >= 70:
        return "background-color:#7f1d1d;color:#fecaca"  # red
    elif v >= 40:
        return "background-color:#78350f;color:#fde68a"  # yellow
    else:
        return "background-color:#052e16;color:#bbf7d0"  # green


def anomaly_style(val):
    return "color:#f87171;font-weight:bold" if val else ""


# --------------------------------------------------------
# Header
# --------------------------------------------------------

st.title("🚧 Gate Monitor Dashboard")
st.caption("Real-time AI Surveillance System")

# --------------------------------------------------------
# Auto refresh
# --------------------------------------------------------

REFRESH_SEC = st.sidebar.slider("Refresh (seconds)", 2, 10, 5)
auto = st.sidebar.checkbox("Auto refresh", True)

# --------------------------------------------------------
# Stats
# --------------------------------------------------------

stats = safe_get(f"{API_BASE}/stats") or {
    "active_tracks": 0,
    "total_entries": 0,
    "alerts_today": 0,
}

c1, c2, c3 = st.columns(3)
c1.metric("Active Tracks", stats.get("active_tracks", 0))
c2.metric("Total Entries", stats.get("total_entries", 0))
c3.metric("Alerts Today", stats.get("alerts_today", 0))

st.divider()

# --------------------------------------------------------
# Main layout
# --------------------------------------------------------

left, right = st.columns([2, 1])

# --------------------------------------------------------
# ACTIVE TRACKS
# --------------------------------------------------------

with left:
    st.subheader("📍 Active Tracks")

    entries = safe_get(f"{API_BASE}/entries/active") or []

    if entries:
        df = pd.DataFrame(entries)

        # Select relevant columns safely
        cols = [c for c in [
            "track_id",
            "object_type",
            "risk_score",
            "is_anomaly",
            "dwell_seconds",
        ] if c in df.columns]

        df = df[cols]

        styled = (
            df.style
            .applymap(color_risk, subset=["risk_score"] if "risk_score" in df.columns else [])
            .applymap(anomaly_style, subset=["is_anomaly"] if "is_anomaly" in df.columns else [])
        )

        st.dataframe(styled, use_container_width=True)
    else:
        st.info("No active tracks")

# --------------------------------------------------------
# ALERTS
# --------------------------------------------------------

with right:
    st.subheader("🚨 Alerts")

    alerts = safe_get(f"{API_BASE}/alerts") or []

    if alerts:
        for alert in alerts:
            aid = alert.get("id")
            risk = alert.get("risk_score", 0)

            with st.container():
                st.markdown(
                    f"""
                    <div style="padding:10px;border-radius:8px;
                                background:#7f1d1d;color:white;margin-bottom:10px;">
                        <b>{alert.get('alert_type')}</b><br>
                        {alert.get('alert_message')}<br>
                        Risk: {risk}<br>
                        {alert.get('triggered_at')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(f"Acknowledge {aid}", key=f"ack_{aid}"):
                    ok = safe_post(f"{API_BASE}/alerts/{aid}/acknowledge")
                    if ok:
                        st.success("Acknowledged")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Failed to acknowledge")
    else:
        st.info("No alerts")

# --------------------------------------------------------
# Footer / auto refresh
# --------------------------------------------------------

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

if auto:
    time.sleep(REFRESH_SEC)
    st.rerun()
