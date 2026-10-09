"""
Smart Attendance AI
Face-recognition attendance system (Streamlit + OpenCV LBPH).

Environment variables / Streamlit secrets (all optional):
    ADMIN_PASSWORD        protects Records, Analytics and Students pages
    APP_TIMEZONE          default "Asia/Kolkata"
    DATA_DIR              folder for attendance.csv (mount a volume here)
    CONFIDENCE_THRESHOLD  LBPH distance limit, default 70 (lower = stricter)
"""

import logging
import os
import pickle
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Smart Attendance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smart-attendance")

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR))
DATA_DIR.mkdir(parents=True, exist_ok=True)

ATTENDANCE_FILE = DATA_DIR / "attendance.csv"
TRAINER_FILE = BASE_DIR / "trainer.yml"
NAMES_FILE = BASE_DIR / "names.pkl"
CASCADE_FILE = BASE_DIR / "haarcascade_frontalface_default.xml"

COLUMNS = ["Name", "Date", "Time"]
TIMEZONE = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Kolkata"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "70"))

_write_lock = threading.Lock()


def get_secret(key: str):
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


# ------------------------------------------------------------
# Theme
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #f5f7fb; }
    .block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] { background-color: #111827; }
    section[data-testid="stSidebar"] * { color: white; }
    h1, h2, h3 { color: #172554; }
    div[data-testid="stMetric"] {
        background-color: white; border: 1px solid #e5e7eb;
        border-radius: 18px; padding: 20px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] * { color: #172554; }
    .stButton > button { border-radius: 10px; font-weight: 600; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------
def load_attendance() -> pd.DataFrame:
    """Load attendance data; always returns a DataFrame with COLUMNS."""
    if not ATTENDANCE_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)
    df.columns = df.columns.str.strip()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["Name"] = df["Name"].astype(str).str.strip()
    return df[COLUMNS]


def mark_attendance(name: str) -> bool:
    """Mark attendance once per student per day. Returns True if newly marked."""
    now = datetime.now(TIMEZONE)
    date, time_ = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

    with _write_lock:
        df = load_attendance()
        if ((df["Name"] == name) & (df["Date"] == date)).any():
            return False
        row = pd.DataFrame({"Name": [name], "Date": [date], "Time": [time_]})
        pd.concat([df, row], ignore_index=True).to_csv(ATTENDANCE_FILE, index=False)
        log.info("Attendance marked: %s on %s", name, date)
        return True


@st.cache_data(show_spinner=False)
def _load_names(mtime: float) -> dict:
    with open(NAMES_FILE, "rb") as f:
        return pickle.load(f)


def get_names() -> dict:
    if not NAMES_FILE.exists():
        return {}
    return _load_names(NAMES_FILE.stat().st_mtime)


@st.cache_resource(show_spinner="Loading face recognition model...")
def _load_models(trainer_mtime: float, cascade_path: str):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(TRAINER_FILE))
    cascade = cv2.CascadeClassifier(cascade_path)
    return recognizer, cascade


def get_models():
    """Cached models; reloads automatically if trainer.yml is retrained."""
    cascade_path = (
        str(CASCADE_FILE)
        if CASCADE_FILE.exists()
        else cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return _load_models(TRAINER_FILE.stat().st_mtime, cascade_path)


# ------------------------------------------------------------
# Admin protection (only active if ADMIN_PASSWORD is set)
# ------------------------------------------------------------
def require_admin() -> bool:
    password = get_secret("ADMIN_PASSWORD")
    if not password or st.session_state.get("is_admin"):
        return True

    st.warning("🔒 This page is restricted to administrators.")
    entered = st.text_input("Admin password", type="password")
    if st.button("Login"):
        if entered == str(password):
            st.session_state["is_admin"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
PAGES = [
    "🏠 Dashboard",
    "📸 Face Recognition",
    "📋 Attendance Records",
    "📊 Analytics",
    "👥 Students",
    "ℹ️ About Project",
]

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:15px 5px 20px 5px;">
            <div style="font-size:48px;">🎓</div>
            <div style="font-size:24px; font-weight:800;">Smart Attendance</div>
            <div style="font-size:13px; opacity:0.75;">AI Attendance Management</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    menu = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.divider()
    if st.session_state.get("is_admin"):
        if st.button("Logout", width="stretch"):
            st.session_state["is_admin"] = False
            st.rerun()
    st.caption("Face Recognition • Analytics")


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def page_dashboard():
    st.title("🏠 Smart Attendance Dashboard")
    st.write(
        "An intelligent attendance management system using face recognition."
    )
    st.divider()

    df = load_attendance()
    registered = len(set(get_names().values()))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Registered Students", registered)
    c2.metric("📋 Attendance Records", len(df))
    c3.metric("📅 Attendance Days", df["Date"].nunique() if not df.empty else 0)
    c4.metric("🎯 Recognition Model", "LBPH")

    st.divider()
    st.subheader("🚀 System Features")
    c1, c2, c3 = st.columns(3)
    c1.info("### 📸 Face Recognition\n\nIdentifies registered students using OpenCV and LBPH.")
    c2.success("### 📋 Automatic Attendance\n\nAttendance is recorded with date and time.")
    c3.warning("### 📊 Analytics\n\nAnalyze attendance using charts and statistics.")

    st.divider()
    st.subheader("🕒 Recent Attendance")
    if df.empty:
        st.info("No attendance records available yet.")
    else:
        st.dataframe(df.tail(10).iloc[::-1], hide_index=True)


def page_recognition():
    st.title("📸 Face Recognition")
    st.write("Take a photo to recognize a student and mark attendance.")
    st.divider()

    missing = [f.name for f in (TRAINER_FILE, NAMES_FILE) if not f.exists()]
    if missing:
        st.error("Required recognition files are missing:")
        for f in missing:
            st.warning(f"⚠️ {f}")
        return

    photo = st.camera_input("📷 Take a photo")
    if photo is None:
        return

    frame = cv2.imdecode(
        np.frombuffer(photo.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        st.error("Could not read the image. Please try again.")
        return

    try:
        recognizer, cascade = get_models()
    except Exception:
        log.exception("Failed to load models")
        st.error("The recognition model could not be loaded. Re-train and try again.")
        return

    names = get_names()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        st.warning("No face detected. Face the camera in good light and retry.")
        return

    recognized_any = False
    for (x, y, w, h) in faces:
        label, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < CONFIDENCE_THRESHOLD and label in names:
            name = names[label]
            recognized_any = True
            colour, text = (0, 180, 0), name

            if mark_attendance(name):
                st.success(f"✅ Attendance marked for {name}")
            else:
                st.info(f"ℹ️ {name} is already marked present today.")
        else:
            colour, text = (0, 0, 255), "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(
            frame, text, (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2,
        )

    st.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        caption="Recognition Result",
        width="stretch",
    )
    if not recognized_any:
        st.warning("Face detected, but the person could not be recognized.")


def page_records():
    st.title("📋 Attendance Records")
    st.write("View, search and download attendance records.")
    st.divider()
    if not require_admin():
        return

    df = load_attendance()
    if df.empty:
        st.info("No attendance records yet.")
        return

    c1, c2 = st.columns([2, 1])
    search = c1.text_input("🔍 Search Student", placeholder="Enter student name...")
    dates = sorted(df["Date"].unique(), reverse=True)
    chosen = c2.selectbox("📅 Date", ["All dates"] + dates)

    filtered = df
    if search:
        filtered = filtered[filtered["Name"].str.contains(search, case=False, na=False)]
    if chosen != "All dates":
        filtered = filtered[filtered["Date"] == chosen]

    c1, c2, c3 = st.columns(3)
    c1.metric("📋 Total Records", len(filtered))
    c2.metric("👥 Students", filtered["Name"].nunique())
    c3.metric("📅 Days", filtered["Date"].nunique())

    st.divider()
    st.dataframe(filtered, hide_index=True)
    st.download_button(
        "⬇️ Download Attendance CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="attendance_records.csv",
        mime="text/csv",
        width="stretch",
    )


def page_analytics():
    st.title("📊 Attendance Analytics")
    st.write("Analyze attendance patterns and student participation.")
    st.divider()
    if not require_admin():
        return

    df = load_attendance()
    if df.empty:
        st.info("No attendance data available for analytics.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("📋 Total Attendance", len(df))
    c2.metric("👥 Total Students", df["Name"].nunique())
    c3.metric("📅 Working Days", df["Date"].nunique())
    st.divider()

    per_student = df["Name"].value_counts().reset_index()
    per_student.columns = ["Name", "Attendance"]

    st.subheader("👥 Attendance by Student")
    fig = px.bar(per_student, x="Name", y="Attendance", text="Attendance")
    fig.update_layout(template="plotly_white", xaxis_title="Student")
    st.plotly_chart(fig)

    st.subheader("📅 Daily Attendance")
    daily = df["Date"].value_counts().sort_index().reset_index()
    daily.columns = ["Date", "Attendance"]
    fig = px.line(daily, x="Date", y="Attendance", markers=True)
    fig.update_layout(template="plotly_white", yaxis_title="Students Present")
    st.plotly_chart(fig)

    st.subheader("🥧 Attendance Distribution")
    st.plotly_chart(px.pie(per_student, names="Name", values="Attendance"))


def page_students():
    st.title("👥 Registered Students")
    st.write("Students registered in the face recognition system.")
    st.divider()
    if not require_admin():
        return

    names = get_names()
    if not names:
        st.warning("names.pkl was not found or is empty.")
        return

    students = sorted(set(names.values()))
    search = st.text_input("🔍 Search Student", placeholder="Enter student name...")
    if search:
        students = [s for s in students if search.lower() in s.lower()]

    st.write(f"**{len(students)} students found**")
    if not students:
        st.warning("No student found.")
        return

    counts = load_attendance()["Name"].value_counts().to_dict()
    cols = st.columns(3)
    for i, student in enumerate(students):
        cols[i % 3].info(
            f"### 🎓 {student}\n\n"
            f"**Attendance Records:** {counts.get(student, 0)}\n\n"
            f"**Recognition:** LBPH Face Recognition"
        )


def page_about():
    st.title("ℹ️ About Smart Attendance AI")
    st.write("A face-recognition-based attendance management system.")
    st.divider()

    st.subheader("🎯 Project Objective")
    st.write(
        "The system automatically recognizes registered students and records "
        "their attendance with the current date and time."
    )

    st.subheader("🧠 Technologies Used")
    c1, c2, c3 = st.columns(3)
    c1.info("### 🐍 Python\n\nCore programming language.")
    c2.info("### 👁️ OpenCV\n\nFace detection and recognition.")
    c3.info("### 📊 Pandas\n\nAttendance data management.")
    c1, c2, c3 = st.columns(3)
    c1.success("### 🎯 LBPH\n\nLocal Binary Pattern Histogram algorithm.")
    c2.success("### 📈 Plotly\n\nInteractive attendance charts.")
    c3.success("### 🌐 Streamlit\n\nWeb application interface.")

    st.divider()
    st.subheader("✨ Main Features")
    st.markdown(
        "- ✅ Face Recognition\n- ✅ Automatic Attendance\n"
        "- ✅ Date and Time Recording\n- ✅ Searchable Records\n"
        "- ✅ Attendance Analytics\n- ✅ CSV Download"
    )


ROUTES = {
    "🏠 Dashboard": page_dashboard,
    "📸 Face Recognition": page_recognition,
    "📋 Attendance Records": page_records,
    "📊 Analytics": page_analytics,
    "👥 Students": page_students,
    "ℹ️ About Project": page_about,
}

ROUTES[menu]()

st.divider()
st.caption("🎓 Smart Attendance AI • Face Recognition • Automated Attendance • Data Analytics")