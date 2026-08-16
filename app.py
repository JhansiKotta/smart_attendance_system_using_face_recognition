import streamlit as st
import pandas as pd
import cv2
import os
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ============================================================
# SMART ATTENDANCE SYSTEM
# PART 1 — IMPORTS & PROFESSIONAL UI
# ============================================================

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Smart Attendance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# File Paths
# ------------------------------------------------------------

ATTENDANCE_FILE = "attendance.csv"
TRAINER_FILE = "trainer.yml"
NAMES_FILE = "names.pkl"
CASCADE_FILE = "haarcascade_frontalface_default.xml"

# ------------------------------------------------------------
# Professional Light Theme
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Main content */
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #172554;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.06);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* Data tables */
    .stDataFrame {
        border-radius: 12px;
    }

    /* Hide footer */
    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Sidebar Branding
# ------------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:15px 5px 20px 5px;">
            <div style="font-size:48px;">🎓</div>
            <div style="font-size:24px; font-weight:800;">
                Smart Attendance
            </div>
            <div style="font-size:13px; opacity:0.75;">
                AI Attendance Management
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("## 🧭 Navigation")

    menu = st.radio(
        "Go to",
        [
            "🏠 Dashboard",
            "📸 Face Recognition",
            "📋 Attendance Records",
            "📊 Analytics",
            "👥 Students",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("Smart Attendance AI")
    st.caption("Face Recognition • Analytics")
# ============================================================
# PART 2 — DASHBOARD
# ============================================================

if menu == "🏠 Dashboard":

    st.title("🏠 Smart Attendance Dashboard")

    st.write(
        "Welcome to Smart Attendance AI — an intelligent "
        "attendance management system using face recognition."
    )

    st.divider()

    # Load attendance data
    if os.path.exists(ATTENDANCE_FILE):
        attendance_df = pd.read_csv(ATTENDANCE_FILE)
    else:
        attendance_df = pd.DataFrame(
            columns=["Name", "Date", "Time"]
        )

    # Dashboard statistics
    total_records = len(attendance_df)

    if not attendance_df.empty:
        total_students = attendance_df["Name"].nunique()
        total_days = attendance_df["Date"].nunique()
    else:
        total_students = 0
        total_days = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Registered Students",
            14
        )

    with col2:
        st.metric(
            "📋 Attendance Records",
            total_records
        )

    with col3:
        st.metric(
            "📅 Attendance Days",
            total_days
        )

    with col4:
        st.metric(
            "🎯 Recognition Model",
            "LBPH"
        )

    st.divider()

    # Project overview
    st.subheader("🚀 System Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
            ### 📸 Face Recognition

            Automatically identifies registered
            students using OpenCV and LBPH.
            """
        )

    with col2:
        st.success(
            """
            ### 📋 Automatic Attendance

            Attendance is recorded automatically
            with date and time.
            """
        )

    with col3:
        st.warning(
            """
            ### 📊 Analytics

            Analyze attendance records using
            charts and statistics.
            """
        )

    st.divider()

    # Recent attendance
    st.subheader("🕒 Recent Attendance")

    if not attendance_df.empty:

        recent_data = attendance_df.tail(10).iloc[::-1]

        st.dataframe(
            recent_data,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No attendance records available yet."
        )
# ============================================================
# PART 3 — ATTENDANCE RECORDS
# ============================================================

elif menu == "📋 Attendance Records":

    st.title("📋 Attendance Records")

    st.write(
        "View, search and download attendance records."
    )

    st.divider()

    if os.path.exists(ATTENDANCE_FILE):

        attendance_df = pd.read_csv(
            ATTENDANCE_FILE
        )

        if not attendance_df.empty:

            # Search
            search_name = st.text_input(
                "🔍 Search Student",
                placeholder="Enter student name..."
            )

            filtered_df = attendance_df.copy()

            if search_name:

                filtered_df = filtered_df[
                    filtered_df["Name"]
                    .astype(str)
                    .str.contains(
                        search_name,
                        case=False,
                        na=False
                    )
                ]

            # Statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📋 Total Records",
                    len(filtered_df)
                )

            with col2:
                st.metric(
                    "👥 Students",
                    filtered_df["Name"].nunique()
                )

            with col3:
                st.metric(
                    "📅 Days",
                    filtered_df["Date"].nunique()
                )

            st.divider()

            # Attendance table
            st.subheader("Attendance Data")

            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )

            # Download
            csv_data = filtered_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Attendance CSV",
                data=csv_data,
                file_name="attendance_records.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "Attendance file is empty."
            )

    else:

        st.warning(
            "attendance.csv was not found."
        )
 # ============================================================
# PART 4 — ATTENDANCE ANALYTICS
# ============================================================

elif menu == "📊 Analytics":

    st.title("📊 Attendance Analytics")

    st.write(
        "Analyze attendance patterns and student participation."
    )

    st.divider()

    if os.path.exists(ATTENDANCE_FILE):

        attendance_df = pd.read_csv(
            ATTENDANCE_FILE
        )

        if not attendance_df.empty:

            # ------------------------------------------------
            # Basic Statistics
            # ------------------------------------------------

            total_records = len(attendance_df)

            total_students = (
                attendance_df["Name"].nunique()
            )

            total_days = (
                attendance_df["Date"].nunique()
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📋 Total Attendance",
                    total_records
                )

            with col2:
                st.metric(
                    "👥 Total Students",
                    total_students
                )

            with col3:
                st.metric(
                    "📅 Working Days",
                    total_days
                )

            st.divider()

            # ------------------------------------------------
            # Attendance Count by Student
            # ------------------------------------------------

            st.subheader(
                "👥 Attendance by Student"
            )

            student_count = (
                attendance_df["Name"]
                .value_counts()
                .reset_index()
            )

            student_count.columns = [
                "Name",
                "Attendance"
            ]

            fig_bar = px.bar(
                student_count,
                x="Name",
                y="Attendance",
                title="Student Attendance Count",
                text="Attendance"
            )

            fig_bar.update_layout(
                xaxis_title="Student",
                yaxis_title="Attendance",
                template="plotly_white"
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

            # ------------------------------------------------
            # Attendance by Date
            # ------------------------------------------------

            st.subheader(
                "📅 Daily Attendance"
            )

            daily_count = (
                attendance_df["Date"]
                .value_counts()
                .sort_index()
                .reset_index()
            )

            daily_count.columns = [
                "Date",
                "Attendance"
            ]

            fig_line = px.line(
                daily_count,
                x="Date",
                y="Attendance",
                markers=True,
                title="Attendance Trend"
            )

            fig_line.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Students",
                template="plotly_white"
            )

            st.plotly_chart(
                fig_line,
                use_container_width=True
            )

            # ------------------------------------------------
            # Attendance Distribution
            # ------------------------------------------------

            st.subheader(
                "🥧 Attendance Distribution"
            )

            fig_pie = px.pie(
                student_count,
                names="Name",
                values="Attendance",
                title="Attendance Distribution"
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

        else:

            st.info(
                "No attendance data available for analytics."
            )

    else:

        st.warning(
            "attendance.csv was not found."
        )
# ============================================================
# PART 5 — FACE RECOGNITION
# ============================================================

elif menu == "📸 Face Recognition":

    st.title("📸 Face Recognition")

    st.write(
        "Capture a student's face and use the trained LBPH "
        "model to recognize the student."
    )

    st.divider()

    # Check required files
    required_files = [
        TRAINER_FILE,
        NAMES_FILE,
        CASCADE_FILE
    ]

    missing_files = [
        file for file in required_files
        if not os.path.exists(file)
    ]

    if missing_files:

        st.error(
            "Required recognition files are missing."
        )

        for file in missing_files:
            st.warning(f"⚠️ {file}")

    else:

        camera_image = st.camera_input(
            "📷 Take a photo"
        )

        if camera_image is not None:

            image_bytes = camera_image.getvalue()

            image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Load recognizer
            recognizer = (
                cv2.face.LBPHFaceRecognizer_create()
            )

            recognizer.read(TRAINER_FILE)

            # Load names
            with open(NAMES_FILE, "rb") as f:
                names = pickle.load(f)

            # Load Haar Cascade
            face_cascade = cv2.CascadeClassifier(
                CASCADE_FILE
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5
            )

            if len(faces) == 0:

                st.warning(
                    "No face detected. Please try again."
                )

            else:

                recognized = False

                for (x, y, w, h) in faces:

                    face = gray[
                        y:y + h,
                        x:x + w
                    ]

                    label, confidence = (
                        recognizer.predict(face)
                    )

                    if confidence < 70:

                        name = names.get(
                            label,
                            "Unknown"
                        )

                        recognized = True

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 180, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            name,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 180, 0),
                            2
                        )

                        # Create attendance file
                        if not os.path.exists(
                            ATTENDANCE_FILE
                        ):

                            pd.DataFrame(
                                columns=[
                                    "Name",
                                    "Date",
                                    "Time"
                                ]
                            ).to_csv(
                                ATTENDANCE_FILE,
                                index=False
                            )

                        attendance_df = pd.read_csv(
                            ATTENDANCE_FILE
                        )

                        now = datetime.now()

                        date = now.strftime(
                            "%Y-%m-%d"
                        )

                        time = now.strftime(
                            "%H:%M:%S"
                        )

                        already_present = (
                            (attendance_df["Name"] == name)
                            &
                            (attendance_df["Date"] == date)
                        )

                        if not already_present.any():

                            new_record = pd.DataFrame({
                                "Name": [name],
                                "Date": [date],
                                "Time": [time]
                            })

                            attendance_df = pd.concat(
                                [
                                    attendance_df,
                                    new_record
                                ],
                                ignore_index=True
                            )

                            attendance_df.to_csv(
                                ATTENDANCE_FILE,
                                index=False
                            )

                            st.success(
                                f"✅ Attendance marked for {name}"
                            )

                        else:

                            st.info(
                                f"ℹ️ {name} is already marked "
                                f"present today."
                            )

                    else:

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            "Unknown",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                # Display result
                result = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    result,
                    caption="Recognition Result",
                    use_container_width=True
                )

                if not recognized:

                    st.warning(
                        "Face detected, but the person "
                        "could not be recognized."
                    ) 
# ============================================================
# PART 6 — STUDENTS
# ============================================================

elif menu == "👥 Students":

    st.title("👥 Registered Students")

    st.write(
        "View students registered in the face recognition system."
    )

    st.divider()

    if os.path.exists(NAMES_FILE):

        with open(NAMES_FILE, "rb") as f:
            names = pickle.load(f)

        student_names = list(names.values())

        # Remove duplicates
        student_names = sorted(
            set(student_names)
        )

        search_student = st.text_input(
            "🔍 Search Student",
            placeholder="Enter student name..."
        )

        if search_student:

            displayed_students = [
                student
                for student in student_names
                if search_student.lower()
                in student.lower()
            ]

        else:

            displayed_students = student_names

        st.write(
            f"**{len(displayed_students)} students found**"
        )

        if displayed_students:

            columns = st.columns(3)

            for index, student in enumerate(
                displayed_students
            ):

                with columns[index % 3]:

                    if os.path.exists(
                        ATTENDANCE_FILE
                    ):

                        student_df = pd.read_csv(
                            ATTENDANCE_FILE
                        )

                        student_records = (
                            student_df[
                                student_df["Name"]
                                .astype(str)
                                == student
                            ]
                        )

                        attendance_count = len(
                            student_records
                        )

                    else:

                        attendance_count = 0

                    st.info(
                        f"""
                        ### 🎓 {student}

                        **Attendance Records:**  
                        {attendance_count}

                        **Recognition:**  
                        LBPH Face Recognition
                        """
                    )

        else:

            st.warning(
                "No student found."
            )

    else:

        st.warning(
            "names.pkl was not found."
        )
# ============================================================
# PART 7 — ABOUT PROJECT
# ============================================================

elif menu == "ℹ️ About Project":

    st.title("ℹ️ About Smart Attendance AI")

    st.write(
        "Smart Attendance AI is a face-recognition-based "
        "attendance management system."
    )

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write(
        "The system automatically recognizes registered "
        "students and records their attendance with the "
        "current date and time."
    )

    st.subheader("🧠 Technologies Used")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            ### 🐍 Python

            Core programming language used
            to develop the application.
            """
        )

    with col2:

        st.info(
            """
            ### 👁️ OpenCV

            Used for face detection and
            face recognition.
            """
        )

    with col3:

        st.info(
            """
            ### 📊 Pandas

            Used for attendance data
            management and analysis.
            """
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success(
            """
            ### 🎯 LBPH

            Local Binary Pattern Histogram
            face recognition algorithm.
            """
        )

    with col2:

        st.success(
            """
            ### 📈 Plotly

            Used to create interactive
            attendance charts.
            """
        )

    with col3:

        st.success(
            """
            ### 🌐 Streamlit

            Used to create the web
            application interface.
            """
        )

    st.divider()

    st.subheader("✨ Main Features")

    st.write(
        """
        ✅ Face Recognition

        ✅ Automatic Attendance

        ✅ Date and Time Recording

        ✅ Attendance Records

        ✅ Student Search

        ✅ Attendance Analytics

        ✅ Interactive Charts

        ✅ CSV Download
        """
    )
# ============================================================
# PART 8 — APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Smart Attendance AI  •  "
    "Face Recognition  •  "
    "Automated Attendance  •  "
    "Data Analytics"
)                    
