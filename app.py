import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from PIL import Image
import os

st.write("Current Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir("."))

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Smart Attendance System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp{
background-color: var(--background-color);
color: var(--text-color);
}
.main-title{
    text-align:center;
    font-size:52px;
    font-weight:800;
    color:#1E3A8A;
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:20px;
}

.card{
    background:rgba(255,255,255,0.85);
    padding:25px;
    border-radius:18px;
    border:1px solid #ddd;
    text-align:center;
    margin-top:10px;
}

.card h2{
    color:#1E3A8A;
}

.card p{
    color:#222;
    font-size:18px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    detector = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    with open("names.pkl","rb") as f:
        names = pickle.load(f)

    return recognizer, detector, names

try:
    recognizer, detector, names = load_model()
except:
    st.error("Model files are missing.")
    st.stop()

# ---------------- SIDEBAR ----------------

st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=90
)

st.sidebar.title("AI Attendance")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🧠 Recognition",
        "📊 Analytics",
        "📚 About"
    ]
)

# ---------------- HOME ----------------

if page=="🏠 Home":

    st.markdown(
        "<div class='main-title'>🤖 AI Smart Attendance System</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Machine Learning Based Face Recognition Attendance Platform</div>",
        unsafe_allow_html=True
    )

    st.write("")

    st.success("System Loaded Successfully")

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class='card'>
        <h1>👤</h1>
        <h2>Face Detection</h2>
        <p>Haar Cascade</p>
        </div>
        """,unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card'>
        <h1>🧠</h1>
        <h2>Recognition</h2>
        <p>LBPH Algorithm</p>
        </div>
        """,unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='card'>
        <h1>📋</h1>
        <h2>Attendance</h2>
        <p>Automatic</p>
        </div>
        """,unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class='card'>
        <h1>📊</h1>
        <h2>Analytics</h2>
        <p>CSV Reports</p>
        </div>
        """,unsafe_allow_html=True)

    st.write("")

    st.subheader("🚀 Project Overview")

    st.write("""
This Smart Attendance System automatically identifies registered
users using Artificial Intelligence and Computer Vision.

### Features

- Face Detection
- Face Recognition
- Attendance Generation
- CSV Report Download
- Machine Learning Based Prediction
- Professional Dashboard

### Technology Stack

- Python
- OpenCV
- LBPH Face Recognition
- Streamlit
- Pandas
- NumPy

### Machine Learning Workflow

Dataset

⬇

Face Detection

⬇

Preprocessing

⬇

LBPH Training

⬇

Prediction

⬇

Attendance Generation
""")
# ==================================================
# RECOGNITION PAGE
# ==================================================

elif page == "🧠 Recognition":

    st.markdown(
        """
        <div class='main-title'>
        🧠 AI Face Recognition
        </div>

        <div class='subtitle'>
        Upload a student's face image for recognition
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    left, right = st.columns([2, 1])

    with left:

        uploaded = st.file_uploader(
            "📤 Upload Image",
            type=["jpg", "jpeg", "png"]
        )

    with right:

        st.info("""
### Instructions

✅ Face should be clear

✅ Look straight at camera

✅ Good lighting

✅ One face preferred

✅ JPG / PNG only
""")

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        frame = np.array(image)

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80,80)
        )

        st.success(f"Faces Detected : {len(faces)}")

        attendance=[]

        for (x,y,w,h) in faces:

            face = gray[y:y+h,x:x+w]

            face = cv2.resize(
                face,
                (200,200)
            )

            person_id, confidence = recognizer.predict(face)

            if confidence < 80:

                name = names.get(
                    person_id,
                    "Unknown"
                )

                color=(0,255,0)

            else:

                name="Unknown"

                color=(255,0,0)

            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                color,
                3
            )

            cv2.putText(
                frame,
                name,
                (x,y-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

            if name!="Unknown":

                attendance.append(
                    [
                        name,
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        round(confidence,2)
                    ]
                )

        st.write("")

        st.image(
            frame,
            caption="Recognition Result",
            use_container_width=True
        )

        st.write("")

        if attendance:

            st.success("🎉 Student Recognized Successfully")

            df = pd.DataFrame(
                attendance,
                columns=[
                    "Student Name",
                    "Attendance Time",
                    "Confidence Score"
                ]
            )

            df.to_csv(
                "attendance.csv",
                index=False
            )

            c1,c2,c3 = st.columns(3)

            c1.metric(
                "Recognized Faces",
                len(df)
            )

            c2.metric(
                "Average Confidence",
                f"{df['Confidence Score'].mean():.2f}"
            )

            c3.metric(
                "Attendance",
                "Marked"
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            st.download_button(
                "⬇ Download Attendance Report",
                df.to_csv(index=False),
                "attendance.csv",
                "text/csv"
            )

        else:

            st.error("❌ No Registered Face Found")

            st.warning("""
Possible Reasons

• Face is not in the trained dataset

• Image quality is poor

• Face is too small

• Lighting conditions are poor

• Model needs retraining
""")
# ==================================================
# ANALYTICS DASHBOARD
# ==================================================

elif page == "📊 Analytics":

    st.markdown("""
    <div class='main-title'>
    📊 Attendance Analytics Dashboard
    </div>

    <div class='subtitle'>
    Visual Attendance Statistics & Reports
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    try:

        df = pd.read_csv("attendance.csv")

        total_records = len(df)
        unique_students = df["Student Name"].nunique()
        latest = df.iloc[-1]["Attendance Time"]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "📋 Total Attendance",
            total_records
        )

        c2.metric(
            "👨‍🎓 Students",
            unique_students
        )

        c3.metric(
            "🕒 Latest Entry",
            latest
        )

        st.divider()

        st.subheader("📄 Attendance Records")

        st.dataframe(
            df,
            use_container_width=True,
            height=350
        )

        st.divider()

        st.subheader("📈 Student Attendance Count")

        chart = (
            df["Student Name"]
            .value_counts()
        )

        st.bar_chart(chart)

        st.divider()

        st.subheader("🥧 Attendance Distribution")

        st.area_chart(chart)

        st.divider()

        st.subheader("📥 Download Report")

        st.download_button(
            label="⬇ Download Attendance CSV",
            data=df.to_csv(index=False),
            file_name="attendance.csv",
            mime="text/csv"
        )

        st.success("Attendance report generated successfully.")

    except Exception:

        st.warning("No attendance records found.")

        st.info("""
Run Face Recognition first to generate attendance.

After recognition this dashboard will automatically
display statistics and charts.
""")
# ==================================================
# ABOUT PROJECT
# ==================================================

else:

    st.markdown("""
    <div class='main-title'>
    🚀 Smart Attendance System
    </div>

    <div class='subtitle'>
    AI & Machine Learning Project Portfolio
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📖 Project Description")

        st.write("""
The **Smart Attendance System Using Face Recognition** is a Machine
Learning application that automates attendance using facial recognition.

The system detects faces using the Haar Cascade algorithm and recognizes
registered users using the LBPH Face Recognizer. Attendance is recorded
automatically and can be downloaded as a CSV report.

This project demonstrates practical applications of Computer Vision,
Machine Learning, and Python for solving real-world attendance problems.
""")

    with col2:

        st.success("Project Status")

        st.metric("Version", "1.0")

        st.metric("Recognition", "LBPH")

        st.metric("Framework", "Streamlit")

        st.metric("Language", "Python")



    st.divider()

    st.subheader("🛠 Technologies Used")

    c1,c2,c3 = st.columns(3)

    with c1:

        st.info("""
### Programming

🐍 Python

📊 NumPy

📈 Pandas
""")

    with c2:

        st.info("""
### Computer Vision

👁 OpenCV

😀 Haar Cascade

🧠 LBPH
""")

    with c3:

        st.info("""
### Deployment

🌐 Streamlit

💻 GitHub

📄 CSV Reports
""")



    st.divider()

    st.subheader("⚙ Machine Learning Workflow")

    st.code("""

Dataset Collection
        │
        ▼
Image Preprocessing
        │
        ▼
Face Detection
(Haar Cascade)
        │
        ▼
Feature Extraction
        │
        ▼
LBPH Model Training
        │
        ▼
Face Recognition
        │
        ▼
Attendance Generation
        │
        ▼
CSV Report
""")



    st.divider()

    st.subheader("⭐ Key Features")

    st.write("""
✅ AI Powered Face Recognition

✅ Automatic Attendance Generation

✅ Machine Learning Prediction

✅ Image Upload Recognition

✅ Attendance Analytics Dashboard

✅ CSV Report Download

✅ Modern Web Interface

✅ Streamlit Deployment

✅ Beginner Friendly

✅ Real World ML Project
""")



    st.divider()

    st.subheader("🚀 Future Enhancements")

    st.write("""
• DeepFace Integration

• FaceNet Recognition

• Live Webcam Attendance

• Firebase Database

• Email Notifications

• QR + Face Hybrid Attendance

• Multi-Class Attendance

• Mobile Application

• Cloud Deployment
""")



    st.divider()

    st.subheader("👩‍💻 Developer")

    st.success("""
Name : Jhansi Reddy

Degree : B.Tech Computer Science Engineering

Domain : Artificial Intelligence & Machine Learning

Project : Smart Attendance System Using Face Recognition
""")


    st.divider()

    st.caption("© 2026 Smart Attendance System | Developed using Python, OpenCV, Streamlit and Machine Learning")