import streamlit as st
import pandas as pd
import cv2
import os
import pickle
from datetime import datetime

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Smart Attendance System Using Face Recognition")

st.markdown("""
Welcome to the **Smart Attendance System**.

Use the menu below to:
- 📸 Face Recognition
- 📋 View Attendance
- 📊 Attendance Analytics
""")

# -----------------------------
# Sidebar
# -----------------------------
menu = st.sidebar.selectbox(
    "Select Option",
    [
        "Home",
        "Face Recognition",
        "Attendance",
        "Analytics",
        "About"
    ]
)

if menu == "Home":

    st.header("🏠 Smart Attendance System")

    st.success("Welcome!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Registered Students", 2)

    with col2:
        if os.path.exists("attendance.csv"):
            df = pd.read_csv("attendance.csv")
            st.metric("Attendance Records", len(df))
        else:
            st.metric("Attendance Records", 0)

    st.write("---")

    st.write("""
    ### Features

    ✅ Face Recognition

    ✅ Automatic Attendance

    ✅ Attendance Records

    ✅ Analytics Dashboard

    """)

elif menu == "Attendance":
    st.header("📋 Attendance Records")

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        st.dataframe(df)
    else:
        st.warning("attendance.csv not found.")
elif menu == "Analytics":

    st.header("📊 Attendance Analytics")

    if os.path.exists("attendance.csv"):

        df = pd.read_csv("attendance.csv")

        st.subheader("Attendance Data")
        st.dataframe(df)

        st.subheader("Students Attendance Count")

        chart = df["Name"].value_counts()

        st.bar_chart(chart)

        st.subheader("Attendance Distribution")

        st.line_chart(chart)

        st.metric("Total Attendance", len(df))
        st.metric("Total Students", df["Name"].nunique())

    else:

        st.warning("attendance.csv not found.")


elif menu == "About":
    st.header("ℹ️ About Project")
    st.write("""
    Smart Attendance System using Face Recognition

    Technologies Used:
    - Python
    - OpenCV
    - Streamlit
    - Pandas
    """)
elif menu == "Face Recognition":

    st.header("📸 Face Recognition")

    uploaded_file = st.file_uploader(
        "Upload a Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.read()

        import numpy as np

        image = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainer.yml")

        with open("names.pkl", "rb") as f:
            names = pickle.load(f)

        face_cascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml"
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        if len(faces) == 0:
            st.warning("No face detected.")

        attendance_file = "attendance.csv"

        if not os.path.exists(attendance_file):
            pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(
                attendance_file,
                index=False
            )

        df = pd.read_csv(attendance_file)

        for (x, y, w, h) in faces:

            label, confidence = recognizer.predict(
                gray[y:y+h, x:x+w]
            )

            if confidence < 70:

                name = names[label]
                color = (0,255,0)

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                already = df[
                    (df["Name"] == name) &
                    (df["Date"] == date)
                ]

                if already.empty:

                    new_row = pd.DataFrame({
                        "Name":[name],
                        "Date":[date],
                        "Time":[time]
                    })

                    df = pd.concat(
                        [df, new_row],
                        ignore_index=True
                    )

                    df.to_csv(
                        attendance_file,
                        index=False
                    )

                st.success(f"Attendance Marked : {name}")

            else:

                name = "Unknown"
                color = (0,0,255)

            cv2.rectangle(
                image,
                (x,y),
                (x+w,y+h),
                color,
                2
            )

            cv2.putText(
                image,
                name,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(image, use_container_width=True)
        
