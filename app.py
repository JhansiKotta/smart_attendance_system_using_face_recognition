import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from PIL import Image


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Smart Attendance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- STYLE ----------------

st.markdown(
"""
<style>

body {
    background-color:#f5f7fb;
}


.main-title {
    font-size:50px;
    font-weight:800;
    color:#12355b;
    text-align:center;
}


.subtitle {
    font-size:22px;
    text-align:center;
    color:#555;
}


.card {

    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.1);
    text-align:center;

}


.section {

    font-size:35px;
    font-weight:bold;
    color:#12355b;

}


.stButton button {

    border-radius:15px;
    height:50px;
    font-size:18px;

}


</style>
""",
unsafe_allow_html=True
)



# ---------------- LOAD MODEL ----------------


@st.cache_resource
def load_model():

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read(
        "trainer.yml"
    )


    detector = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )


    with open(
        "names.pkl",
        "rb"
    ) as file:

        names = pickle.load(file)


    return recognizer, detector, names



try:

    recognizer, detector, names = load_model()


except:

    st.error(
        "Required model files are missing."
    )

    st.stop()



# ---------------- SIDEBAR ----------------


st.sidebar.title(
    "🎓 Smart Attendance AI"
)


st.sidebar.markdown(
"""
---
### Navigation

Choose a module:

"""
)


page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "🤖 AI Recognition",
        "📊 Attendance Dashboard",
        "🚀 Project Details"
    ]
)



# ==================================================
# HOME
# ==================================================


if page=="🏠 Home":


    st.markdown(
    """
    <div class="main-title">
    🎓 Smart Attendance System
    </div>

    <div class="subtitle">
    AI Powered Face Recognition Attendance Management System
    </div>

    """,
    unsafe_allow_html=True
    )


    st.write("")


    st.success(
        "🚀 Artificial Intelligence System Online"
    )


    st.markdown(
    "## 🌟 About The Project"
    )


    st.write(
"""
Smart Attendance System is an AI based attendance
automation platform that uses Computer Vision technology
to identify individuals and automatically record attendance.

The system replaces traditional manual attendance methods
with a faster, smarter and more accurate solution.
"""
)



    st.markdown(
    "## ✨ Key Features"
    )



    c1,c2,c3 = st.columns(3)


    with c1:

        st.markdown(
        """
        <div class="card">

        👤

        ### Face Detection

        Detects human faces
        using Haar Cascade.

        </div>
        """,
        unsafe_allow_html=True
        )


    with c2:

        st.markdown(
        """
        <div class="card">

        🤖

        ### Face Recognition

        Identifies users
        using LBPH algorithm.

        </div>
        """,
        unsafe_allow_html=True
        )


    with c3:

        st.markdown(
        """
        <div class="card">

        📋

        ### Smart Attendance

        Automatically saves
        attendance records.

        </div>
        """,
        unsafe_allow_html=True
        )




    st.markdown(
    "## 🛠 Technology Stack"
    )


    tech1,tech2,tech3,tech4 = st.columns(4)


    tech1.info("🐍 Python")
    tech2.info("👁 OpenCV")
    tech3.info("📊 Pandas")
    tech4.info("🌐 Streamlit")




    st.markdown(
    "## 🔄 System Workflow"
    )


    st.write(
"""
1️⃣ Collect Face Dataset

⬇

2️⃣ Train Recognition Model

⬇

3️⃣ Detect Face

⬇

4️⃣ Recognize Person

⬇

5️⃣ Store Attendance With Time
"""
)



# ==================================================
# RECOGNITION
# ==================================================


elif page=="🤖 AI Recognition":


    st.markdown(
    "## 🤖 AI Face Recognition Module"
    )


    st.write(
    """
    Upload a person's image.
    The AI model will detect and recognize the face.
    """
    )


    uploaded = st.file_uploader(
        "Upload Image",
        type=[
            "jpg",
            "png",
            "jpeg"
        ]
    )


    if uploaded:


        image = Image.open(uploaded)


        frame=np.array(image)


        gray=cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )


        faces=detector.detectMultiScale(
            gray,
            1.2,
            5
        )


        attendance=[]



        for x,y,w,h in faces:


            face=gray[
                y:y+h,
                x:x+w
            ]


            person_id,confidence = recognizer.predict(
                face
            )


            if confidence < 80:

                name=names.get(
                    person_id,
                    "Unknown"
                )

            else:

                name="Unknown"



            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                3
            )


            cv2.putText(
                frame,
                name,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
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



        st.image(
            frame,
            caption="AI Recognition Result"
        )



        if attendance:


            df=pd.DataFrame(
                attendance,
                columns=[
                    "Name",
                    "Time",
                    "Confidence"
                ]
            )


            df.to_csv(
                "attendance.csv",
                index=False
            )


            st.success(
                "✅ Attendance Successfully Marked"
            )


            st.dataframe(
                df,
                use_container_width=True
            )


        else:

            st.warning(
                "Face not recognized"
            )




# ==================================================
# DASHBOARD
# ==================================================


elif page=="📊 Attendance Dashboard":


    st.markdown(
    "## 📊 Attendance Analytics"
    )


    try:

        df=pd.read_csv(
            "attendance.csv"
        )


        col1,col2,col3=st.columns(3)


        col1.metric(
            "Total Attendance",
            len(df)
        )


        col2.metric(
            "Unique Persons",
            df["Name"].nunique()
        )


        col3.metric(
            "Latest Entry",
            df.iloc[-1]["Time"]
        )



        st.dataframe(
            df,
            use_container_width=True
        )


        st.download_button(
            "⬇ Download Report",
            df.to_csv(index=False),
            "attendance.csv"
        )


    except:


        st.info(
            "No attendance data available"
        )





# ==================================================
# DETAILS
# ==================================================


else:


    st.markdown(
    "## 🚀 Project Information"
    )


    st.write(
"""
### Smart Attendance System Using Face Recognition


### Objective

To create an intelligent attendance system
that automatically identifies people using
Artificial Intelligence.


### Algorithms

• Haar Cascade Face Detection

• LBPH Face Recognition


### Future Improvements

✔ Deep Learning Face Recognition

✔ Cloud Database

✔ Mobile Application

✔ Real Time Camera Attendance


### Developer

**Jhansi Reddy**

AI / ML Engineering Student

"""
)