import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from PIL import Image


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="AI Smart Attendance System",
    page_icon="🤖",
    layout="wide"
)


# ================= CUSTOM DESIGN =================

st.markdown(
"""
<style>

body{
background:#f5f7fb;
}


.main-title{
font-size:55px;
font-weight:900;
text-align:center;
color:#0B3D91;
}


.tagline{
font-size:24px;
text-align:center;
color:#555;
}


.card{
background:white;
padding:25px;
border-radius:20px;
box-shadow:0px 8px 25px rgba(0,0,0,0.12);
text-align:center;
}


.feature{
font-size:18px;
line-height:2;
}


.section-title{
font-size:35px;
font-weight:800;
color:#0B3D91;
}


</style>
""",
unsafe_allow_html=True
)



# ================= LOAD MODEL =================


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
    ) as f:

        names = pickle.load(f)


    return recognizer, detector, names



try:

    recognizer, detector, names = load_model()

except Exception as e:

    st.error(
        "Model files are missing. Upload trainer.yml, names.pkl and Haar Cascade file."
    )

    st.stop()



# ================= SIDEBAR =================


st.sidebar.markdown(
"""
# 🤖 AI Attendance

Smart Face Recognition System

---
"""
)


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🧠 Recognition",
        "📊 Analytics",
        "🚀 Project Info"
    ]
)



# ==================================================
# HOME PAGE
# ==================================================


if page=="🏠 Home":


    st.markdown(
    """
    <div class="main-title">
    🤖 AI Smart Attendance System
    </div>

    <div class="tagline">
    Machine Learning Based Face Recognition Attendance Platform
    </div>

    """,
    unsafe_allow_html=True
    )


    st.write("")


    st.success(
        "🟢 AI Model Loaded Successfully"
    )



    st.markdown(
    """
    <div class="section-title">
    About The Project
    </div>
    """,
    unsafe_allow_html=True
    )


    st.write(
"""
Smart Attendance System is an Artificial Intelligence
based attendance automation solution.

It uses Computer Vision and Machine Learning algorithms
to detect faces, recognize registered users and generate
attendance records automatically.
"""
)



    st.markdown(
    """
    <div class="section-title">
    Key Features
    </div>
    """,
    unsafe_allow_html=True
    )



    c1,c2,c3,c4 = st.columns(4)


    with c1:
        st.markdown(
        """
        <div class="card">

        👤

        ### Face Detection

        Haar Cascade

        </div>
        """,
        unsafe_allow_html=True
        )


    with c2:
        st.markdown(
        """
        <div class="card">

        🧠

        ### Recognition

        LBPH Model

        </div>
        """,
        unsafe_allow_html=True
        )


    with c3:
        st.markdown(
        """
        <div class="card">

        📋

        ### Attendance

        Auto Records

        </div>
        """,
        unsafe_allow_html=True
        )


    with c4:
        st.markdown(
        """
        <div class="card">

        📊

        ### Analytics

        Reports

        </div>
        """,
        unsafe_allow_html=True
        )



    st.markdown(
    """
    ## ⚙️ Machine Learning Pipeline

    Dataset  
    ↓  
    Face Detection  
    ↓  
    Image Processing  
    ↓  
    Feature Extraction  
    ↓  
    LBPH Training  
    ↓  
    Face Prediction  
    ↓  
    Attendance Generation

    """
    )



# ==================================================
# RECOGNITION PAGE
# ==================================================


elif page=="🧠 Recognition":


    st.markdown(
    """
    <div class="section-title">
    🧠 AI Face Recognition Module
    </div>
    """,
    unsafe_allow_html=True
    )


    uploaded = st.file_uploader(
        "Upload a face image",
        type=[
            "jpg",
            "jpeg",
            "png"
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
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80,80)
        )


        st.info(
            f"Faces Detected: {len(faces)}"
        )


        attendance=[]



        for x,y,w,h in faces:


            face=gray[
                y:y+h,
                x:x+w
            ]


            face=cv2.resize(
                face,
                (200,200)
            )


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
            caption="AI Recognition Result",
            use_container_width=True
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
                "✅ Attendance Generated Successfully"
            )


            st.dataframe(
                df,
                use_container_width=True
            )


        else:

            st.warning(
                "No Known Face Found"
            )



# ==================================================
# ANALYTICS
# ==================================================


elif page=="📊 Analytics":


    st.markdown(
    """
    <div class="section-title">
    📊 Attendance Analytics
    </div>
    """,
    unsafe_allow_html=True
    )


    try:


        df=pd.read_csv(
            "attendance.csv"
        )


        a,b,c=st.columns(3)


        a.metric(
            "Total Records",
            len(df)
        )


        b.metric(
            "Students",
            df["Name"].nunique()
        )


        c.metric(
            "Latest Attendance",
            df.iloc[-1]["Time"]
        )



        st.dataframe(
            df,
            use_container_width=True
        )



        st.download_button(
            "⬇ Download Attendance Report",
            df.to_csv(index=False),
            "attendance.csv"
        )



    except:


        st.info(
            "Attendance data not available"
        )



# ==================================================
# PROJECT INFO
# ==================================================


else:


    st.markdown(
    """
    <div class="section-title">
    🚀 Project Information
    </div>
    """,
    unsafe_allow_html=True
    )


    st.write(
"""
## Technologies Used

🐍 Python

👁 OpenCV

🤖 LBPH Face Recognition

📊 Pandas

🔢 NumPy

🌐 Streamlit


## ML Concepts

• Image Processing

• Feature Extraction

• Classification

• Model Prediction


## Future Improvements

✅ CNN Based Recognition

✅ FaceNet / DeepFace

✅ Cloud Database

✅ Real Time CCTV Attendance


## Developer

**Jhansi Reddy**

Computer Science Engineering Student

Aspiring AI/ML Engineer
"""
)