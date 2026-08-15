import streamlit as st
import pandas as pd
import cv2
import os
import pickle
from datetime import datetime
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Attendance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

ATTENDANCE_FILE = "attendance.csv"
TRAINER_FILE = "trainer.yml"
NAMES_FILE = "names.pkl"
CASCADE_FILE = "haarcascade_frontalface_default.xml"


# ============================================================
# LOAD ATTENDANCE DATA
# ============================================================

if os.path.exists(ATTENDANCE_FILE):
    attendance_df = pd.read_csv(ATTENDANCE_FILE)
else:
    attendance_df = pd.DataFrame(
        columns=["Name", "Date", "Time"]
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 Smart Attendance")

    st.caption("AI-Powered Attendance Management")

    st.divider()

    page = st.radio(
        "Navigation",
        [
    "🏠 Dashboard",
    "📸 Face Recognition",
    "📋 Attendance Records",
    "📊 Analytics",
    "📈 Attendance Trends",
    "🔎 Search",
    "👥 Students",
    "📥 Reports",
    "⚙️ System",
    "ℹ️ About Project"
        ]
        

    st.divider()

    st.caption("Smart Attendance AI")
    st.caption("Face Recognition • Analytics")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🎓 Smart Attendance AI")

    st.write(
        "Intelligent attendance management using "
        "Face Recognition, Computer Vision and Data Analytics."
    )

    st.divider()

    # --------------------------------------------------------
    # CALCULATE DASHBOARD VALUES
    # --------------------------------------------------------

    total_records = len(attendance_df)

    if "Name" in attendance_df.columns:
        total_students = attendance_df["Name"].nunique()
    else:
        total_students = 0

    if "Date" in attendance_df.columns:
        total_days = attendance_df["Date"].nunique()
    else:
        total_days = 0

    today = datetime.now().strftime("%Y-%m-%d")

    if "Date" in attendance_df.columns:
        today_records = len(
            attendance_df[
                attendance_df["Date"].astype(str) == today
            ]
        )
    else:
        today_records = 0


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Students",
            total_students
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
            "🟢 Today",
            today_records
        )


    st.divider()


    # --------------------------------------------------------
    # TODAY'S ATTENDANCE
    # --------------------------------------------------------

    st.subheader("📅 Today's Attendance")

    if today_records > 0:

        today_df = attendance_df[
            attendance_df["Date"].astype(str) == today
        ]

        st.dataframe(
            today_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No attendance has been recorded today."
        )


    # --------------------------------------------------------
    # RECENT ATTENDANCE
    # --------------------------------------------------------

    st.subheader("🕒 Recent Attendance")

    if not attendance_df.empty:

        recent_df = attendance_df.tail(10)

        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Attendance records will appear here."
)
# ============================================================
# PART 2 — FACE RECOGNITION
# ============================================================

elif page == "📸 Face Recognition":

    st.title("📸 Face Recognition")

    st.write(
        "Capture an image and use the trained face recognition "
        "model to identify a registered student."
    )

    st.divider()

    # --------------------------------------------------------
    # CHECK REQUIRED FILES
    # --------------------------------------------------------

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
            "Face recognition model files are missing."
        )

        st.write("Please make sure the required files are available.")

    else:

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        try:

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(TRAINER_FILE)

            with open(NAMES_FILE, "rb") as f:
                names = pickle.load(f)

            face_cascade = cv2.CascadeClassifier(
                CASCADE_FILE
            )

            st.success(
                "Face recognition model loaded successfully."
            )

        except Exception as e:

            st.error(
                "Unable to load the face recognition model."
            )

            st.stop()


        # ----------------------------------------------------
        # CAMERA INPUT
        # ----------------------------------------------------

        st.subheader("📷 Capture Student Image")

        camera_image = st.camera_input(
            "Take a picture"
        )


        # ----------------------------------------------------
        # PROCESS IMAGE
        # ----------------------------------------------------

        if camera_image is not None:

            image = Image.open(camera_image)

            image_array = np.array(image)

            gray = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2GRAY
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

                recognized_name = None
                best_confidence = None

                # --------------------------------------------
                # RECOGNIZE FACE
                # --------------------------------------------

                for (x, y, w, h) in faces:

                    label, confidence = recognizer.predict(
                        gray[y:y+h, x:x+w]
                    )

                    if confidence < 70:

                        if label in names:

                            recognized_name = names[label]
                            best_confidence = confidence

                            cv2.rectangle(
                                image_array,
                                (x, y),
                                (x+w, y+h),
                                (0, 200, 0),
                                3
                            )

                            cv2.putText(
                                image_array,
                                str(recognized_name),
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 200, 0),
                                2
                            )

                    else:

                        cv2.rectangle(
                            image_array,
                            (x, y),
                            (x+w, y+h),
                            (220, 0, 0),
                            3
                        )

                        cv2.putText(
                            image_array,
                            "Unknown",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (220, 0, 0),
                            2
                        )


                # --------------------------------------------
                # DISPLAY RESULT
                # --------------------------------------------

                st.image(
                    image_array,
                    caption="Recognition Result",
                    use_container_width=True
                )


                # --------------------------------------------
                # MARK ATTENDANCE
                # --------------------------------------------

                if recognized_name is not None:

                    st.success(
                        f"✅ Recognized: {recognized_name}"
                    )

                    if best_confidence is not None:

                        st.metric(
                            "Recognition Confidence",
                            f"{best_confidence:.2f}"
                        )


                    now = datetime.now()

                    date = now.strftime(
                        "%Y-%m-%d"
                    )

                    time = now.strftime(
                        "%H:%M:%S"
                    )


                    # ----------------------------------------
                    # CREATE ATTENDANCE FILE
                    # ----------------------------------------

                    if not os.path.exists(
                        ATTENDANCE_FILE
                    ):

                        attendance_df = pd.DataFrame(
                            columns=[
                                "Name",
                                "Date",
                                "Time"
                            ]
                        )

                    else:

                        attendance_df = pd.read_csv(
                            ATTENDANCE_FILE
                        )


                    # ----------------------------------------
                    # PREVENT DUPLICATE ATTENDANCE
                    # ----------------------------------------

                    already_present = attendance_df[
                        (attendance_df["Name"] == recognized_name)
                        &
                        (attendance_df["Date"].astype(str) == date)
                    ]


                    if already_present.empty:

                        new_record = pd.DataFrame({
                            "Name": [recognized_name],
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
                            f"🎉 Attendance marked for {recognized_name}"
                        )

                    else:

                        st.info(
                            f"ℹ️ {recognized_name}'s attendance "
                            "is already marked today."
                        )

                else:

                    st.error(
                        "❌ Face not recognized."
                    )

        else:

            st.info(
                "📷 Use the camera above to capture a student's face."
        )
        # ============================================================
# PART 3 — ATTENDANCE ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Attendance Analytics")

    st.write(
        "Analyze attendance patterns, student participation, "
        "and daily attendance trends."
    )

    st.divider()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not os.path.exists(ATTENDANCE_FILE):

        st.info(
            "No attendance data is available yet."
        )

    else:

        df = pd.read_csv(ATTENDANCE_FILE)

        if df.empty:

            st.info(
                "No attendance records are available yet."
            )

        else:

            # ------------------------------------------------
            # DATA PREPARATION
            # ------------------------------------------------

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            df["Name"] = df["Name"].astype(str)

            total_records = len(df)

            total_students = df["Name"].nunique()

            total_days = df["Date"].nunique()

            today = pd.Timestamp.today().normalize()

            today_count = len(
                df[df["Date"] == today]
            )


            # ------------------------------------------------
            # KPI CARDS
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "👥 Students",
                    total_students
                )

            with col2:
                st.metric(
                    "📋 Total Records",
                    total_records
                )

            with col3:
                st.metric(
                    "📅 Active Days",
                    total_days
                )

            with col4:
                st.metric(
                    "🟢 Today",
                    today_count
                )


            st.divider()


            # ------------------------------------------------
            # STUDENT ATTENDANCE
            # ------------------------------------------------

            st.subheader(
                "👥 Student Attendance"
            )

            student_counts = (
                df["Name"]
                .value_counts()
                .reset_index()
            )

            student_counts.columns = [
                "Name",
                "Attendance"
            ]


            fig_students = px.bar(
                student_counts,
                x="Name",
                y="Attendance",
                title="Attendance by Student",
                text="Attendance"
            )

            fig_students.update_layout(
                xaxis_title="Student",
                yaxis_title="Attendance Count",
                height=450
            )

            st.plotly_chart(
                fig_students,
                use_container_width=True
            )


            # ------------------------------------------------
            # DAILY ATTENDANCE TREND
            # ------------------------------------------------

            st.subheader(
                "📈 Daily Attendance Trend"
            )

            daily_counts = (
                df.groupby("Date")
                .size()
                .reset_index(name="Attendance")
                .sort_values("Date")
            )


            fig_daily = px.line(
                daily_counts,
                x="Date",
                y="Attendance",
                markers=True,
                title="Attendance Over Time"
            )

            fig_daily.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Students",
                height=400
            )

            st.plotly_chart(
                fig_daily,
                use_container_width=True
            )


            # ------------------------------------------------
            # ATTENDANCE DISTRIBUTION
            # ------------------------------------------------

            st.subheader(
                "🥧 Attendance Distribution"
            )

            distribution = (
                df["Name"]
                .value_counts()
                .reset_index()
            )

            distribution.columns = [
                "Name",
                "Attendance"
            ]


            fig_pie = px.pie(
                distribution,
                names="Name",
                values="Attendance",
                title="Attendance Distribution",
                hole=0.4
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )


            # ------------------------------------------------
            # ATTENDANCE TABLE
            # ------------------------------------------------

            st.subheader(
                "📋 Attendance Summary"
            )

            st.dataframe(
                student_counts,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # DOWNLOAD REPORT
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📥 Download Attendance Data"
            )

            csv_data = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Attendance CSV",
                data=csv_data,
                file_name="attendance_report.csv",
                mime="text/csv"
    )
    # ============================================================
# PART 4 — ATTENDANCE RECORDS
# ============================================================

elif page == "📋 Attendance Records":

    st.title("📋 Attendance Records")

    st.write(
        "View, search, filter and download attendance records."
    )

    st.divider()

    if not os.path.exists(ATTENDANCE_FILE):

        st.info("No attendance records found.")

    else:

        df = pd.read_csv(ATTENDANCE_FILE)

        if df.empty:

            st.info("No attendance records available.")

        else:

            # Search
            search = st.text_input(
                "🔍 Search student",
                placeholder="Enter student name..."
            )

            if search:

                filtered_df = df[
                    df["Name"]
                    .astype(str)
                    .str.contains(
                        search,
                        case=False,
                        na=False
                    )
                ]

            else:

                filtered_df = df.copy()


            # Date filter
            if "Date" in filtered_df.columns:

                dates = sorted(
                    filtered_df["Date"]
                    .dropna()
                    .astype(str)
                    .unique(),
                    reverse=True
                )

                selected_date = st.selectbox(
                    "📅 Filter by date",
                    ["All Dates"] + dates
                )

                if selected_date != "All Dates":

                    filtered_df = filtered_df[
                        filtered_df["Date"].astype(str)
                        == selected_date
                    ]


            # Summary
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Records",
                    len(filtered_df)
                )

            with col2:
                st.metric(
                    "Students",
                    filtered_df["Name"].nunique()
                )

            with col3:
                st.metric(
                    "Days",
                    filtered_df["Date"].nunique()
                )


            st.divider()

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
                "⬇️ Download Filtered Records",
                data=csv_data,
                file_name="attendance_records.csv",
                mime="text/csv"
        )
        # ============================================================
# PART 5 — STUDENTS
# ============================================================

elif page == "👥 Students":

    st.title("👥 Registered Students")

    st.write(
        "View students detected and recorded by the attendance system."
    )

    st.divider()

    if not os.path.exists(NAMES_FILE):

        st.warning(
            "Student information file is not available."
        )

    else:

        try:

            with open(NAMES_FILE, "rb") as f:
                names = pickle.load(f)

            # Convert dictionary/list into a clean list
            if isinstance(names, dict):

                students = list(names.values())

            else:

                students = list(names)


            students = sorted(
                set(str(student) for student in students)
            )


            # ------------------------------------------------
            # STUDENT SUMMARY
            # ------------------------------------------------

            total_registered = len(students)

            attendance_df = load_attendance()

            if not attendance_df.empty:

                attendance_students = (
                    attendance_df["Name"]
                    .astype(str)
                    .nunique()
                )

            else:

                attendance_students = 0


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "👥 Registered Students",
                    total_registered
                )

            with col2:

                st.metric(
                    "✅ Students With Attendance",
                    attendance_students
                )


            st.divider()


            # ------------------------------------------------
            # STUDENT LIST
            # ------------------------------------------------

            st.subheader(
                "🎓 Student Directory"
            )

            student_data = pd.DataFrame({
                "S.No": range(
                    1,
                    len(students) + 1
                ),
                "Student Name": students
            })


            st.dataframe(
                student_data,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # ATTENDANCE COUNT
            # ------------------------------------------------

            st.subheader(
                "📊 Attendance by Student"
            )

            if not attendance_df.empty:

                student_attendance = (
                    attendance_df["Name"]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                student_attendance.columns = [
                    "Student",
                    "Attendance Count"
                ]

                st.dataframe(
                    student_attendance,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "Attendance statistics will appear after "
                    "students are recognized."
                )


        except Exception:

            st.error(
                "Unable to read student information."
                )
            # ============================================================
# PART 6 — ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Smart Attendance AI")

    st.write(
        "A computer vision based attendance management "
        "system designed to automate student attendance."
    )

    st.divider()


    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.subheader("🎯 Project Overview")

    st.write(
        "Smart Attendance AI uses face recognition to identify "
        "registered students and automatically record their "
        "attendance with date and time."
    )


    # --------------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------------

    st.subheader("🛠️ Technologies Used")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("🐍 Python")

    with col2:
        st.info("👁️ OpenCV")

    with col3:
        st.info("📊 Pandas")

    with col4:
        st.info("🚀 Streamlit")


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader("✨ Key Features")

    features = [
        "📸 Browser-based face capture",
        "🤖 Face recognition",
        "📝 Automatic attendance marking",
        "🔐 Duplicate attendance prevention",
        "📊 Attendance analytics",
        "📈 Interactive charts",
        "🔍 Attendance search and filtering",
        "📥 Attendance report download"
    ]

    for feature in features:

        st.write(
            f"✅ {feature}"
        )


    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.divider()

    st.subheader("🟢 System Status")

    status1, status2, status3, status4 = st.columns(4)


    with status1:

        if os.path.exists(TRAINER_FILE):
            st.success("Model Ready")
        else:
            st.error("Model Missing")


    with status2:

        if os.path.exists(NAMES_FILE):
            st.success("Students Ready")
        else:
            st.error("Names Missing")


    with status3:

        if os.path.exists(CASCADE_FILE):
            st.success("Face Detector Ready")
        else:
            st.error("Detector Missing")


    with status4:

        if os.path.exists(ATTENDANCE_FILE):
            st.success("Database Ready")
        else:
            st.warning("No Records")


    # --------------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("📌 Project Information")

    st.write(
        "**Smart Attendance System Using Face Recognition**"
    )

    st.write(
        "Built as a practical Machine Learning and Computer "
        "Vision project for automated attendance management."
        )
    # ============================================================
# PART 7 — ATTENDANCE TRENDS
# ============================================================

elif page == "📈 Attendance Trends":

    st.title("📈 Attendance Trends")

    st.write(
        "Explore attendance patterns across different dates."
    )

    st.divider()

    if not os.path.exists(ATTENDANCE_FILE):

        st.info("No attendance data available.")

    else:

        df = pd.read_csv(ATTENDANCE_FILE)

        if df.empty:

            st.info("No attendance records available.")

        else:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            daily = (
                df.groupby("Date")
                .agg(
                    Attendance=("Name", "count"),
                    Students=("Name", "nunique")
                )
                .reset_index()
                .sort_values("Date")
            )

            st.subheader("📊 Daily Attendance")

            fig = px.area(
                daily,
                x="Date",
                y="Attendance",
                markers=True,
                title="Daily Attendance Trend"
            )

            fig.update_layout(
                height=450,
                xaxis_title="Date",
                yaxis_title="Attendance"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("📋 Daily Summary")

            st.dataframe(
                daily,
                use_container_width=True,
                hide_index=True
    )
    # ============================================================
# PART 8 — ADVANCED SEARCH
# ============================================================

elif page == "🔎 Search":

    st.title("🔎 Search Attendance")

    st.write(
        "Quickly find attendance records using student name or date."
    )

    st.divider()

    if not os.path.exists(ATTENDANCE_FILE):

        st.info("Attendance database is empty.")

    else:

        df = pd.read_csv(ATTENDANCE_FILE)

        if df.empty:

            st.info("No attendance records available.")

        else:

            col1, col2 = st.columns(2)

            with col1:

                student_search = st.text_input(
                    "👤 Student Name",
                    placeholder="Search by name..."
                )

            with col2:

                date_search = st.text_input(
                    "📅 Date",
                    placeholder="YYYY-MM-DD"
                )


            result = df.copy()


            if student_search:

                result = result[
                    result["Name"]
                    .astype(str)
                    .str.contains(
                        student_search,
                        case=False,
                        na=False
                    )
                ]


            if date_search:

                result = result[
                    result["Date"]
                    .astype(str)
                    .str.contains(
                        date_search,
                        case=False,
                        na=False
                    )
                ]


            st.divider()

            st.metric(
                "🔎 Matching Records",
                len(result)
            )

            if result.empty:

                st.warning(
                    "No matching attendance records found."
                )

            else:

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
        )
        # ============================================================
# PART 9 — REPORT CENTER
# ============================================================

elif page == "📥 Reports":

    st.title("📥 Report Center")

    st.write(
        "Generate and download attendance reports."
    )

    st.divider()

    if not os.path.exists(ATTENDANCE_FILE):

        st.info("No attendance data available.")

    else:

        df = pd.read_csv(ATTENDANCE_FILE)

        if df.empty:

            st.info("No attendance records available.")

        else:

            st.subheader("📊 Report Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Records",
                    len(df)
                )

            with col2:
                st.metric(
                    "Students",
                    df["Name"].nunique()
                )

            with col3:
                st.metric(
                    "Days",
                    df["Date"].nunique()
                )


            st.divider()


            # Student summary

            summary = (
                df.groupby("Name")
                .size()
                .reset_index(
                    name="Attendance Count"
                )
                .sort_values(
                    "Attendance Count",
                    ascending=False
                )
            )


            st.subheader(
                "👥 Student Attendance Summary"
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )


            # CSV download

            csv_file = df.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                label="⬇️ Download Full Attendance Report",
                data=csv_file,
                file_name="smart_attendance_report.csv",
                mime="text/csv"
            )


            # Summary download

            summary_file = summary.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                label="⬇️ Download Student Summary",
                data=summary_file,
                file_name="student_attendance_summary.csv",
                mime="text/csv"
        )
        # ============================================================
# PART 10 — SYSTEM INFORMATION
# ============================================================

elif page == "⚙️ System":

    st.title("⚙️ System Information")

    st.write(
        "Monitor the components used by Smart Attendance AI."
    )

    st.divider()


    st.subheader("🤖 Recognition System")

    col1, col2 = st.columns(2)

    with col1:

        if os.path.exists(TRAINER_FILE):

            st.success(
                "✅ Face recognition model available"
            )

        else:

            st.error(
                "❌ Face recognition model unavailable"
            )


    with col2:

        if os.path.exists(NAMES_FILE):

            st.success(
                "✅ Student database available"
            )

        else:

            st.error(
                "❌ Student database unavailable"
            )


    st.divider()


    st.subheader("👁️ Computer Vision")

    if os.path.exists(CASCADE_FILE):

        st.success(
            "✅ Haar Cascade face detector available"
        )

    else:

        st.error(
            "❌ Haar Cascade file unavailable"
        )


    st.divider()


    st.subheader("📋 Attendance Database")

    if os.path.exists(ATTENDANCE_FILE):

        system_df = pd.read_csv(
            ATTENDANCE_FILE
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Records",
                len(system_df)
            )

        with col2:

            st.metric(
                "Students",
                system_df["Name"].nunique()
                if not system_df.empty
                else 0
            )

        with col3:

            st.metric(
                "Attendance Days",
                system_df["Date"].nunique()
                if not system_df.empty
                else 0
            )

    else:

        st.info(
            "Attendance database will be created "
            "when the first attendance is recorded."
        )


    st.divider()


    st.subheader("🛠️ Technology Stack")

    st.write(
        "🐍 Python"
    )

    st.write(
        "👁️ OpenCV"
    )

    st.write(
        "📊 Pandas"
    )

    st.write(
        "📈 Plotly"
    )

    st.write(
        "🚀 Streamlit"
    )


    st.divider()

    st.success(
        "🎓 Smart Attendance AI is ready."
        )
