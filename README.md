# 🎓 Smart Attendance System Using Face Recognition

An AI-powered attendance management system that uses **face recognition** to automatically identify registered students and record their attendance with date and time.

The project combines **Computer Vision, Machine Learning, Data Processing, Visualization, and Streamlit** into a complete web-based application.

---

## 🚀 Live Demo

🔗 **Streamlit Application:**  
[Open Smart Attendance System](YOUR_STREAMLIT_LINK_HERE)

---

## 📌 Project Overview

Traditional attendance systems can require manual work and consume valuable time.

This project provides an automated approach where a student's face is detected and recognized using a trained **LBPH Face Recognizer**. Once the student is recognized, the system records the attendance automatically.

The application also provides attendance records and analytics through an interactive Streamlit dashboard.

---

## ✨ Key Features

- 📸 Face detection and recognition
- 🤖 LBPH-based face recognition
- 📝 Automatic attendance marking
- 📅 Automatic date and time recording
- 👥 Student management
- 📋 Attendance record viewing
- 📊 Attendance analytics
- 📈 Interactive data visualization
- 🔎 Student search
- ⬇️ Attendance CSV download
- 🌐 Streamlit web application

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming |
| OpenCV | Face detection and recognition |
| LBPH | Face recognition algorithm |
| NumPy | Numerical operations |
| Pandas | Data processing and attendance management |
| Plotly | Interactive data visualization |
| Streamlit | Web application and deployment |
| Pickle | Storing student name mappings |

---

## 🧠 How It Works

### 1. Dataset Preparation

Images of registered students are collected and organized into the dataset.

### 2. Face Detection

OpenCV Haar Cascade is used to detect faces from images or camera input.

### 3. Model Training

The detected face images are used to train the LBPH face recognition model.

The trained model is saved as:

```text
trainer.yml
