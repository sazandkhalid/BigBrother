# 🎥 Interview Emotion & Performance Analysis System

*A Computer Vision & Behavioral Analytics Project from Hoyalytics*

## 📌 Overview

This project was developed as part of an **interview assessment for Hoyalytics**. The goal was to analyze recorded interviews and provide interviewees with **actionable feedback** on their performance using **computer vision and behavioral analytics**.

The system uses facial expression detection, body movement tracking, and audio cadence analysis to evaluate key emotional and engagement metrics throughout an interview.

---

## 🧠 Key Features

* 🎭 **Emotion Detection** via PyFeat facial landmark tracking
* 👁️ **Eye Contact & Gaze Stability** using Haar Cascades
* ✋ **Body Movement & Gesture Analysis**
* ⏱️ **Talking Speed & Silence Duration Metrics**
* 📊 **Engagement and Nervousness Scores**
* 🧾 **Curated Summary Reports** (e.g., % time spent smiling, nervous, disengaged)

---

## 🔧 Tools & Technologies

| Category         | Stack                                |
| ---------------- | ------------------------------------ |
| Language         | Python 3.9                           |
| CV Libraries     | `OpenCV`, `PyFeat`, `dlib`           |
| Video Processing | `cv2`, `ffmpeg`, `moviepy`           |
| Data Handling    | `pandas`, `numpy`                    |
| Visualization    | `matplotlib`, `seaborn`              |
| Model Evaluation | Custom thresholds, empirical testing |

---

## 🧪 How It Works

1. **Video Ingestion**

   * Accepts `.mp4` or `.avi` format
   * Frames are extracted at regular intervals

2. **Facial Feature Detection**

   * Uses PyFeat to detect AU (Action Units) and map them to emotions
   * Haar Cascades identify eyes, face, and hand movement patterns

3. **Behavioral Analysis**

   * Duration of speech vs silence
   * Detection of nervousness (e.g. lip biting, eye darting)
   * Engagement level computed from facial orientation and gestures

4. **Summary Generation**

   * Outputs a report with time percentages for detected emotions
   * Engagement and nervousness scoring system
   * Visuals showing facial activity over time

---

## 📁 Folder Structure

```
interview-emotion-analysis/
│
├── data/                # Sample interview videos
├── scripts/             # Feature extraction and analysis code
│   ├── emotion_tracker.py
│   ├── body_movement.py
│   └── engagement_score.py
├── reports/             # Auto-generated feedback reports
├── notebooks/           # Exploratory analysis and testing
├── README.md            # Project overview
└── requirements.txt     # Dependencies
```

---

## 📊 Sample Outputs

| Metric                   | Value      |
| ------------------------ | ---------- |
| Smiling (Duration %)     | 15.2%      |
| Nervous (Detected Peaks) | 8          |
| Talking Time             | 82%        |
| Eye Contact Score        | 91.4 / 100 |
| Engagement Level         | High       |

📈 Emotion chart over time
📍 Gaze heatmap
📝 Text-based summary of tips and insights

---

## 🚀 Impact

✅ Delivered a **functioning emotion analysis pipeline** under interview constraints
✅ Showcased **CV and behavioral modeling** for real-world application
✅ Demonstrated ability to build **insightful, measurable feedback systems**

---

## 👤 My Contributions

* Developed all core **feature extraction and processing scripts**
* Tuned emotion detection thresholds via empirical testing
* Built logic for **engagement & nervousness scoring**
* Conducted integration testing across modules

---

## 📍 Notes

* All videos used were synthetic or consented samples
* This system is not intended for clinical or diagnostic use
* Requires \~30 FPS interview video for best performance

