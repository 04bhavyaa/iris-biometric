# Iris Recognition System

## 📌 Overview
This project implements an **Iris Recognition System** using **ORB (Oriented FAST and Rotated BRIEF) feature extraction** and **Brute Force Matching (BFMatcher)** to authenticate individuals based on iris images.

## 🔧 Features
- Extracts key points from iris images using **ORB detector**.
- Matches key points between two iris images using **BFMatcher**.
- Computes a **similarity score** to determine if the images belong to the same person.
- **Visualizes results** with matched key points.

## 🏗 Technologies Used
- **Python**
- **OpenCV** (cv2) for image processing
- **NumPy** for numerical operations
- **Matplotlib** for result visualization

## 📊 Output
- Displays the **match score**.
- Shows images with detected **key points and matched features**.
- Identifies if the images belong to the **same person** or **different people**.

## 📌 Notes
- Ensure **grayscale images** of similar resolution for best results.
- Adjust `MATCH_THRESHOLD` in `iris_recognition.py` to fine-tune accuracy.
