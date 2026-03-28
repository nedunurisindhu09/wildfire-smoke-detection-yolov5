# 🔥 Real-Time Wildfire Smoke Detection using YOLOv5

## 📌 Overview

This project presents an AI-powered system for **real-time wildfire smoke and fire detection** using the YOLOv5 object detection model. It leverages computer vision and deep learning to identify early smoke patterns, enabling faster disaster response and mitigation.

---

## 🚀 Features

* 🔍 Real-time smoke & fire detection using YOLOv5
* 🎥 Live video streaming via Flask web app
* ⚡ Edge deployment support (Raspberry Pi)
* 🚨 Alert system (LED + buzzer)
* 📊 High accuracy with reduced false positives

---

## 🧠 Tech Stack

* **Python**
* **PyTorch**
* **YOLOv5 (Ultralytics)**
* **OpenCV**
* **Flask**
* **GPIO (Raspberry Pi)**

---

## 📂 Project Structure

```
wildfire-smoke-detection-yolov5/
│── app.py
│── detect.py
│── alert.py
│── best.pt
│── requirements.txt
│── templates/
│── static/
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📊 Results

* Achieved high precision in fire detection
* Moderate accuracy in smoke detection (challenging due to fog/cloud similarity)
* Real-time inference capability

---

## 📸 Demo

(Add screenshots or GIF here)

---

## 📌 Future Improvements

* 🔹 Integration with drones for large-area monitoring
* 🔹 Cloud-based dashboard
* 🔹 Upgrade to YOLOv8/YOLOv11
* 🔹 Multi-sensor fusion (temperature + gas sensors)

---

## 👩‍💻 Author

Nedunuri Sindhu
B.Tech AI & ML

---

## ⭐ Acknowledgements

* YOLOv5 by Ultralytics
* Kaggle datasets for fire/smoke images

