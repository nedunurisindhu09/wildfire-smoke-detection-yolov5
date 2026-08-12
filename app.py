"""
app.py
------
Flask web app that streams live webcam video with YOLOv5 fire/smoke
detection overlaid, and triggers the buzzer/LED alert system when a
detection crosses the confidence threshold.

Usage:
    python app.py
    Then open http://127.0.0.1:5000 in your browser.
"""

import time

import cv2
from flask import Flask, Response, render_template, jsonify

from alert import AlertSystem
from detect import FireSmokeDetector

app = Flask(__name__)

WEIGHTS_PATH = "best.pt"
VIDEO_SOURCE = 0  # webcam index; change to a file path for a video file

detector = FireSmokeDetector(weights_path=WEIGHTS_PATH)
alert_system = AlertSystem(led_pin=17, buzzer_pin=27)

# Shared state for the dashboard's status panel
latest_status = {
    "detected": False,
    "labels": [],
    "last_update": None,
}


def generate_frames():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, detections = detector.infer(frame)

            latest_status["detected"] = bool(detections)
            latest_status["labels"] = [d["label"] for d in detections]
            latest_status["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")

            if detections:
                top_label = max(detections, key=lambda d: d["confidence"])["label"]
                alert_system.trigger(top_label)

            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(latest_status)


@app.teardown_appcontext
def cleanup(exception=None):
    pass  # alert_system.cleanup() is called on process exit, not per-request


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        alert_system.cleanup()
