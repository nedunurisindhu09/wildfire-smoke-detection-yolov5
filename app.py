"""
app.py
------
Flask Dashboard Module — the top-level orchestrator described in the
report's implementation flow (Chapter 4.4):

    1. Initialize hardware (camera, sensors, LEDs, buzzer)
    2. Load the trained YOLOv5m model
    3. Continuously: capture frame -> detect -> read sensors -> fuse ->
       dispatch alerts -> log to DB -> stream to dashboard

Run:
    python app.py
Then open http://<device-ip>:5000 in a browser.
"""

import time

import cv2
from flask import Flask, Response, jsonify, render_template

import config
import database
from alerts import AlertManager
from detect import YOLOv5Detector
from fusion import FusionEngine
from sensors import SensorModule

app = Flask(__name__)

database.init_db()
detector = YOLOv5Detector()
sensor_module = SensorModule()
fusion_engine = FusionEngine()
alert_manager = AlertManager()

# Shared state consumed by the dashboard's polling endpoint
latest_status = {
    "danger_level": "Safe",
    "yolo_confidence": 0.0,
    "sensors": {},
    "sensors_exceeded": [],
    "last_update": None,
}

database.log_system_event("System started.")


def generate_frames():
    cap_source = int(config.CAMERA_SOURCE) if config.CAMERA_SOURCE.isdigit() else config.CAMERA_SOURCE
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {config.CAMERA_SOURCE}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, detections = detector.detect(frame)
            sensor_values = sensor_module.read_all()
            result = fusion_engine.fuse(detections, sensor_values)

            # Persist this cycle
            database.log_sensor_reading(sensor_values)
            detection_id = database.log_detection(detections, result.danger_level)
            if result.danger_level != "Safe":
                database.log_alert(result.danger_level, detection_id)

            alert_manager.dispatch(result.danger_level, result.yolo_confidence, result.sensors_exceeded)

            latest_status["danger_level"] = result.danger_level
            latest_status["yolo_confidence"] = result.yolo_confidence
            latest_status["sensors"] = sensor_values
            latest_status["sensors_exceeded"] = result.sensors_exceeded
            latest_status["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")

            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
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


@app.route("/alerts")
def alerts():
    return jsonify(database.get_recent_alerts(limit=20))


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        alert_manager.cleanup()
        sensor_module.cleanup()
        database.log_system_event("System stopped.")
        
