"""
detect.py
---------
YOLOv5 Detection Module.

Loads the trained YOLOv5m fire/smoke model (best.pt, trained on the
project's custom 10,835-image annotated dataset) and runs inference on a
live frame, video file, or single image. Returns bounding boxes and
confidence scores that feed into the Fusion Engine (see fusion.py).

Usage (CLI, no sensors/dashboard — for quick visual testing only):
    python detect.py --weights best.pt --source 0
"""

import argparse
import time

import cv2
import torch

import config

CLASS_NAMES = ["fire", "smoke"]
BOX_COLORS = {"fire": (0, 0, 255), "smoke": (0, 165, 255)}  # BGR: red, orange


class YOLOv5Detector:
    def __init__(self, weights_path: str = None, conf_thresh: float = None, device: str = ""):
        """Loads the YOLOv5m model via torch.hub ('ultralytics/yolov5', 'custom').
        The repo is cached locally by torch.hub after the first run."""
        self.conf_thresh = conf_thresh or config.YOLO_CONF_THRESHOLD
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom",
            path=weights_path or config.YOLO_WEIGHTS_PATH,
            force_reload=False,
        )
        self.model.conf = self.conf_thresh
        if device:
            self.model.to(device)

    def detect(self, frame):
        """Runs inference on a single BGR frame (numpy array).
        Returns: (annotated_frame, detections) where detections is a list of
        dicts: {label, confidence, box: (x1,y1,x2,y2)}
        """
        results = self.model(frame[:, :, ::-1])  # BGR -> RGB
        detections = []

        for *box, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, box)
            label = self.model.names[int(cls)]
            detections.append({
                "label": label,
                "confidence": round(float(conf), 3),
                "box": (x1, y1, x2, y2),
            })
            color = BOX_COLORS.get(label, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame, detections

    @staticmethod
    def max_confidence(detections):
        """Highest confidence score across all detections in this frame (0 if none)."""
        return max((d["confidence"] for d in detections), default=0.0)


def run_cli(weights: str, source: str):
    """Standalone visual test loop — no sensors, no fusion, no alerts."""
    detector = YOLOv5Detector(weights_path=weights)
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, detections = detector.detect(frame)
        if detections:
            labels = ", ".join(f"{d['label']} ({d['confidence']:.2f})" for d in detections)
            print(f"[detect] {time.strftime('%H:%M:%S')} — {labels}")

        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLOv5m Fire/Smoke Detection (visual test only)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv5m fire/smoke detection (visual test).")
    parser.add_argument("--weights", type=str, default=config.YOLO_WEIGHTS_PATH)
    parser.add_argument("--source", type=str, default=config.CAMERA_SOURCE)
    args = parser.parse_args()
    run_cli(args.weights, args.source)
