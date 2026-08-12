"""

---------
Multi-Sensor Fusion Engine.

Combines YOLOv5m detection confidence with DHT11 / MQ-2 / IR flame sensor
readings to classify the current scene into one of three danger levels:
Safe, Moderate, or High Danger — following the fusion logic described in
the project report (Chapter 4.3):

    Safe:      no strong YOLOv5 detection AND all sensors within normal range
    Moderate:  partial detection OR exactly one sensor crosses its threshold
    High:      strong YOLOv5 detection AND multiple sensors cross threshold

This reduces false positives from fog, shadows, or lighting changes (which
can fool the vision model alone) and false negatives from sensor noise
(which can fool the sensors alone).
"""

from dataclasses import dataclass, field

import config

SAFE = "Safe"
MODERATE = "Moderate"
HIGH_DANGER = "High Danger"

# YOLOv5 confidence bands
CONF_HIGH = 0.60   # strong detection
CONF_LOW = config.YOLO_CONF_THRESHOLD  # anything below this the detector wouldn't even report


@dataclass
class FusionResult:
    danger_level: str
    yolo_confidence: float
    sensors_exceeded: list = field(default_factory=list)
    reason: str = ""


class FusionEngine:
    def __init__(self,
                 temp_threshold: float = config.TEMP_THRESHOLD_C,
                 smoke_threshold: int = config.SMOKE_THRESHOLD_RAW):
        self.temp_threshold = temp_threshold
        self.smoke_threshold = smoke_threshold

    def _sensor_exceedances(self, sensor_values: dict):
        """Returns a list of sensor names that are currently reading abnormal."""
        exceeded = []

        temp = sensor_values.get("temperature")
        if temp is not None and temp >= self.temp_threshold:
            exceeded.append("temperature")

        smoke = sensor_values.get("smoke")
        if smoke is not None and smoke >= self.smoke_threshold:
            exceeded.append("smoke")

        if sensor_values.get("flame"):
            exceeded.append("flame")

        return exceeded

    def fuse(self, detections: list, sensor_values: dict) -> FusionResult:
        """
        detections: list of dicts from YOLOv5Detector.detect(), e.g.
            [{"label": "fire", "confidence": 0.69, "box": (...)}]
        sensor_values: dict from SensorModule.read_all()
        """
        yolo_conf = max((d["confidence"] for d in detections), default=0.0)
        exceeded = self._sensor_exceedances(sensor_values)

        if yolo_conf >= CONF_HIGH and len(exceeded) >= 2:
            return FusionResult(
                danger_level=HIGH_DANGER,
                yolo_confidence=yolo_conf,
                sensors_exceeded=exceeded,
                reason="Strong YOLOv5 detection confirmed by multiple sensors.",
            )

        if yolo_conf >= CONF_LOW or len(exceeded) == 1:
            return FusionResult(
                danger_level=MODERATE,
                yolo_confidence=yolo_conf,
                sensors_exceeded=exceeded,
                reason="Partial detection or a single sensor reading is abnormal.",
            )

        return FusionResult(
            danger_level=SAFE,
            yolo_confidence=yolo_conf,
            sensors_exceeded=exceeded,
            reason="No significant detection; all sensors within normal range.",
        )


if __name__ == "__main__":
    # Quick manual sanity check
    engine = FusionEngine()

    safe = engine.fuse([], {"temperature": 26, "smoke": 90, "flame": False})
    print(safe)

    moderate = engine.fuse(
        [{"label": "fire", "confidence": 0.50, "box": (0, 0, 10, 10)}],
        {"temperature": 26, "smoke": 90, "flame": False},
    )
    print(moderate)

    high = engine.fuse(
        [{"label": "fire", "confidence": 0.69, "box": (0, 0, 10, 10)}],
        {"temperature": 47, "smoke": 171, "flame": True},
    )
    print(high)
