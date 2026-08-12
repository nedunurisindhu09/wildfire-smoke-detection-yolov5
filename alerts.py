"""
alerts.py
---------
Alert Manager.

Responds to a danger-level classification from the Fusion Engine
(fusion.SAFE / fusion.MODERATE / fusion.HIGH_DANGER) by:
    - Driving the LED indicators (green / yellow / red) and buzzer on the
      Raspberry Pi 5 GPIO pins.
    - Sending remote notifications through Email (SMTP), Telegram Bot API,
      and MQTT, per the danger level.

Falls back to a simulated / no-op mode for any channel whose hardware or
credentials aren't available, so the rest of the system can still run
(e.g. during development on a laptop, or if Telegram isn't configured).
"""

import smtplib
import ssl
import threading
import time
from email.mime.text import MIMEText

import config
from fusion import SAFE, MODERATE, HIGH_DANGER

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False

try:
    import telebot
    TELEBOT_AVAILABLE = True
except ImportError:
    TELEBOT_AVAILABLE = False

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


class AlertManager:
    def __init__(self):
        self._cooldown = config.ALERT_COOLDOWN_SECONDS
        self._last_notification_time = 0
        self._lock = threading.Lock()

        # --- Local indicators (LED + buzzer) ---
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            for pin in (config.PIN_LED_GREEN, config.PIN_LED_YELLOW, config.PIN_LED_RED, config.PIN_BUZZER):
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
        else:
            print("[alerts] RPi.GPIO not available — LED/buzzer output will be simulated (console only).")

        # --- Telegram ---
        self._telegram_bot = None
        if config.TELEGRAM_ENABLED and TELEBOT_AVAILABLE and config.TELEGRAM_BOT_TOKEN:
            self._telegram_bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
        elif config.TELEGRAM_ENABLED:
            print("[alerts] Telegram enabled in config but 'telebot' or bot token is missing — skipping.")

        # --- MQTT ---
        self._mqtt_client = None
        if config.MQTT_ENABLED and MQTT_AVAILABLE:
            self._mqtt_client = mqtt.Client()
            try:
                self._mqtt_client.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT, keepalive=30)
                self._mqtt_client.loop_start()
            except Exception as e:
                print(f"[alerts] MQTT connection failed: {e}")
                self._mqtt_client = None
        elif config.MQTT_ENABLED:
            print("[alerts] MQTT enabled in config but 'paho-mqtt' is not installed — skipping.")

    
    # Local indicators
    
    def _set_local_indicators(self, danger_level: str):
        if not GPIO_AVAILABLE:
            print(f"[alerts] (simulated) LEDs/buzzer set for level: {danger_level}")
            return

        GPIO.output(config.PIN_LED_GREEN, GPIO.LOW)
        GPIO.output(config.PIN_LED_YELLOW, GPIO.LOW)
        GPIO.output(config.PIN_LED_RED, GPIO.LOW)
        GPIO.output(config.PIN_BUZZER, GPIO.LOW)

        if danger_level == SAFE:
            GPIO.output(config.PIN_LED_GREEN, GPIO.HIGH)
        elif danger_level == MODERATE:
            GPIO.output(config.PIN_LED_YELLOW, GPIO.HIGH)
            self._pulse_buzzer(pattern="intermittent")
        elif danger_level == HIGH_DANGER:
            GPIO.output(config.PIN_LED_RED, GPIO.HIGH)
            self._pulse_buzzer(pattern="continuous")

    def _pulse_buzzer(self, pattern: str):
        if not GPIO_AVAILABLE:
            return

        def _run():
            if pattern == "continuous":
                GPIO.output(config.PIN_BUZZER, GPIO.HIGH)
                time.sleep(2)
                GPIO.output(config.PIN_BUZZER, GPIO.LOW)
            else:  # intermittent
                for _ in range(3):
                    GPIO.output(config.PIN_BUZZER, GPIO.HIGH)
                    time.sleep(0.3)
                    GPIO.output(config.PIN_BUZZER, GPIO.LOW)
                    time.sleep(0.3)

        threading.Thread(target=_run, daemon=True).start()

    
    # Remote notification channels

    def _send_email(self, subject: str, body: str):
        if not (config.EMAIL_ENABLED and config.EMAIL_SENDER and config.EMAIL_PASSWORD and config.EMAIL_RECIPIENTS):
            return
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = config.EMAIL_SENDER
            msg["To"] = ", ".join(config.EMAIL_RECIPIENTS)

            context = ssl.create_default_context()
            with smtplib.SMTP(config.EMAIL_SMTP_HOST, config.EMAIL_SMTP_PORT) as server:
                server.starttls(context=context)
                server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
                server.sendmail(config.EMAIL_SENDER, config.EMAIL_RECIPIENTS, msg.as_string())
        except Exception as e:
            print(f"[alerts] Email send failed: {e}")

    def _send_telegram(self, message: str):
        if not (self._telegram_bot and config.TELEGRAM_CHAT_ID):
            return
        try:
            self._telegram_bot.send_message(config.TELEGRAM_CHAT_ID, message)
        except Exception as e:
            print(f"[alerts] Telegram send failed: {e}")

    def _send_mqtt(self, payload: str):
        if not self._mqtt_client:
            return
        try:
            self._mqtt_client.publish(config.MQTT_TOPIC, payload)
        except Exception as e:
            print(f"[alerts] MQTT publish failed: {e}")

    
    # Public entry point
    
    def dispatch(self, danger_level: str, yolo_confidence: float, sensors_exceeded: list):
        """Call this once per fusion cycle. Handles local indicators immediately;
        remote notifications are rate-limited by ALERT_COOLDOWN_SECONDS and are
        skipped for Safe."""
        self._set_local_indicators(danger_level)

        if danger_level == SAFE:
            return

        now = time.time()
        with self._lock:
            if now - self._last_notification_time < self._cooldown:
                return
            self._last_notification_time = now

        message = (
            f"[Wildfire Alert] Danger level: {danger_level}\n"
            f"YOLOv5 confidence: {yolo_confidence:.2f}\n"
            f"Sensors exceeded: {', '.join(sensors_exceeded) if sensors_exceeded else 'none'}\n"
            f"Device: {config.DEVICE_NAME} ({config.DEVICE_LOCATION})\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        threading.Thread(target=self._send_email, args=(f"Wildfire Alert - {danger_level}", message), daemon=True).start()
        threading.Thread(target=self._send_telegram, args=(message,), daemon=True).start()
        threading.Thread(target=self._send_mqtt, args=(message,), daemon=True).start()

    def cleanup(self):
        if GPIO_AVAILABLE:
            for pin in (config.PIN_LED_GREEN, config.PIN_LED_YELLOW, config.PIN_LED_RED, config.PIN_BUZZER):
                GPIO.output(pin, GPIO.LOW)
            GPIO.cleanup()
        if self._mqtt_client:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()


if __name__ == "__main__":
    manager = AlertManager()
    manager.dispatch(HIGH_DANGER, 0.69, ["temperature", "smoke", "flame"])
    time.sleep(3)
    manager.cleanup()

