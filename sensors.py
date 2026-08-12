"""
sensors.py
----------
Sensor Acquisition Module.

Reads the three environmental sensors used for multi-sensor fusion:
    - DHT11: temperature (°C) and humidity (%)
    - MQ-2: smoke/gas concentration (analog, read via an MCP3008 ADC over SPI,
            since the Raspberry Pi has no analog input pins)
    - IR flame sensor: digital flame presence (most modules are active-low:
            LOW = flame detected, HIGH = no flame)

On non-Pi hardware (e.g. a laptop used for development), all three sensors
fall back to a simulated mode that returns plausible "safe" readings with a
small amount of random jitter, so the rest of the pipeline can be exercised
without physical hardware.
"""

import random
import time

import config

try:
    import board
    import adafruit_dht
    DHT_AVAILABLE = True
except (ImportError, NotImplementedError):
    DHT_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False

try:
    import busio
    import digitalio
    from adafruit_mcp3xxx.mcp3008 import MCP3008
    from adafruit_mcp3xxx.analog_in import AnalogIn
    MCP3008_AVAILABLE = True
except (ImportError, NotImplementedError):
    MCP3008_AVAILABLE = False

HARDWARE_AVAILABLE = DHT_AVAILABLE and GPIO_AVAILABLE and MCP3008_AVAILABLE


class SensorModule:
    def __init__(self):
        self.simulated = not HARDWARE_AVAILABLE

        if self.simulated:
            print("[sensors] Hardware libraries not fully available — "
                  "running SensorModule in simulated mode.")
            return

        # DHT11
        self._dht = adafruit_dht.DHT11(getattr(board, f"D{config.PIN_DHT11}"))

        # IR flame sensor (digital input)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(config.PIN_IR_FLAME, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # MQ-2 via MCP3008 ADC over SPI
        spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
        cs = digitalio.DigitalInOut(getattr(board, f"D{config.SPI_DEVICE}"))
        mcp = MCP3008(spi, cs)
        channel_attr = getattr(__import__("adafruit_mcp3xxx.mcp3008", fromlist=["P" + str(config.MCP3008_CHANNEL_MQ2)]),
                                f"P{config.MCP3008_CHANNEL_MQ2}")
        self._mq2_channel = AnalogIn(mcp, channel_attr)

    def read_dht(self):
        """Returns (temperature_c, humidity_pct). Falls back to simulated values on read failure."""
        if self.simulated:
            return round(random.uniform(24.0, 30.0), 1), round(random.uniform(35.0, 55.0), 1)
        try:
            return float(self._dht.temperature), float(self._dht.humidity)
        except RuntimeError as e:
            # DHT11 sensors regularly drop reads; this is expected and not fatal.
            print(f"[sensors] DHT11 read skipped: {e}")
            return None, None

    def read_mq2(self):
        """Returns the MQ-2 smoke concentration as a raw ADC value (0-1023)."""
        if self.simulated:
            return random.randint(80, 130)
        # AnalogIn.value is 16-bit (0-65535); scale down to a 10-bit-equivalent range
        return int(self._mq2_channel.value / 64)

    def read_flame(self):
        """Returns True if flame is detected, False otherwise."""
        if self.simulated:
            return random.random() < 0.02  # rare simulated flame event
        # Most IR flame modules are active-low: LOW means flame detected
        return GPIO.input(config.PIN_IR_FLAME) == GPIO.LOW

    def read_all(self):
        """Convenience method: reads all three sensors and returns a dict."""
        temp, humidity = self.read_dht()
        smoke = self.read_mq2()
        flame = self.read_flame()
        return {
            "temperature": temp,
            "humidity": humidity,
            "smoke": smoke,
            "flame": flame,
            "reading_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def cleanup(self):
        if GPIO_AVAILABLE and not self.simulated:
            GPIO.cleanup()


if __name__ == "__main__":
    sensors = SensorModule()
    for _ in range(5):
        print(sensors.read_all())
        time.sleep(2)
    sensors.cleanup()

