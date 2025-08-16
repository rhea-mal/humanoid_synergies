import redis
import RPi.GPIO as GPIO
import time

# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# GPIO Pin setup
RED_PIN = 17
GREEN_PIN = 27
BLUE_PIN = 22
WHITE_PIN = 23  # If using RGBW LED strip

GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN, GPIO.OUT)
GPIO.setup(WHITE_PIN, GPIO.OUT)  # Only if using RGBW

# Set up PWM channels
red_pwm = GPIO.PWM(RED_PIN, 1000)  # 1kHz frequency
green_pwm = GPIO.PWM(GREEN_PIN, 1000)
blue_pwm = GPIO.PWM(BLUE_PIN, 1000)
white_pwm = GPIO.PWM(WHITE_PIN, 1000)  # Only if using RGBW

# Start PWM with 0% duty cycle (off)
red_pwm.start(0)
green_pwm.start(0)
blue_pwm.start(0)
white_pwm.start(0)  # Only if using RGBW

# Redis key for RGB values
RGB_KEY = "RGB_COLOR"

def set_color(r, g, b, w=0):
    red_pwm.ChangeDutyCycle(r)
    green_pwm.ChangeDutyCycle(g)
    blue_pwm.ChangeDutyCycle(b)
    white_pwm.ChangeDutyCycle(w)  # Only if using RGBW

def read_and_set_color():
    while True:
        # Read RGB value from Redis
        rgb_value = redis_client.get(RGB_KEY)
        if rgb_value:
            try:
                # Split the RGB value (expected format: "r,g,b" or "r,g,b,w")
                rgb_components = rgb_value.decode('utf-8').split(',')
                r = int(rgb_components[0])
                g = int(rgb_components[1])
                b = int(rgb_components[2])
                w = int(rgb_components[3]) if len(rgb_components) > 3 else 0  # For RGBW

                # Set the color on the LED strip
                set_color(r, g, b, w)

                print(f"Set RGBW to: {r}, {g}, {b}, {w}")

            except ValueError:
                print("Invalid RGB format in Redis.")
        else:
            print("No RGB value found in Redis.")
        
        time.sleep(0.1)  # Polling interval

if __name__ == "__main__":
    try:
        read_and_set_color()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        red_pwm.stop()
        green_pwm.stop()
        blue_pwm.stop()
        white_pwm.stop()  # Only if using RGBW
        GPIO.cleanup()
