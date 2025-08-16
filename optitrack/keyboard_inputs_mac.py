from pynput import keyboard
import redis

USER_READY_KEY = ["sai2::optitrack::user_ready", "sai2::optitrack::user_1_ready", "sai2::optitrack::user_2_ready"]

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def on_release(key):
    try:
        if key.char == 'r':
            timestamp = r.get("timestamp")
            print("     " + timestamp)
            # print("Robot Reset")
            # for key in USER_READY_KEY:
            #     r.set(key, 0)
        elif key.char == '1':
            print("User 1 Ready")
            r.set("sai2::optitrack::user_1_ready", 1)
        elif key.char == '2':
            print("User 2 Ready")
            r.set("sai2::optitrack::user_2_ready", 1)
        elif key.char == '0':
            print("Single User Ready")
            r.set("sai2::optitrack::user_ready", 1)
        elif key.char == 'q':
            print("Quit")
            return False  # Stops the listener
    except AttributeError:
        pass  # Handle special keys like shift, ctrl, etc.

with keyboard.Listener(on_release=on_release) as listener:
    listener.join()
