import keyboard
import redis 

USER_READY_KEY = ["sai2::optitrack::user_ready", "sai2::optitrack::user_1_ready", "sai2::optitrack::user_2_ready"]

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def on_key_event(keyboard_event):
    # print(f"Key {keyboard_event.name} {keyboard_event.event_type}")
    if (keyboard_event.name == "r" and keyboard_event.event_type == "up"):
        print("Robot Reset")
#        r.set("sai2::sim::reset", 1)
        # reset user ready key 
 #       for key in USER_READY_KEY:
  #          r.set(key, 0)
    elif (keyboard_event.name == "1" and keyboard_event.event_type == "up"):
        print("User 1 Ready")
        r.set("sai2::optitrack::user_1_ready", 1)
    elif (keyboard_event.name == "2" and keyboard_event.event_type == "up"):
        print("User 2 Ready")
        r.set("sai2::optitrack::user_2_ready", 1)
    elif (keyboard_event.name == "0" and keyboard_event.event_type == "up"):
        print("Single User Ready")
        r.set("sai2::optitrack::user_ready", 1)

# Set up a listener for all key events
keyboard.hook(on_key_event)

# Block the program from exiting immediately
keyboard.wait("q")
