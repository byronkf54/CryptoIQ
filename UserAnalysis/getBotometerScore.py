from pynput.keyboard import Key, Controller
import time

keyboard = Controller()

time.sleep(5)
with open("user_ids.txt", "r") as f:
    for userid in f:
        print("typing: " + userid)
        keyboard.type(userid)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        time.sleep(2)
        keyboard.press(Key.ctrl)
        keyboard.press("a")
        keyboard.release("a")
        keyboard.release(Key.ctrl)
        keyboard.press(Key.delete)
        keyboard.release(Key.delete)
        time.sleep(0.5)
