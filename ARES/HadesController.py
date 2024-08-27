import pyautogui
import mss
import numpy as np
import cv2
import os


int_to_direction = {0:"w", 1:"a", 2:"s", 3:"d"}


def move_cursor(x_vel, y_vel):
    pyautogui.moveRel(x_vel, y_vel, duration = 0.01)

def basic_attack():
    print("HERE")
    pyautogui.mouseDown()

def cast():
    pyautogui.click(button="right")

def special():
    pyautogui.press('q')     

def dash():
    pyautogui.press('space')

def move(direction):
    for v in int_to_direction.values():
        pyautogui.keyUp(v)
    if direction > 3:
        return
    pyautogui.keyDown(int_to_direction[direction])

def attack(action):
    print(action)
    pyautogui.mouseUp()
    if action > 2:
        return
    int_to_attack = {0:basic_attack, 1:special, 2:cast}
    int_to_attack[action]()

class ScreenHandler():
    def __init__(self) -> None:
        width, height = pyautogui.size()
        cap_width = width // 3
        cap_height = height // 3
        # print(width - 2*cap_width)
        # print(height - 2*cap_height)
        self.monitor = {"top": cap_width, "left": cap_height, "width": cap_width, "height": cap_height}
        # self.monitor = {"top": cap_width, "left": cap_height, "width": width - cap_width, "height": height - cap_height}
        self.mss_capture = mss.mss()
        self.template = None
        self.patern_match_threshold = 0.7

    def capture_screen(self):
        screenshot = self.mss_capture.grab(self.monitor)
        # Convert the screenshot to a numpy array
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, axis=2)

        # Convert to BGR format (OpenCV uses BGR by default)
        return img
    
    def set_room_type(self, room_type):
        template_path = os.path.join("artifact_imgs", f"{room_type}.png")
        self.template = cv2.imread(template_path)
        self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        # self.template = np.expand_dims(self.template, axis=2)
        # self.template = self.template.astype(np.uint8)
        cv2.imshow("HERE", self.template)
        cv2.waitKey(0)

        # closing all open windows
        cv2.destroyAllWindows()


    def get_end_of_room(self, screen):
        # screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        screen = np.squeeze(screen, axis=2)
        # print(f"SCREEN:{screen.shape}")

        # screen = screen.astype(np.uint8)
        # print(f"TEMPLATE{self.template.shape}")

        matches = cv2.matchTemplate(screen, self.template, cv2.TM_CCOEFF_NORMED)
        if np.max(matches) > self.patern_match_threshold:
            return True
        return False
    
if __name__ == "__main__":
    while True:
        move(0)