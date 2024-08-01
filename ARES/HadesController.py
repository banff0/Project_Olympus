import pyautogui

int_to_direction = {0:"w", 1:"a", 2:"s", 3:"d"}


def move_cursor(x_vel, y_vel):
    pyautogui.moveRel(x_vel, y_vel, duration = 0.01)

def basic_attack():
    pyautogui.click()

def cast():
    pyautogui.click(button="right")

def special():
    pyautogui.press('q')     

def dash():
    pyautogui.press('space')

def move(direction):
    if direction > 3:
        return
    pyautogui.press(int_to_direction[direction])

def attack(action):
    if action > 2:
        return
    int_to_attack = {0:basic_attack, 1:special, 2:cast}
    int_to_attack[action]()

     