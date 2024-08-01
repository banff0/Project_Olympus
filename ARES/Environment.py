import mss
from collections import deque
import numpy as np
import cv2


class HadesEnv:

    FULL_SCREEN = {"top": 0, "left": 0, "width": 960, "height": 1080}

    def __init__(self, input_nb_of_images, monitor = FULL_SCREEN) -> None:
        
        self.input_nb_of_images = input_nb_of_images
        self.monitor = monitor

        self.saved_images = deque(maxlen=self.input_nb_of_images)

        self.actions = [
            self._move_up,
            self._move_right,
            self._move_left,
            self._move_down,
            self._cast,
            self._attack,
            self._special,
            self._move_cursor # How
            ]

    #--- Initialization ---

    def get_inputs_numbers(self):
        return
    
    def outputs_numbers(self):
        return
    
    #--- Playing ---
    
    def get_metadata(self):
        return
    
    def update_images(self):

        with mss.mss() as sct:
            screenshot = sct.grab(self.monitor)
            # Convert the screenshot to a numpy array
            img = np.array(screenshot)
            # Convert to BGR format (OpenCV uses BGR by default)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self.saved_images.append(img)

            return self.saved_images
    
    def take_action(self, action : int):
        
        self.actions[action]()
        

    def _move_up(self):
        pass
    def _move_right(self):
        pass
    def _move_left(self):
        pass
    def _move_down(self):
        pass
    def _cast(self):
        pass
    def _attack(self):
        pass
    def _special(self):
        pass
    def _move_cursor(self): # How
        pass