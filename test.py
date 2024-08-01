import mss
import numpy as np
import easyocr

with mss.mss() as sct:
    screenshot = sct.grab({"top": 0, "left": 0, "width": 960, "height": 1080})
    # Convert the screenshot to a numpy array
    img = np.array(screenshot)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)


