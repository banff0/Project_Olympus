import mss
import numpy as np
import cv2
# import easyocr
import time

def capture_screen(monitor):
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        # Convert the screenshot to a numpy array
        img = np.array(screenshot)
        # Convert to BGR format (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    
monitor = {"top": 0, "left": 0, "width": 960*2, "height": 1080}
result = cv2.VideoWriter('filename.mp4',  
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         30, [960*2, 1080]) 

template = cv2.imread('boon.png')

while True:
    t0 = time.time()
    # Capture the screen
    screen_img = capture_screen(monitor)
    result.write(screen_img) 

    res = cv2.matchTemplate(screen_img,template,cv2.TM_CCOEFF_NORMED)
    print(f"max:{np.max(res)}")
    if np.max(res) > 0.7:
        print("COIN!!")

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(time.time() - t0)

result.release()

cv2.destroyAllWindows()

# video = cv2.VideoCapture("filename.mp4")

# while(True): 
#     ret, frame = video.read() 
  
#     if ret == True:  
#         # Display the frame 
#         # saved in the file 
#         frame = cv2.inRange(frame, (250, 250, 250), (255, 255, 255))
#         cv2.imshow('Frame', frame) 
  
#         # Press S on keyboard  
#         # to stop the process 
#         if cv2.waitKey(1) & 0xFF == ord('s'): 
#             break
#         time.sleep(0.01)
  
#     # Break the loop 
#     else: 
#         break
  
# # When everything done, release  
# # the video capture and video  
# # write objects 
# video.release() 
    
# # Closes all the frames 
# cv2.destroyAllWindows() 

