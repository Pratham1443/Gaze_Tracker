import cv2
from pynput.mouse import Controller
import wx
import os



app = wx.App(False)
screen = wx.GetDisplaySize()
camera = (640, 480)
mouse = Controller()
cap = cv2.VideoCapture(0)
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_eye.xml')
eye_cascade = cv2.CascadeClassifier(haar_model)

while True:
  _, frame = cap.read()

  frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  eyes = eye_cascade.detectMultiScale(frame_hsv, 1.3, 10)
  coords = []
  for (x, y, w, h) in eyes:
    coords.append((x+w//2, y+h//2))
    cv2.rectangle(frame, (x, y),
                      (x + w, y + h), (0, 255, 0), 2)
  if len(coords)>=2:
    center = ((coords[0][0]+coords[1][0])//2, (coords[0][1]+coords[1][1])//2)

    cv2.circle(frame, center, 2, (255, 0, 0), 2)

    mouse.position = (screen[0] - center[0]*screen[0]/camera[0], center[1]*screen[1]/camera[1])
    # while mouse.position != (center[0]*screen[0]/camera[0], center[1]*screen[1]/camera[1]):
    #   pass
  
  
  cv2.imshow("frame", frame)
  k = cv2.waitKey(10)
  if k==ord('q'):
    break




