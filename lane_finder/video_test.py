import cv2
import numpy as np
from lane_finder import findLane
from auxFunctions import *
from time import time
#import warnings
#from cv2 import cuda
#warnings.filterwarnings('error')

from ultralytics import YOLO


model = YOLO("best.pt")

#CAMERA PARAMETERS
f = 26
sensorPixelW = 0.008
roadWidth = 3600


for i in range(8,23):
  vid = cv2.VideoCapture('test_videos/test'+str(i)+'.mp4')
  totalTime = 0
  totalFrames = 0
  bestLinePointsLeft = (None, None)
  bestLinePointsRight = (None, None)
  while(vid.isOpened()):
    # Capture frame-by-frame
    ret, frame = vid.read()
    
    if ret == True:

      imgHeight, imgWidth, _ = frame.shape
      halfImgHeight = int(imgHeight/2)
      
      st = time()
      newBestLinePointsLeft, newBestLinePointsRight = findLane(frame)
      totalTime += (time()-st)*1000
      totalFrames += 1

      if(newBestLinePointsLeft[0]):
          bestLinePointsLeft = newBestLinePointsLeft
      if(newBestLinePointsRight[0]):
          bestLinePointsRight = newBestLinePointsRight

      if(bestLinePointsLeft[0] and bestLinePointsLeft[1]):
          cv2.line(frame, (bestLinePointsLeft[1], halfImgHeight), (bestLinePointsLeft[0], imgHeight), (0, 0, 255), 2)
      if(bestLinePointsRight[0] and bestLinePointsRight[1]):
          cv2.line(frame, (bestLinePointsRight[1], halfImgHeight), (bestLinePointsRight[0], imgHeight), (0, 0, 255), 2)


      #YOLO 
      lx1, lx2 = bestLinePointsLeft
      rx1, rx2 = bestLinePointsRight
      if(not lx1 or not lx2 or not rx1 or not rx2):
          cv2.imshow('Frame',frame)
          continue
      
      
      results = model(frame, verbose=False)[0]
      for result in results.boxes.data.tolist():
          x1, y1, x2, y2, score, _ = result

          if score > 0.2:
              cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

              #get lines equations
              lm = halfImgHeight/(lx2-lx1)
              
              lb = -lm * lx1
              rm = halfImgHeight/(rx2-rx1)
              rb = -rm * rx1
              
              lx3 = ((imgHeight-y2)-lb)/lm
              rx3 = ((imgHeight-y2)-rb)/rm

              #if car is in lane
              if(y2 < imgHeight and y2 > halfImgHeight and ((lx3 < x1 and rx3 > x1) or (lx3 < x2 and rx3 > x2))):
                  d = (f*roadWidth*imgWidth)/((rx3-lx3)*(sensorPixelW*imgWidth))
                  cv2.line(frame, (int(lx3), int(y2)), (int(rx3), int(y2)), (255, 50, 50), 2)
                  print((rx3-lx3))
                  print("distance: "+str(d))

      cv2.imshow('Frame',frame)
      
  
      # Press Q on keyboard to  exit
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
    # Break the loop
    else: 
      break
  if(totalFrames>0):
    print("avg time: "+ str(totalTime/totalFrames) + "ms")