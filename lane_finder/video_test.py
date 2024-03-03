import cv2
import numpy as np
from lane_finder import findLane
from auxFunctions import *
from time import time
import warnings
warnings.filterwarnings('error')

for i in [23]: #7 10 23
  vid = cv2.VideoCapture('test_videos/test'+str(i)+'.mp4')
  totalTime = 0
  totalFrames = 0
  bestLinePointsLeft = None
  bestLinePointsRight = None
  while(vid.isOpened()):
    # Capture frame-by-frame
    ret, frame = vid.read()
    
    if ret == True:
      imgHeight, imgWidth, _ = frame.shape
      newImgHeight = int(imgHeight/2)

      st = time()
      newBestLinePointsLeft, newBestLinePointsRight = findLane(frame)
      totalTime += (time()-st)*1000
      totalFrames += 1
      #print("time: " + str((time()-st)*1000) + "ms")
      
      if(newBestLinePointsLeft[0]):
          bestLinePointsLeft = newBestLinePointsLeft
      if(newBestLinePointsRight[0]):
          bestLinePointsRight = newBestLinePointsRight

      if(bestLinePointsLeft):
          cv2.line(frame, (bestLinePointsLeft[1], newImgHeight), (bestLinePointsLeft[0], imgHeight), (0, 0, 255), 2)
      if(bestLinePointsRight):
          cv2.line(frame, (bestLinePointsRight[1], newImgHeight), (bestLinePointsRight[0], imgHeight), (0, 0, 255), 2)
          
      cv2.imshow('Frame',frame)
      
  
      # Press Q on keyboard to  exit
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
    # Break the loop
    else: 
      break
  if(totalFrames>0):
    print("avg time: "+ str(totalTime/totalFrames) + "ms")