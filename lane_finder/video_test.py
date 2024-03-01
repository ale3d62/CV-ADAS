import cv2
import numpy as np
from lane_finder import findLane
from auxFunctions import *
from time import time

for i in range(7,23):
  vid = cv2.VideoCapture('test_videos/test'+str(i)+'.mp4')
  totalTime = 0
  totalFrames = 0
  while(vid.isOpened()):
    # Capture frame-by-frame
    ret, frame = vid.read()
    
    if ret == True:
      imgHeight, imgWidth, _ = frame.shape
  
      # Display the resulting frame
      st = time()
      frame = findLane(frame)
      totalTime += (time()-st)*1000
      totalFrames += 1
      #print("time: " + str((time()-st)*1000) + "ms")
      cv2.imshow('Frame',frame)
      
  
      # Press Q on keyboard to  exit
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
    # Break the loop
    else: 
      break
  if(totalFrames>0):
    print("avg time: "+ str(totalTime/totalFrames) + "ms")