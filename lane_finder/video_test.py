import cv2
from lane_finder import findLane
from car_finder import findCars, findCarsPartial
from distance_calculator import getDistances
from auxFunctions import *
from time import time
import sys

#for screen capture
import numpy as np
from mss import mss

#YOLO
from ultralytics import YOLO


#----------PARAMETERS----------------
modelName = "yolov8n.pt"
#Predictions below this confidence value are skipped (range: [0-1])
yoloConfidenceThreshold = 0.2 
#Indexes of the only yolo object classes to consider
acceptedClasses = set([2, 3, 4, 6, 7])
showLines = True
#To scale the video down and make it faster
# number in the range (0-1]
resScaling = 0.5
searchRegion = 0.5
#Source of the image to process
# - video: test videos at test_videos directory
# - screen: screen capture
video_source = "video" 
maxLAge = 20
maxYAge = 5
#CAMERA PARAMETERS
f = 2.5
sensorPixelW = 0.008
roadWidth = 3600
#------------------------------------


#Load yolo model
model = YOLO(modelName)

#for screen capture
bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
sct = mss()

#indexes of the videos to use as input
inputVideos = [*range(8,27)] 



#MAIN LOOP
while(canProcessVideo(inputVideos, video_source)):
    
    #Get input video
    if(video_source == "video"):
        vid = cv2.VideoCapture('test_videos/test'+str(inputVideos[0])+'.mp4')
        inputVideos.pop(0)
    
    #initialize variables
    totalTime = 0
    totalTimeYolo = 0
    totalTimeLane = 0
    totalFrames = 0
    bestLinePointsLeft = (None, None)
    bestLinePointsRight = (None, None)
    ret = True
    iFrame = 0
    lastLFrame = -sys.maxsize
    lastYFrame = -sys.maxsize
    bBoxes = []

    #start timer
    st = time()

    while(ret):

        #Get frame
        if(video_source == "video"):
            ret, frame = vid.read()
        else:  
            sct_img = sct.grab(bounding_box)
            frame = np.array(sct_img)
            frame = np.delete(frame, 3, axis=2)
        

        if ret == False:
            break
        

        totalFrames += 1
        iFrame += 1


        #Apply res scaling
        if(resScaling != 1):
            frame = cv2.resize(frame, (0,0), fx = resScaling, fy = resScaling, interpolation=cv2.INTER_NEAREST)


                
        #SCAN FOR LINES
        stl = time()
        frame, bestLinePointsLeft, bestLinePointsRight, linesUpdated = findLane(frame, bestLinePointsLeft, bestLinePointsRight, showLines)
        totalTimeLane += (time()-stl)*1000
        
        if(not linesUpdated):
            if(iFrame - lastLFrame > maxLAge):
                bestLinePointsRight = (None, None)
                bestLinePointsLeft = (None, None)
                showFrame(frame)
                continue
        else:
            lastLFrame = iFrame


        #SCAN FOR CARS
        sty = time()
        if(iFrame - lastYFrame > maxYAge):
            bBoxes = findCars(model, frame, acceptedClasses, True)
            
            if(len(bBoxes) > 0):
                lastYFrame = iFrame
        else:
            bBoxes = findCarsPartial(model, frame, acceptedClasses, bBoxes, searchRegion)
            #bBoxes = findCars(model, frame, acceptedClasses)
        totalTimeYolo += (time()-sty)*1000

        #If there are no cars, skip to next frame
        if(len(bBoxes) == 0):
            showFrame(frame)
            continue





        #GET DISTANCE TO CAR
        distances = getDistances(frame, bBoxes, bestLinePointsLeft, bestLinePointsRight, roadWidth, sensorPixelW, f)
        for distance in distances:
            d, x1, y1, x2, = distance
            cv2.putText(frame, "Distance: {:6.2f}m".format(d), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(100, 100, 255))


        #show new frame
        showFrame(frame)



    #Measure average time
    totalTime += (time()-st)*1000
    if(totalFrames>0):
        pass
        print("avg time yolo: "+ str(totalTimeYolo/totalFrames) + "ms")
        print("avg time lane: "+ str(totalTimeLane/totalFrames) + "ms")
        print("avg time: "+ str(totalTime/totalFrames) + "ms")