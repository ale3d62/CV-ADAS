import cv2
from lane_finder import findLane
from car_finder import CarDetector
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

#Source of the image to process
# - video: test videos at videoPath directory
# - screen: screen capture
# - camera: device camera
videoSource = "video" 
videoPath = "../test_videos/"
screenCaptureW = 1920
screenCaptureH = 1080

#Maximum number of frames without updating lane
maxLAge = 10

#CAMERA PARAMETERS
cameraId = 0
camParams = {
    "f": 2.5,
    "sensorPixelW": 0.008,
    "roadWidth": 3600
}

#Algorithm parameters
iouThresh = 0.5


#DEBUGGING
showTimes = True
enableOptimizations = True
#------------------------------------


#Load yolo model
model = YOLO(modelName)


#for screen capture
bounding_box = {'top': 0, 'left': 0, 'width': screenCaptureW, 'height': screenCaptureH}
sct = mss()

#indexes of the videos to use as input
inputVideos = [*range(8,27)] 

if(videoSource == "camera"):
    vid = cv2.VideoCapture(cameraId)


#MAIN LOOP
while(canProcessVideo(inputVideos, videoSource)):
    
    #Get input video
    if(videoSource == "video"):
        vid = cv2.VideoCapture(videoPath+'test'+str(inputVideos[0])+'.mp4')
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
    lastLFrame = -sys.maxsize #-INF
    lastYFrame = -sys.maxsize #-INF
    carDetector = CarDetector(iouThresh, camParams)



    #start timer
    st = time()

    while(ret):

        #Get frame
        if(videoSource == "video" or videoSource == "camera"):
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
        
        if enableOptimizations:
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
        carDetector.findCars(model, frame, acceptedClasses)
        totalTimeYolo += (time()-sty)*1000

        #If there are no cars, skip to next frame
        if(carDetector.nCars() == 0):
            showFrame(frame)
            continue
        

        #GET DISTANCE TO CAR
        distances = carDetector.updateDist(frame.shape, bestLinePointsLeft, bestLinePointsRight)
        for distance in distances:
            d = distance['distance']
            x1, y1, x2, y2 = distance['bbox']
            cv2.putText(frame, "Distance: {:6.2f}m".format(d), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(100, 100, 255))


        #show new frame
        showFrame(frame)



    #Measure average time
    totalTime += (time()-st)*1000
    if(totalFrames>0):
        if(showTimes):
            print("[INFO] avg time car detection: "+ str(totalTimeYolo/totalFrames) + "ms")
            print("[INFO] avg time lane detection: "+ str(totalTimeLane/totalFrames) + "ms")
            print("[INFO] avg time total: "+ str(totalTime/totalFrames) + "ms")