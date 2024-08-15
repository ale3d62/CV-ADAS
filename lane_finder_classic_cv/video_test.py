import cv2
from lane_finder import findLane
from car_finder import CarDetector
from frame_visualizer import FrameVisualizer
from auxFunctions import *
from time import time
import sys


#for screen capture
import numpy as np
from mss import mss

#YOLO
from ultralytics import YOLO


#----------PARAMETERS-----------------------------------------------
modelName = "yolov8n.pt"
modelPath = "../models/" 

#Predictions below this confidence value are skipped (range: [0-1])
yoloConfidenceThreshold = 0.2 

#Indexes of the only yolo object classes to consider
acceptedClasses = set([2, 3, 4, 6, 7])
showLines = True
showCars = True

#To scale the video down and make it faster
# number in the range (0-1]
resScaling = 0.5

#Source of the image to process
# - video: test videos at videoPath directory
# - screen: screen capture
# - camera: device camera
videoSource = "screen" 
videoPath = "../test_videos/"
screenCaptureW = 1920
screenCaptureH = 1080

#Maximum number of frames without updating lane
maxLAge = 1000

#CAMERA PARAMETERS
cameraId = 0
camParams = {
    "f": 2.5,
    "sensorPixelW": 0.008,
    "roadWidth": 3600
}

#VISUALIZATION
# - none: no visualization
# - screen: on screen
# - server: on web server 
visualizationMode = "screen"
#when selected mode is server:
serverParameters = {
    "ip": "0.0.0.0",
    "port": 5000
}

#ALGORITHM PARAMETERS
iouThresh = 0.5


#DEBUGGING
showTimes = True
enableOptimizations = True
#-------------------------------------------------------------


#Load yolo model
model = YOLO(modelPath + modelName)

#Load visualizer
frameVisualizer = FrameVisualizer(visualizationMode, serverParameters)

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
    carDetector = CarDetector(iouThresh, camParams, showCars)



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
                    frameVisualizer.showFrame(frame)
                    continue
            else:
                lastLFrame = iFrame
        

        #SCAN FOR CARS
        sty = time()
        carDetector.findCars(model, frame, acceptedClasses)
        totalTimeYolo += (time()-sty)*1000

        #If there are no cars, skip to next frame
        if(carDetector.nCars() == 0):
            frameVisualizer.showFrame(frame)
            continue
        

        #GET DISTANCE TO CAR
        carDetector.updateDist(frame.shape, bestLinePointsLeft, bestLinePointsRight)
        """
        for distance in distances:
            d = distance['distance']
            x1, y1, x2, y2 = distance['bbox']
            cv2.putText(frame, "Distance: {:6.2f}m".format(d), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(100, 100, 255))
        """
        #GET CAR SPEED
        cars = carDetector.getCars()
        for car in cars:
            if(car['old']):
                frameTime = car['new']['time'] - car['old']['time']

                distanceDiff = car['old']['distance'] - car['new']['distance']
                
                #get speed in m/ms and convert to km/h
                carSpeed = (distanceDiff/frameTime) * 3600
                print(frameTime, distanceDiff, carSpeed)
                #Display speed next to car
                x1, y1, x2, y2 = car['new']['bbox']
                cv2.putText(frame, "Rel Speed: {:6.2f}km/h".format(distanceDiff), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(100, 100, 255))


        #show new frame
        frameVisualizer.showFrame(frame)



    #Measure average time
    totalTime += (time()-st)*1000
    if(totalFrames>0):
        if(showTimes):
            print("[INFO] avg time car detection: "+ str(totalTimeYolo/totalFrames) + "ms")
            print("[INFO] avg time lane detection: "+ str(totalTimeLane/totalFrames) + "ms")
            print("[INFO] avg time total: "+ str(totalTime/totalFrames) + "ms")