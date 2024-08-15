import cv2
from car_finder_onnx import CarDetector
from frame_visualizer import FrameVisualizer
from auxFunctions import *
from time import time
import sys
import torch

#for screen capture
import numpy as np
from mss import mss

#YOLO
from ultralytics import YOLO


#----------PARAMETERS----------------
modelName = "v4n_lane_det.onnx"
modelPath = "../models/"

#Predictions below this confidence value are skipped (range: [0-1])
yoloConfidenceThreshold = 0.2 

#Select the predictions to show
showSettings = {
    "cars": False,
    "lanes": False
}

#Source of the image to process
# - video: test videos at videoPath directory
# - screen: screen capture
# - camera: device camera
videoSource = "video" 
videoPath = "../test_videos/"
screenCaptureW = 1920
screenCaptureH = 1080

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
visualizationMode = "server"
#when selected mode is server:
serverParameters = {
    "ip": "0.0.0.0",
    "port": 5000
}

#ALGORITHM PARAMETERS
yoloConfThresh = 0.3
yoloIouThresh = 0.5
trackIouThresh = 0.5


#DEBUGGING
showTimes = True
#------------------------------------


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
    ret = True
    iFrame = 0
    lastLFrame = -sys.maxsize #-INF
    lastYFrame = -sys.maxsize #-INF
    carDetector = CarDetector(yoloConfThresh, yoloIouThresh, trackIouThresh, camParams, showSettings)



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


        #Resize to model size
        frame = cv2.resize(frame, (672,374), interpolation=cv2.INTER_NEAREST)

        

        #SCAN FOR CARS AND LINES
        sty = time()
        carDetector.findCars(model, frame)
        totalTimeYolo += (time()-sty)*1000

        #If there are no cars, skip to next frame
        if(carDetector.nCars() == 0):
            frameVisualizer.showFrame(frame)
            continue
        

        #GET DISTANCE TO CAR
        """
        distances = carDetector.updateDist(frame.shape, bestLinePointsLeft, bestLinePointsRight)
        for distance in distances:
            d = distance['distance']
            x1, y1, x2, y2 = distance['bbox']
            cv2.putText(frame, "Distance: {:6.2f}m".format(d), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(100, 100, 255))
        """

        #show new frame
        frameVisualizer.showFrame(frame)



    #Measure average time
    totalTime += (time()-st)*1000
    if(totalFrames>0):
        if(showTimes):
            print("[INFO] avg time car detection: "+ str(totalTimeYolo/totalFrames) + "ms")
            print("[INFO] avg time lane detection: "+ str(totalTimeLane/totalFrames) + "ms")
            print("[INFO] avg time total: "+ str(totalTime/totalFrames) + "ms")