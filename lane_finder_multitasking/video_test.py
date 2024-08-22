import cv2
from lane_car_finder import Detector
from frame_visualizer import FrameVisualizer
from auxFunctions import *
from time import time

#for screen capture
import numpy as np
from mss import mss

#YOLO
from ultralytics import YOLO


#----------PARAMETERS----------------
#Security distance estimation
userVehicleDeceleration = 9
otherVehiclesDeceleration = 11
reactionTime = 0.5

modelName = "v4n_lane_det.onnx"
modelPath = "../models/"

#Predictions below this confidence value are skipped (range: [0-1])
yoloConfidenceThreshold = 0.2 

#Select the predictions to show
showSettings = {
    "cars": True,
    "lanes": True
}

#Source of the image to process
# - video: test videos at videoPath directory
# - screen: screen capture
# - camera: device camera
videoSource = "screen" 
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
visualizationMode = "screen"
#when selected mode is server:
serverParameters = {
    "ip": "0.0.0.0",
    "port": 5000
}

#ALGORITHM PARAMETERS
yoloConfThresh = 0.3
yoloIouThresh = 0.5
trackingIouThresh = 0.5

#SPEED MEASURING
frameTimeThreshold = 500 #ms

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
        vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        inputVideos.pop(0)

    
    #initialize variables
    totalTime = 0
    totalTimeYolo = 0
    totalTimeLane = 0
    totalFrames = 0
    ret = True
    detector = Detector(yoloConfThresh, yoloIouThresh, trackingIouThresh, camParams, showSettings)



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


        #Resize to model size
        frame = cv2.resize(frame, (672,374), interpolation=cv2.INTER_NEAREST)

        

        #SCAN FOR CARS AND LINES
        sty = time()
        detector.detect(model, frame)
        totalTimeYolo += (time()-sty)*1000

        #If there are no cars, skip to next frame
        if(detector.nCars() == 0):
            frameVisualizer.showFrame(frame)
            continue
        

        #GET CAR SPEED
        cars = detector.getCars()
        for car in cars:
            if(car['old']):
                frameTime = car['new']['time'] - car['old']['time']
                
                if frameTime > frameTimeThreshold:
                    
                    if(not car['new']['distance'] or not car['old']['distance']):
                        break

                    distanceDiff = car['new']['distance'] - car['old']['distance']
                    
                    #get speed in m/ms and convert to km/h
                    carSpeed = (distanceDiff/frameTime) * 3600

                    car['new']['speed'] = carSpeed

                    #Update old car
                    car['old'] = {"distance": car['new']['distance'], "time": car['new']['time']}

                    #Get security distance
                    relVel = car['new']['speed']
                    secDist = relVel*reactionTime + pow(reactionTime, 2)/(2*(userVehicleDeceleration - otherVehiclesDeceleration))

                    if(secDist <= car['new']['distance']):
                        alert()

                if car['new']['speed']:
                    #Display speed next to car
                    x1, y1, x2, y2 = car['new']['bbox']
                    cv2.putText(frame, "{:6.2f}km/h".format(car['new']['speed']), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(255, 60, 255), lineType=cv2.LINE_AA)



        #show new frame
        frameVisualizer.showFrame(frame)



    #Measure average time
    totalTime += (time()-st)*1000
    if(totalFrames>0):
        if(showTimes):
            print("[INFO] avg time car detection: "+ str(totalTimeYolo/totalFrames) + "ms")
            print("[INFO] avg time lane detection: "+ str(totalTimeLane/totalFrames) + "ms")
            print("[INFO] avg time total: "+ str(totalTime/totalFrames) + "ms")