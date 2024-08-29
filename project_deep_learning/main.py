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

import sys

#----------PARAMETERS----------------
#Source of the image to process
# - video: test videos at videoPath directory
# - screen: screen capture
# - camera: device camera
videoSource = "screen" 
videoPath = "../test_videos/"
inputVideos = [f"test{i}.mp4" for i in range(8,27)] 
screenCaptureW = 1920
screenCaptureH = 1080

#Detection model
modelName = "v4_2_tasks.onnx"
modelPath = "../models/"

#ALGORITHM PARAMETERS
yoloConfThresh = 0.3
yoloIouThresh = 0.5
trackingIouThresh = 0.5
bBoxMinSize = 0.025 #bboxes with a size smaller than 2.5% of the image are ignored


#CAMERA PARAMETERS
cameraId = 0
camParams = {
    "fReal": 4.8,  #mm
    "fEq": 26.8,     #mm
    "roadWidth": 3.5 #m
}

#SPEED MEASURING
frameTimeThreshold = 1000 #ms

#Security distance estimation
vehiclesDeceleration = 11 #m/s^2
slowUserBrake = False #user's vehicle has no abs or brakes slower than others
reactionTime = 0.5 #sec
reactionAproxVel = 100 #km/h
vehicleBonnetSize = 1.5 #m


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
defaultBboxColor = (0, 255, 0)

#Select the predictions to show
showSettings = {
    "cars": True,
    "lanes": True
}
showDistances = False #Takes priority over showSpeed
showSpeed = True 


#DEBUGGING
printTimes = False
filterCarInLane = False
printDistances = True
#------------------------------------

#Optimize parameters
if visualizationMode == "none":
    showSettings = {
        "cars": False,
        "lanes": False
    }
    showDistances = False
    showSpeed = False

#Load yolo model
model = YOLO(modelPath + modelName)

#model warmup
model.predict(source=np.zeros((374,672, 3), dtype=np.uint8), imgsz=(374,672), device="cpu")
print("Model loaded")

#Load visualizer
frameVisualizer = FrameVisualizer(visualizationMode, serverParameters)

#for screen capture
if(videoSource == "screen"):
    bounding_box = {'top': 0, 'left': 0, 'width': screenCaptureW, 'height': screenCaptureH}
    sct = mss()


if(videoSource == "camera"):
    vid = cv2.VideoCapture(cameraId)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)


#MAIN LOOP
print("Starting predictions")
while(canProcessVideo(inputVideos, videoSource)):
    
    #Get input video
    if(videoSource == "video"):
        vid = cv2.VideoCapture(videoPath+inputVideos[0])
        vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        inputVideos.pop(0)

    
    #initialize variables
    totalTime = 0
    totalTimeYolo = 0
    totalTimeLane = 0
    totalFrames = 0
    ret = True
    detector = Detector(yoloConfThresh, yoloIouThresh, trackingIouThresh, bBoxMinSize, camParams, showSettings, filterCarInLane, defaultBboxColor)



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

            if(showDistances and car['new']['distance']):
                x1, y1, x2, y2 = car['new']['bbox']
                cv2.putText(frame, "{:6.2f}m".format(car['new']['distance']), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(255, 60, 255), lineType=cv2.LINE_AA)

            if(car['old']):
                frameTime = car['new']['time'] - car['old']['time']
                
                if frameTime > frameTimeThreshold:
                    
                    if(not car['new']['distance'] or not car['old']['distance']):
                        continue

                    distanceDiff = car['new']['distance'] - car['old']['distance']
                    
                    #get speed in m/ms and convert to m/s
                    carSpeed = (distanceDiff/frameTime) * 1000

                    car['new']['speed'] = carSpeed

                    #Update old car
                    car['old'] = {"distance": car['new']['distance'], "time": car['new']['time']}


                    #GET SECURITY DISTANCE
                    relVel = car['new']['speed']

                    secDist = -relVel / (2*vehiclesDeceleration)

                    #if user's car brakes slower, add extra distance
                    if(slowUserBrake):
                        secDist *= 1.5

                    secDist  += (reactionAproxVel/3.6) * reactionTime


                    if(printDistances and not printTimes):
                        printMsg = f"\rRelVel: " + "{:.2f}".format(relVel) + "m/s Distance: " + "{:.2f}".format(car['new']['distance'] - vehicleBonnetSize) + "m secDist: "+"{:.2f}".format(secDist) + "m "
                        sys.stdout.write(printMsg)
                        sys.stdout.flush()

                    if(car['new']['distance'] - vehicleBonnetSize <= secDist):
                        car['color'] = (0, 0, 255) #Set bounding box color to red
                        #alert()
                    else:
                        car['color'] = defaultBboxColor
                if(not showDistances and showSpeed and car['new']['speed'] != None):
                    #Display speed next to car
                    x1, y1, x2, y2 = car['new']['bbox']
                    speedKmH = car['new']['speed'] * 3.6 #m/s to km/h
                    cv2.putText(frame, "{:6.2f}km/h".format(speedKmH), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(255, 60, 255), lineType=cv2.LINE_AA)

                



        #show new frame
        frameVisualizer.showFrame(frame)



        #Measure average time
        totalTime += (time()-st)*1000
        if(totalFrames>0):
            if(printTimes):
                printMsg = f"\r[INFO] avg time: "+"{:.2f}".format(totalTimeYolo/totalFrames)+"ms "
                sys.stdout.write(printMsg)
                sys.stdout.flush()