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
modelName = "yolov8n.pt"
modelPath = "../models/" 

#ALGORITHM PARAMETERS
yoloConfThresh = 0.3
yoloIouThresh = 0.5
trackingIouThresh = 0.5
bBoxMinSize = 0.025 #bboxes with a size smaller than 2.5% of the image are ignored

#Indexes of the only yolo object classes to consider
acceptedClasses = set([2, 3, 4, 6, 7])

#To scale the video down and make it faster
# number in the range (0-1]
resScaling = 0.5

enableOptimizations = True

#Maximum number of frames without updating lane
maxLAge = 1000

#CAMERA PARAMETERS
cameraId = 0
camParams = {
    "fReal": 4.8,  #mm
    "fEq": 26.8,     #mm
    "roadWidth": 3.5 #m
}

#SPEED MEASURING
frameTimeThreshold = 500 #ms

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

showLines = True
showCars = True
showCarId = False
showDistances = False #Takes priority over showSpeed
showSpeed = True 

#DEBUGGING
printTimes = True
filterCarInLane = False
printDistances = True
#-------------------------------------------------------------

#Optimize parameters
if visualizationMode == "none":
    showLines = False
    showCars = False
    showDistances = False
    showSpeed = False


#Load yolo model
model = YOLO(modelPath + modelName)

#model warmup
model.predict(source=np.zeros((640,640, 3), dtype=np.uint8), imgsz=(640,640))
print("Model loaded")

#Load visualizer
frameVisualizer = FrameVisualizer(visualizationMode, serverParameters)

#for screen capture
if(videoSource == "screen"):
    bounding_box = {'top': 0, 'left': 0, 'width': screenCaptureW, 'height': screenCaptureH}
    sct = mss()

if(videoSource == "camera"):
    vid = cv2.VideoCapture(cameraId)


#MAIN LOOP
print("Starting predictions")
while(canProcessVideo(inputVideos, videoSource)):
    
    #Get input video
    if(videoSource == "video"):
        vid = cv2.VideoCapture(videoPath+inputVideos[0])
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
    carDetector = CarDetector(yoloConfThresh, yoloIouThresh, trackingIouThresh, bBoxMinSize, camParams, showCars, showCarId, filterCarInLane, defaultBboxColor)



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
        if enableOptimizations:
            if(carDetector.nCars() == 0):
                frameVisualizer.showFrame(frame)
                continue
        

        #GET DISTANCE TO CAR
        carDetector.updateDist(frame.shape, bestLinePointsLeft, bestLinePointsRight)

        #GET CAR SPEED AND SECURITY ESTIMATION
        cars = carDetector.getCars()
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
                printMsg = f"\r[INFO] avg times: car detection: "+"{:.2f}".format(totalTimeYolo/totalFrames)+"ms | lane detection: "+"{:.2f}".format(totalTimeLane/totalFrames)+"ms "
                sys.stdout.write(printMsg)
                sys.stdout.flush()
