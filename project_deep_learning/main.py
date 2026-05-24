import cv2
import numpy as np
from detector_multi_task import DetectorMultiTask
from detector_single_task import DetectorSingleTask
from auxFunctions import *
from time import time
from ultralytics import YOLO
import sys
from lane_finder_cv import findLaneCV, getVanishingPoint, houghFiltering
from math import atan, cos

#-------------------------------DATASET SEQUENCES------------------------------
class SequenceConfig:
    def __init__(self, inputVideo, f, sensorW, isKITTI):
        self.inputVideo = inputVideo
        self.f = f
        self.sensorW = sensorW
        self.isKITTI = isKITTI

sequenceKITTI15 = SequenceConfig("video_KITTI_15.mp4", 4, 7.469, True)
sequenceWaymo10625 = SequenceConfig("video_waymo_10625.mp4", 38.54, 36, False)
sequenceWaymo10923 = SequenceConfig("video_waymo_10923.mp4", 38.69, 36, False)
sequenceWaymo11199 = SequenceConfig("video_waymo_11199.mp4", 38.99, 36, False)

#Choose here the dataset sequence to use
sequence = sequenceWaymo10625
#------------------------------------------------------------------------------


#---------------SYSTEM PARAMETERS----------------
videoPath = "../test_videos/"

#Detection model
multitaskModelName = "v4_2_tasks.onnx"  #Used for estimation method 1
detectionModelName = "yolov8n.pt"       #Used for estimation method 2
modelPath = "../models/"

#ALGORITHM PARAMETERS
yoloConfThresh = 0.3
yoloIouThresh = 0.5
trackingIouThresh = 0.5
bBoxMinSize = 0.025 #bboxes with a size smaller than 2.5% of the image are ignored

#ESTIMATION METHODS
# 1: Roadwidth and camera parameters
# 2: Reverse projection
estimationMethod = 1

roadWidth = 3.5 #m

#Use the estimated camera height to increase precision (for method 1)
heightCorrection = True

#SPEED MEASURING
frameTimeThreshold = 1000 #ms
distanceDiffThreshold = 1.5 #m

#Security distance estimation
vehiclesDeceleration = 11 #m/s^2
slowUserBrake = False #user's vehicle has no abs or brakes slower than others
reactionTime = 0.5 #sec
reactionAproxVel = 100 #km/h
vehicleBonnetSize = 1.5 #m


defaultBboxColor = (0, 255, 0)

#Select the predictions to show
showSettings = {
    "cars": True,
    "carId": False,
    "lanes": True
}
showDistances = True #Takes priority over showSpeed
showSpeed = True


#DEBUGGING
printTimes = False #Takes priority over printDistances
filterCarInLane = True
printDistances = False
#------------------------------------

#Load yolo model
if(estimationMethod == 1):
    modelName = multitaskModelName
elif(estimationMethod == 2):
    modelName = detectionModelName
else:
    print("[ERROR] Select a valid estimation method (1,2)")
    exit()

model = YOLO(modelPath + modelName)

#model warmup
model.predict(source=np.zeros((384,672, 3), dtype=np.uint8), imgsz=(384,672), device="cpu")
print("Model loaded")


#MAIN LOOP
print("Starting predictions")

#Get input video
vid = cv2.VideoCapture(videoPath+sequence.inputVideo)
vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)


#initialize variables
totalTime = 0
totalTimeYolo = 0
totalTimeLane = 0
totalFrames = 0
ret = True
#read first frame to get resolution
ret, frame = vid.read()
if(estimationMethod == 1):
    detector = DetectorMultiTask(yoloConfThresh, yoloIouThresh, trackingIouThresh, bBoxMinSize, sequence.f, sequence.sensorW, roadWidth, showSettings, filterCarInLane, heightCorrection, defaultBboxColor)
elif(estimationMethod == 2):
    #KITTI width adjustment (from 1242px to 1392px)
    if sequence.isKITTI:
        frame = cv2.copyMakeBorder(frame, 0, 0, 75, 75, cv2.BORDER_CONSTANT, value=(0,0,0))

    #get possible image padding when resizing
    #positive when padding is added, negative when the image is cropped
    _, paddingTop = resizeFrame(frame)

    detector = DetectorSingleTask(yoloConfThresh, yoloIouThresh, trackingIouThresh, bBoxMinSize, sequence.f, sequence.sensorW, frame.shape, paddingTop, showSettings, filterCarInLane, defaultBboxColor)
else:
    print("[ERROR] Select a valid estimation method (1,2)")
    exit()
currentDistance = -1
originalImgH, originalImgW, _ = frame.shape
pixelW = sequence.sensorW/originalImgW
f_u = sequence.f/pixelW


#start timer
st = time()

while(ret):
    printed = False
    #Get frame
    ret, frame = vid.read()


    if ret == False:
        break


    totalFrames += 1

    #KITTI width adjustment (from 1242px to 1392px)
    if sequence.isKITTI:
        frame = cv2.copyMakeBorder(frame, 0, 0, 75, 75, cv2.BORDER_CONSTANT, value=(0,0,0))

    #ESTIMATE CAMERA HEIGHT AND ROTATION
    linePointsLeft = (None, None)
    linePointsRight = (None, None)
    frame, linePointsLeft, linePointsRight, linesUpdated = findLaneCV(frame, linePointsLeft, linePointsRight)
    vanishingPoint = getVanishingPoint(linePointsLeft, linePointsRight, frame.shape[0])

    if(vanishingPoint):
        v_u = vanishingPoint[0] - originalImgW/2
        cameraPitch = -atan(v_u/f_u)
        v_v = vanishingPoint[1] - originalImgH/2
        cameraYaw = -atan(v_v/f_u*cos(cameraPitch))
        cameraHeight = roadWidth*((originalImgH-vanishingPoint[1])/vanishingPoint[0])

        detector.setCameraEstimations(cameraPitch, cameraYaw, cameraHeight)

        if(estimationMethod == 2):
            #Adjust lines to new frame size
            scaledLinePointsLeft = scaleRoadLinePoints(linePointsLeft, frame)
            scaledLinePointsRight = scaleRoadLinePoints(linePointsRight, frame)
            #Update detector road lines
            detector.setRoadLines(scaledLinePointsLeft, scaledLinePointsRight)

    frame, _ = resizeFrame(frame)


    #SCAN FOR CARS AND LINES
    sty = time()
    detector.detect(model, frame)
    ##-----------------------------------TEST-----------------------------
    mask = detector.getLaneMask()
    linePointsLeft = (None, None)
    linePointsRight = (None, None)
    linePointsLeft, linePointsRight, linesUpdated = houghFiltering(mask, linePointsLeft, linePointsRight)
    print(linePointsLeft, linePointsRight)
    showMask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    showMask = showRoadLines(showMask, linePointsLeft, linePointsRight)
    cv2.imshow('Frame',showMask)
    cv2.waitKey(1)
    continue
    ##---------------------------------------------------------------------
    totalTimeYolo += (time()-sty)*1000

    #If there are no cars, skip to next frame
    if(detector.nCars() == 0):
        cv2.imshow('Frame',frame)
        cv2.waitKey(1)
        print("-")
        continue


    #GET CAR SPEED
    cars = detector.getCars()
    for car in cars:

        if(showDistances and car['new']['distance'] and not printed):
            x1, y1, x2, y2 = car['new']['bbox']
            cv2.putText(frame, "{:6.2f}m".format(car['new']['distance']), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(255, 60, 255), lineType=cv2.LINE_AA)
            if(currentDistance < 0):
                currentDistance = car['new']['distance']
            else:
                newCurrentDistance = car['new']['distance']
                if abs(currentDistance-newCurrentDistance) < 300:
                    currentDistance = newCurrentDistance

            print(str(currentDistance).replace(".", ","))
            printed = True

        if(car['old']):
            frameTime = car['new']['time'] - car['old']['time']

            if(not car['new']['distance'] or not car['old']['distance']):
                    continue

            distanceDiff = car['new']['distance'] - car['old']['distance']

            if frameTime > frameTimeThreshold:

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
                    printMsg = f"\rRelVel: " + "{:.2f}".format(relVel) + "m/s Distance: " + "{:.2f}".format(car['new']['distance'] - vehicleBonnetSize) + "m secDist: "+"{:.2f}".format(secDist) + "m         "
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
    cv2.imshow('Frame',frame)
    cv2.waitKey(1)


    #Measure average time
    totalTime += (time()-st)*1000
    if(totalFrames>0):
        if(printTimes):
            printMsg = f"\r[INFO] avg time: "+"{:.2f}".format(totalTimeYolo/totalFrames)+"ms "
            sys.stdout.write(printMsg)
            sys.stdout.flush()
    if(not printed):
        print("-")

print(f"Totalframes: {totalFrames}")
print("System exiting successfully")
