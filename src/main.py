import cv2
import numpy as np
from estimation_methods import EstimationMethods
from data_extraction import DataExtractionTypes
from distance_detector import DistanceDetector
from auxFunctions import *
from time import time
from ultralytics import YOLO
from collections import deque

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
sequence = sequenceKITTI15
#------------------------------------------------------------------------------


#------------------------------SYSTEM PARAMETERS-------------------------------
videoPath = "../test_videos/"

#Detection model
modelName = "v4_2_tasks.onnx"
modelPath = "../models/"


#ALGORITHM PARAMETERS
yoloConfThresh = 0.3
yoloIouThresh = 0.5
trackingIouThresh = 0.5
bBoxMinSize = 0.025 #bboxes with a size smaller than 2.5% of the image are ignored


#ESTIMATION METHODS
#estimationMethod = EstimationMethods.roadWidthEstimation
estimationMethod = EstimationMethods.inverseProjection

roadWidth = 3.5 #m

#Use the estimated camera height to increase precision (for method 1)
heightCorrection = False

#To filter out estimation errors, a buffer of size n will be used. The selected
#distance will be the median of the last n estimated distances
distanceBufferSize = 5


#SPEED MEASURING
frameTimeThreshold = 1000 #ms


#VISUALIZATION
defaultBboxColor = (0, 255, 0)
filterCarInLane = True

#Select the predictions to show
showSettings = {
    "cars": True,
    "carId": False,
    "lanes": True
}

#DATA EXTRACTION
dataExtractionType = DataExtractionTypes.distances
#------------------------------------------------------------------------------

#Load model
model = YOLO(modelPath + modelName)
#Model warmup
model.predict(source=np.zeros((384,672, 3), dtype=np.uint8), imgsz=(384,672), device="cpu")
print("Model loaded")


#Get input video
vid = cv2.VideoCapture(videoPath+sequence.inputVideo)
vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)


#====== INITIALIZE VARIABLES =======
totalTime = 0
totalTimeYolo = 0
totalTimeLane = 0
totalFrames = 0
currentDistance = -1
ret = True
distanceBuffer = deque(maxlen=distanceBufferSize)


#read first frame to get resolution
ret, frame = vid.read()
vid.set(cv2.CAP_PROP_POS_FRAMES, 0) #reset video to first frame

#KITTI width adjustment (from 1242px to 1392px)
if sequence.isKITTI:
    frame = cv2.copyMakeBorder(frame, 0, 0, 75, 75, cv2.BORDER_CONSTANT, value=(0,0,0))

#get possible image padding when resizing
#positive when padding is added, negative when the image is cropped
_, paddingTop = resizeFrame(frame)


detector = DistanceDetector(
    #Car detection parameters
    yoloConfThresh,
    yoloIouThresh,
    trackingIouThresh,
    bBoxMinSize,
    #Camera parameters
    sequence.f,
    sequence.sensorW,
    #Frame resolution
    frame.shape,
    paddingTop,
    #Distance estimation settings
    estimationMethod,
    roadWidth,
    heightCorrection,
    #Visualization settings
    showSettings,
    filterCarInLane,
    defaultBboxColor)



#start timer
st = time()


#====== MAIN LOOP ======
print("Starting predictions")
while(ret):
    printedDataExtraction = False
    #Get frame
    ret, frame = vid.read()


    if ret == False:
        break


    totalFrames += 1

    #KITTI width adjustment (from 1242px to 1392px)
    if sequence.isKITTI:
        frame = cv2.copyMakeBorder(frame, 0, 0, 75, 75, cv2.BORDER_CONSTANT, value=(0,0,0))

    frame, _ = resizeFrame(frame)


    #SCAN FOR CARS AND LINES
    sty = time()
    detector.detectDistances(model, frame)

    totalTimeYolo += (time()-sty)*1000

    #If there are no cars, skip to next frame
    if(detector.nCars() == 0):
        cv2.imshow('Frame',frame)
        cv2.waitKey(1)
        print("-")
        continue


    #GET CAR DISTANCE
    cars = detector.getCars()
    for car in cars:

        if (car['new']['distance'] and not printedDataExtraction):
            x1, y1, x2, y2 = car['new']['bbox']
            cv2.putText(frame,
                        "{:6.2f}m".format(car['new']['distance']), (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        thickness=1,
                        color=(255, 60, 255),
                        lineType=cv2.LINE_AA)

            if(car['new']['distance'] > 0):
                if(currentDistance < 0):
                    currentDistance = car['new']['distance']
                else:
                    newCurrentDistance = car['new']['distance']
                    distanceBuffer.append(newCurrentDistance)
                    currentDistance = np.median(distanceBuffer)

            print(str(currentDistance).replace(".", ","))
            printedDataExtraction = True

        if(car['old']):
            frameTime = car['new']['time'] - car['old']['time']

            if(not car['new']['distance'] or not car['old']['distance']):
                    continue

            if frameTime > frameTimeThreshold:

                #Update old car
                car['old'] = {"distance": car['new']['distance'], "time": car['new']['time']}


    #show new frame
    cv2.imshow('Frame',frame)
    cv2.waitKey(1)


    if(dataExtractionType != dataExtractionType.none and
       not printedDataExtraction):
        print("-")


print("")
print(f"Totalframes: {totalFrames}")
print("System exiting successfully")
