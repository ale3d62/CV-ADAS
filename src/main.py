import cv2
import numpy as np
from estimation_methods import EstimationMethods
from data_extraction import DataExtractionTypes
from distance_detector import DistanceDetector
from auxFunctions import *
from time import time
import psutil
import os
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
sequenceWaymo13064 = SequenceConfig("video_waymo_13064.mp4", 38.33, 36, False)
sequenceWaymo13182 = SequenceConfig("video_waymo_13182.mp4", 38.95, 36, False)

#Choose here the dataset sequence to use
sequence = sequenceKITTI15
#------------------------------------------------------------------------------


#------------------------------SYSTEM PARAMETERS-------------------------------
videoPath = "../test_videos/"

#Detection model
modelName = "v4_2_tasks.onnx"
modelPath = "../models/"
resizedFrameSize = (384, 672) #(height, width)

#ALGORITHM PARAMETERS
yoloConfThresh = 0.3
yoloIouThresh = 0.5
trackingIouThresh = 0.5
bBoxMinSize = 0.025 #bboxes with a size smaller than 2.5% of the image are ignored


#ESTIMATION METHODS
#estimationMethod = EstimationMethods.roadWidthEstimation
estimationMethod = EstimationMethods.inverseProjection

roadWidth = 3.5 #m

#Use the estimated camera height to increase precision (for roadWidthEstimation method)
heightCorrection = True

#To filter out estimation errors, a buffer of size n will be used. The selected
#distance will be the median of the last n estimated distances
distanceBufferSize = 10


#VISUALIZATION
defaultBboxColor = (0, 255, 0)
filterCarInLane = True

#Select the predictions to show
showSettings = {
    "cars": True,
    "carId": False,
    "lanes": True
}

#Other statistics
showTime = True
showCPUUsage = True
showMemoryUsage = True

#DATA EXTRACTION
#dataExtractionType = DataExtractionTypes.none
dataExtractionType = DataExtractionTypes.distances
#dataExtractionType = DataExtractionTypes.cameraHeight
#dataExtractionType = DataExtractionTypes.cameraPitch
#dataExtractionType = DataExtractionTypes.cameraYaw
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
process = psutil.Process(os.getpid())
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
_, paddingTop = resizeFrame(frame, resizedFrameSize)


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
    resizedFrameSize,
    paddingTop,
    #Distance estimation settings
    estimationMethod,
    roadWidth,
    heightCorrection,
    #Visualization settings
    showSettings,
    filterCarInLane,
    defaultBboxColor)



#Start timer
startTime = time()
cpu_before = process.cpu_percent(interval=None)
mem_before = process.memory_info().rss / 1024**2  #MB


#====== MAIN LOOP ======
print("Starting predictions")
while(ret):

    #Get frame
    ret, frame = vid.read()

    if ret == False:
        break

    printedDataExtraction = False
    totalFrames += 1

    #KITTI width adjustment (from 1242px to 1392px)
    if sequence.isKITTI:
        frame = cv2.copyMakeBorder(frame, 0, 0, 75, 75, cv2.BORDER_CONSTANT, value=(0,0,0))

    frame, _ = resizeFrame(frame, resizedFrameSize)


    #SCAN FOR CARS AND LINES
    detector.detectDistances(model, frame)

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


            if(dataExtractionType == DataExtractionTypes.distances):
                print(str(currentDistance).replace(".", ","))

            elif(dataExtractionType == DataExtractionTypes.cameraHeight):
                print(str(detector.getCameraHeight()).replace(".", ","))

            elif(dataExtractionType == DataExtractionTypes.cameraPitch):
                print(str(np.degrees(detector.getCameraPitch())).replace(".", ","))

            elif(dataExtractionType == DataExtractionTypes.cameraYaw):
                print(str(np.degrees(detector.getCameraYaw())).replace(".", ","))

            printedDataExtraction = True


    #show new frame
    cv2.imshow('Frame',frame)
    cv2.waitKey(1)


    if(dataExtractionType != DataExtractionTypes.none and
       not printedDataExtraction):
        print("-")


elapsedTime = time() - startTime
cpu_after = process.cpu_percent(interval=None)
mem_after = process.memory_info().rss / 1024**2 #MB


print("")
print(f"Totalframes: {totalFrames}")

if(showTime):
    avgIterationTime = elapsedTime/totalFrames
    avgIterationTimeMs = avgIterationTime*1000
    print(f"Average time per iteration: {avgIterationTimeMs:.4f}ms".replace(".",","))
if(showCPUUsage):
    print(f"CPU usage: {(cpu_after/psutil.cpu_count()):.1f}%".replace(".",","))
if(showMemoryUsage):
    print(f"Memory usage: {mem_after:.1f}MB".replace(".",","))

print("System exiting successfully")
