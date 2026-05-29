from time import time
import torch
from estimation_methods import EstimationMethods, roadWidthDistanceEstimation, inverseProjectionDistanceEstimation
from road_lane import RoadLane
import cv2
from math import atan, cos
import numpy as np
from collections import deque


class DistanceDetector():

    def __init__(self,
                 _yoloConfThresh,
                 _yoloIouThresh,
                 _trackingIouThresh,
                 _bBoxMinSize,
                 #Camera parameters
                 _f,
                 _sensorW,
                 #Frame resolution
                 _originalFrameShape,
                 _resizedFrameSize,
                 _paddingTop,
                 #Distance estimation settings
                 _estimationMethod,
                 _roadWidth,
                 _heightCorrection,
                 #Display settings
                 _showSettings,
                 _filterCarInLane,
                 _defaultBboxColor):

        #Car detection parameters
        self.yoloConfThresh     = _yoloConfThresh
        self.yoloIouThresh      = _yoloIouThresh
        self.bBoxMinSize        = _bBoxMinSize
        self.trackingIouThresh  = _trackingIouThresh

        #Camera parameters
        self.f          = _f
        self.sensorW    = _sensorW
        self.f_u        = 0 #Initialiced in estimateCameraParameters()
        self.f_v        = 0 #...

        #Frame resolution
        self.originalImgH, self.originalImgW, _ = _originalFrameShape
        self.resizedImgH, self.resizedImgW = _resizedFrameSize
        self.paddingTop = _paddingTop

        #Distance estimation settings
        self.estimationMethod  = _estimationMethod
        self.roadWidth          = _roadWidth
        self.heightCorrection   = _heightCorrection

        #Camera estimations (these are set in setCameraEstimations())
        camParameterBufferSize = 40
        self.cameraHeightBuffer = deque(maxlen=camParameterBufferSize)
        self.cameraHeight = 0
        self.cameraPitchBuffer = deque(maxlen=camParameterBufferSize)
        self.cameraPitch = 0
        self.cameraYawBuffer = deque(maxlen=camParameterBufferSize)
        self.cameraYaw = 0

        #Settings
        self.showCars           = _showSettings["cars"]
        self.showCarId          = _showSettings["carId"]
        self.showLanes          = _showSettings["lanes"]
        self.filterCarInLane    = _filterCarInLane
        self.bBoxColor          = _defaultBboxColor

        #Internal variables
        self.cars               = []
        self.newCarId           = 0
        self.currentTime        = None

        #Road lines
        self.roadLane = RoadLane(2, _originalFrameShape, self.paddingTop)



    #Observer methods
    def nCars(self):
        return len(self.cars)

    def getCars(self):
        return self.cars

    def getCameraHeight(self):
        return self.cameraHeight

    def getCameraPitch(self):
        return self.cameraPitch

    def getCameraYaw(self):
        return self.cameraYaw



    #Main method
    def detectDistances(self, model, frame):

        #Detect cars and road lanes
        newBboxes = self.predict(model, frame)

        #Estimate camera parameters
        self.estimateCameraParameters()

        #Update and show detected cars
        self.updateCars(frame, newBboxes)

        #Update distances
        self.updateDistances(frame.shape)

        #Show detected lanes
        if self.showLanes:
            self.roadLane.showLane(self.estimationMethod, frame)



    def estimateCameraParameters(self):
        #Get focal distances
        if(self.originalImgW > 0):
            pixelW = self.sensorW/self.originalImgW
            self.f_u = self.f_v = self.f/pixelW

            #Get vanishing point (vpx, vpy)
            vanishingPoint = self.roadLane.estimateVanishingPoint(self.resizedImgH)
            vanishingPoint = self.roadLane.scaleVanishingPoint(vanishingPoint, self.resizedImgH, self.resizedImgW)

            #Get road width in pixels at the bottom of the image
            scaledLinePointsLeft = self.roadLane.scaleRoadLinePoints(self.roadLane.linePointsLeft, (self.resizedImgH, self.resizedImgW), self.originalImgW, self.originalImgH)
            scaledLinePointsRight = self.roadLane.scaleRoadLinePoints(self.roadLane.linePointsRight, (self.resizedImgH, self.resizedImgW), self.originalImgW, self.originalImgH)

            w_px = scaledLinePointsRight[0]-scaledLinePointsLeft[0]

            if(vanishingPoint):
                v_u = vanishingPoint[1] - self.originalImgH / 2
                v_v = vanishingPoint[0] - self.originalImgW / 2
                #Pitch
                estimatedCameraPitch = -atan(v_u / self.f_u)
                self.cameraPitchBuffer.append(estimatedCameraPitch)
                self.cameraPitch = np.average(self.cameraPitchBuffer)
                #Yaw
                estimatedCameraYaw = -atan(v_v / self.f_u * cos(estimatedCameraPitch))
                self.cameraYawBuffer.append(estimatedCameraYaw)
                self.cameraYaw = np.average(self.cameraYawBuffer)
                #Height
                estimatedCameraHeight = self.roadWidth * (
                    (self.originalImgH - vanishingPoint[1]) / w_px)
                self.cameraHeightBuffer.append(estimatedCameraHeight)
                self.cameraHeight = np.average(self.cameraHeightBuffer)



    def predict(self, model, frame):

        results = model.predict(source=frame,
                                imgsz=(384,672),
                                conf=self.yoloConfThresh,
                                iou=self.yoloIouThresh,
                                verbose=False,
                                device="cpu",
                                stream=True)

        self.currentTime = time()*1000

        newBboxes = []

        if results:

            # PROCESS LANE
            self.roadLane.laneMask = results[-1][0].to(torch.uint8).cpu().numpy()

            frameH, frameW, _ = frame.shape

            # PROCESS BBOXES
            for bbox in results[0][0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = bbox

                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                bBox = (x1,y1,x2,y2)

                if(x2-x1 > frameW*self.bBoxMinSize and
                   y2-y1 > frameH*self.bBoxMinSize):
                    newBboxes.append(bBox)

        return newBboxes



    #Car detection
    def nextCarId(self):
        self.newCarId += 1
        return self.newCarId

    def calculateIou(self, bbox1, bbox2):

        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)

        interArea = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1Area = (x2 - x1) * (y2 - y1)
        bbox2Area = (x4 - x3) * (y4 - y3)
        unionArea = bbox1Area + bbox2Area - interArea

        iou = interArea / unionArea

        return iou

    #Updates the cars bounding boxes
    def updateCars(self, frame, newBboxes):

        #Replace previous bBoxes
        updatedBboxes = []
        for bBox in self.cars:

            #if bBox has already been replaced, skip
            if(bBox['updated']):
                continue

            #calculate iou with every new bBox
            ious = []
            for newBbox in newBboxes:
                ious.append(self.calculateIou(newBbox, bBox['new']['bbox']))

            #replace the current bBox with the new bBox with the best iou
            maxIou = max(ious) if len(ious) > 0 else 0
            if maxIou > self.trackingIouThresh:
                iMaxIou = ious.index(maxIou)
                if not bBox['old'] or not bBox['old']['distance']:
                    bBox['old'] = {
                        "distance": bBox['new']['distance'],
                        "time": bBox['new']['time']
                        }

                bBox['new'] = {
                    "bbox": newBboxes[iMaxIou],
                    "time": self.currentTime,
                    "distance": None
                    }
                bBox['updated'] = True
                updatedBboxes.append(bBox)
                #remove from newBboxes list
                newBboxes.pop(iMaxIou)

        self.cars = updatedBboxes

        #Add remaining newBboxes
        for newBbox in newBboxes:
            self.cars.append({
                "id": self.nextCarId(),
                "color": self.bBoxColor,
                "old": None,
                "new": {
                    "bbox": newBbox,
                    "time": self.currentTime,
                    "distance": None,
                    "speed": None
                    },
                "updated": True
                })

        #show bBoxes
        for bBox in self.cars:
            if self.showCars:
                x1, y1, x2, y2 = bBox['new']['bbox']
                id = bBox['id']

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              bBox['color'],
                              1)

                if self.showCarId:
                    cv2.putText(frame,
                                "ID: {:.0f}".format(id),
                                (x1, y2),
                                cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1,
                                thickness=1,
                                color=(100, 100, 255))
            bBox['updated'] = False

    #Updates the distance to every car
    def updateDistances(self, frameDim):

        for car in self.cars:

            if(self.estimationMethod == EstimationMethods.roadWidthEstimation):
                car['new']['distance'] = roadWidthDistanceEstimation(
                    frameDim,
                    self.roadLane,
                    car['new']['bbox'],
                    self.filterCarInLane,
                    self.roadWidth,
                    self.f,
                    self.sensorW,
                    self.heightCorrection,
                    self.cameraHeight)

            elif(self.estimationMethod == EstimationMethods.inverseProjection):
                car['new']['distance'] = inverseProjectionDistanceEstimation(frameDim,
                 self.roadLane,
                 car['new']['bbox'],
                 self.paddingTop,
                 self.originalImgH,
                 self.originalImgW,
                 self.f_u,
                 self.f_v,
                 self.cameraPitch,
                 self.cameraYaw,
                 self.cameraHeight)
