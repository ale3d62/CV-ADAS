NoneType = type(None)
from auxFunctions import *
import cv2
import torch
from time import time
from math import sqrt, atan, cos
import numpy as np


class DetectorMultiTask:

    def __init__(self, yoloConfThresh, yoloIouThresh, trackingIouThresh, bBoxMinSize, f, sensorW, roadWidth, showSettings, filterCarInLane, heightCorrection, defaultBboxColor):
        self._cars = []
        self._yoloConfThresh = yoloConfThresh
        self._yoloIouThresh = yoloIouThresh
        self._bBoxMinSize = bBoxMinSize
        self._trackingIouThresh = trackingIouThresh
        self._id = 0

        #camera parameters
        self._f = f
        self._roadWidth = roadWidth
        self._sensorW = sensorW

        #camera estimations (these are set in setCameraEstimations())
        self._cameraPitch = 0
        self._cameraYaw = 0
        self._cameraHeight = 0

        #Settings
        self._showCars = showSettings["cars"]
        self._showCarId = showSettings["carId"]
        self._showLanes = showSettings["lanes"]
        self._filterCarInLane = filterCarInLane
        self._heightCorrection = heightCorrection

        #Internal variables
        self._currentTime = None
        self._laneMask = None
        self._minLineWidth = 2
        self._bBoxColor = defaultBboxColor
        self._bestLinePointsLeft = (None, None)
        self._bestLinePointsRight = (None, None)



    def nCars(self):
        return len(self._cars)

    def getCars(self):
        return self._cars

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



    def setCameraEstimations(self, pitch, yaw, height):
        self._cameraPitch = pitch
        self._cameraYaw = yaw
        self._cameraHeight = height



    def nextId(self):
        self._id += 1
        return self._id



    def updateCars(self, frame, newBboxes):

        #Replace previous bBoxes
        updatedBboxes = []
        for bBox in self._cars:

            #if bBox has already been replaced, skip
            if(bBox['updated']):
                continue

            #calculate iou with every new bBox
            ious = []
            for newBbox in newBboxes:
                ious.append(self.calculateIou(newBbox, bBox['new']['bbox']))

            #replace the current bBox with the new bBox with the best iou
            maxIou = max(ious) if len(ious) > 0 else 0
            if maxIou > self._trackingIouThresh:
                iMaxIou = ious.index(maxIou)
                if not bBox['old'] or not bBox['old']['distance']:
                    bBox['old'] = {"distance": bBox['new']['distance'], "time": bBox['new']['time']}

                bBox['new'] = {"bbox": newBboxes[iMaxIou], "time": self._currentTime, "distance": None, "speed": bBox['new']['speed']}
                bBox['updated'] = True
                updatedBboxes.append(bBox)
                #remove from newBboxes list
                newBboxes.pop(iMaxIou)

        self._cars = updatedBboxes


        #Add remaining newBboxes
        for newBbox in newBboxes:
            self._cars.append({"id": self.nextId(), "color": self._bBoxColor, "old": None, "new": {"bbox": newBbox, "time": self._currentTime, "distance": None, "speed": None}, "updated": True})


        #show bBoxes
        for bBox in self._cars:
            if self._showCars:
                x1, y1, x2, y2 = bBox['new']['bbox']
                id = bBox['id']
                cv2.rectangle(frame, (x1, y1), (x2, y2), bBox['color'], 1)
                if self._showCarId:
                    cv2.putText(frame, "ID: {:.0f}".format(id), (x1, y2), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(100, 100, 255))
            bBox['updated'] = False



    #Returns the bboxes of the acceptedClasses found in the frame
    def detect(self, model, frame):

        #Detect cars and road lanes
        newBboxes = self.predict(model, frame)

        #Estimate camera parameters
        self.estimateCameraParameters()

        #Update and show detected cars
        self.updateCars(frame, newBboxes)

        #Update distances
        self.updateDist(frame.shape)



    def predict(self, model, frame):

        results = model.predict(source=frame, imgsz=(384,672), conf=self._yoloConfThresh, iou=self._yoloIouThresh, verbose=False, device="cpu", stream=True)
        self._currentTime = time()*1000

        newBboxes = []

        if results:

            # PROCESS LANE
            self._laneMask = results[-1][0].to(torch.uint8).cpu().numpy()

            # Show lane
            if self._showLanes:
                frame[self._laneMask==1] = (30, 255, 15)

            frameH, frameW, _ = frame.shape

            # PROCESS BBOXES
            for bbox in results[0][0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = bbox

                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                bBox = (x1,y1,x2,y2)

                if(x2-x1 > frameW*self._bBoxMinSize and y2-y1 > frameH*self._bBoxMinSize):
                    newBboxes.append(bBox)

        return newBboxes



    def updateDist(self, frameDim):
        for car in self._cars:
            car['new']['distance'] = self.getDistance(frameDim, car['new']['bbox'])



    def carInlane(self, x1,x2,y2, lx3, rx3, imgDim):
        imgH, imgW, _ = imgDim
        imgCenter = imgW/2

        #if y2 is at the wrong height
        if(y2 > imgH or y2 < imgH * 0.25):
            return False

        #Detect if car is to the left, center, or right

        #center
        if(x1 < imgCenter and x2 > imgCenter):
            return True

        bBoxW = x2-x1

        #left
        if(x1 < imgCenter and x2 < imgCenter):
            return ((x2-lx3) / bBoxW > 0.3)

        #right
        if(x1 > imgCenter and x2 > imgCenter):
            return ((rx3-x1) / bBoxW > 0.3)

        return False



    def getDistance(self, frameDim, bBox):

        imgHeight, imgWidth, _ = frameDim

        x1, y1, x2, y2 = bBox
        #1% mas bajo que el borde inferior de la bbox
        #y2 = int(y2*1.01)

        #coordinates x of the lines at the car's height
        lx3, rx3 = self.getLinesCoords(int(imgWidth/2), y2, frameDim)

        if not lx3 or not rx3:
            return None

        roadWidthPx = rx3-lx3

        if roadWidthPx == 0:
            return None

        #if car is in lane
        if(self._filterCarInLane and not self.carInlane(x1,x2,y2, lx3, rx3, frameDim)):
            d = None
        else:
            d = (self._roadWidth * self._f)/(self._sensorW * (roadWidthPx/imgWidth))

        #Apply height correction
        if(self._heightCorrection and d and self._cameraHeight > 0):
            d = sqrt(d**2-self._cameraHeight**2)

        return d



    #Returns the closest pixel of the mask to both sides of the x,y point at y height in both sides
    def getLinesCoords(self, x, y, frameDim):
        imgHeight, imgWidth, _ = frameDim

        if y < 0 or y >= imgHeight:
            return (None, None)

        lx3 = rx3 = None

        #left line
        lx = x
        while lx > 0 and lx3 == None:
            if self._laneMask[y][lx] == 0:
                lx -= (self._minLineWidth-1)
            else:
                while lx < imgWidth and self._laneMask[y][lx] == 1:
                    lx+=1
                lx3 = lx + 1

        if(lx3 != None):
            lx -=1
            while(lx>0 and self._laneMask[y][lx] == 1):
                lx -=1
            lx3 = lx3 - ((lx3-lx)/2)

        #right line
        rx = x
        while rx > 0 and rx < imgWidth and rx3 == None:
            if self._laneMask[y][rx] == 0:
                rx += (self._minLineWidth-1)
            else:
                while self._laneMask[y][rx] == 1:
                    rx-=1
                rx3 = rx - 1

        if(rx3 != None):
            rx+=1
            while(rx < imgWidth and self._laneMask[y][rx] == 1):
                rx +=1
            rx3 = rx3 + ((rx-rx3)/2)

        return (lx3, rx3)



    def getLaneMask(self):
        mask_2d = np.squeeze(self._laneMask)

        # 2. Asegurar el rango de valores (0 y 255)
        # Si la máscara de YOLO devuelve 0 y 1, la escalamos a 255
        if np.max(mask_2d) <= 1:
            mask_2d = (mask_2d * 255)

        # Asegurar el tipo de dato requerido por OpenCV
        mask_2d = mask_2d.astype(np.uint8)

        # 3. Aplicar el algoritmo de adelgazamiento (Zhang-Suen es el estándar)
        thinned_mask = cv2.ximgproc.thinning(mask_2d, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        return thinned_mask



    def estimateCameraParameters(self):
        #Clean the road lane mask
        cleanLaneMask = self.getCleanLaneMask()

        #Get road lanes using hough
        linesUpdated = False
        houghFiltering(cleanLaneMask, self._linePointsLeft, self._linePointsRight, linesUpdated)

        vanishingPoint = getVanishingPoint(self._linePointsLeft, self._linePointsRight, cleanLaneMask.shape[0])

        if(vanishingPoint):
            v_u = vanishingPoint[0] - self._originalImgW/2
            self._cameraPitch = -atan(v_u/self._f_u)
            v_v = vanishingPoint[1] - self._originalImgH/2
            self._cameraYaw = -atan(v_v/self._f_u*cos(self._cameraPitch))
            self._cameraHeight = self._roadWidth*((self._originalImgH-vanishingPoint[1])/vanishingPoint[0])



def houghFiltering(mask, bestLinePointsLeft, bestLinePointsRight):
    linesUpdated = False
    imgHeight, imgWidth = mask.shape
    halfImgHeight = int(imgHeight/2)
    halfImg = mask[halfImgHeight:imgHeight, 1:imgWidth]

    lines = cv2.HoughLinesP(halfImg, 1, np.pi/180, 30, minLineLength=50, maxLineGap=30)

    if(type(lines) == NoneType):
        return (bestLinePointsLeft, bestLinePointsRight, linesUpdated)
    #showMask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    #Lines processing
    linesLeft = [[],[]]
    linesRight = [[],[]]
    for line in lines:
        arr = np.array(line[0], dtype=np.int32)
        x1,y1,x2,y2 = arr

        if(x2 == x1):
            continue

        m = (y2-y1)/(x2-x1)
        b = y1 - m * x1


        #filter lines by angle
        lineAngle = abs(np.arctan(m))

        #not a line (60-120 deg or < 25 deg or > 155 deg)
        if((lineAngle > 1 and lineAngle < 2.2) or lineAngle < 0.43 or lineAngle > 2.7):
            continue

        xCutBottom = int((halfImgHeight-b)/m)
        xCutTop = int(-b/m)

        #filter mask line
        maskAngle = np.arcsin(halfImgHeight/(np.sqrt(halfImgHeight**2+(imgWidth*0.3)**2)))
        if((xCutBottom < imgWidth*0.01 and lineAngle > maskAngle-maskAngle*0.1  and lineAngle < maskAngle+maskAngle*0.1) or
            xCutBottom > imgWidth*0.99 and lineAngle > maskAngle-maskAngle*0.1 and lineAngle < maskAngle+maskAngle*0.1):
            continue

        #Classify left and right
        if(m > 0):  #rightLine
            if(xCutBottom < imgWidth*0.55):
                continue
            linesRight[0].append(xCutBottom)
            linesRight[1].append(xCutTop)
        else:       #left line
            if(xCutBottom > imgWidth*0.45):
                continue
            linesLeft[0].append(xCutBottom)
            linesLeft[1].append(xCutTop)
    #for i in range(min(len(linesLeft[0]), len(linesRight[0]))):
    #    linePointsShowLeft = (linesLeft[0][i], linesLeft[1][i])
    #    linePointsShowRight = (linesRight[0][i], linesRight[1][i])
    #    showMask = showRoadLines(showMask, linePointsShowLeft, linePointsShowRight)
    #cv2.imshow('Frame',showMask)
    #cv2.waitKey(1)

    newBestLinePointsLeft = getBestLine(linesLeft, 30, max(2, int(len(linesRight))), False)
    newBestLinePointsRight = getBestLine(linesRight, 30, max(2, int(len(linesRight))), False)

    if(newBestLinePointsLeft[0]):
        bestLinePointsLeft = newBestLinePointsLeft
    if(newBestLinePointsRight[0]):
        bestLinePointsRight = newBestLinePointsRight


    if(newBestLinePointsLeft[0] and newBestLinePointsRight[0]):
        linesUpdated = True

    return (bestLinePointsLeft,bestLinePointsRight, linesUpdated)



def getVanishingPoint(lineLeft, lineRight, height):
    if(lineLeft[0] == None or lineRight[0] == None):
        return None

    x_bottom1, x_top1 = lineLeft
    x_bottom2, x_top2 = lineRight

    H = height - 1

    # Pendientes en forma x = x_top + m*y
    m1 = (x_bottom1 - x_top1) / H
    m2 = (x_bottom2 - x_top2) / H

    # Comprobamos paralelismo
    if abs(m1 - m2) < 1e-6:
        return None  # Líneas casi paralelas

    # Coordenada y del punto de fuga
    y_vp = (x_top2 - x_top1) / (m1 - m2)

    # Coordenada x
    x_vp = x_top1 + m1 * y_vp

    return (x_vp, y_vp + height/2)