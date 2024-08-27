import cv2
import torch
from time import time
from math import sqrt

class Detector:

    def __init__(self, yoloConfThresh, yoloIouThresh, trackingIouThresh, bBoxMinSize, camParams, showSettings):
        self._cars = []
        self._yoloConfThresh = yoloConfThresh
        self._yoloIouThresh = yoloIouThresh
        self._bBoxMinSize = bBoxMinSize
        self._trackingIouThresh = trackingIouThresh
        self._id = 0
        
        #camera parameters
        self._fReal = camParams["fReal"]
        self._roadWidth = camParams["roadWidth"]
        fEq = camParams["fEq"]
        self._sensorDiag = self._fReal * 43.267 / fEq
        self._sensorW = None

        self._showCars = showSettings["cars"]
        self._showLanes = showSettings["lanes"]
        self._currentTime = None
        self._laneMask = None
        self._minLineWidth = 4


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
                if not bBox['old']:
                    bBox['old'] = {"distance": bBox['new']['distance'], "time": bBox['new']['time']}
                bBox['new'] = {"bbox": newBboxes[iMaxIou], "time": self._currentTime, "distance": None, "speed": bBox['new']['speed']}
                bBox['updated'] = True
                updatedBboxes.append(bBox)
                #remove from newBboxes list
                newBboxes.pop(iMaxIou)
        
        self._cars = updatedBboxes
                
        
        #Add remaining newBboxes
        for newBbox in newBboxes:
            self._cars.append({"id": self.nextId(), "old": None, "new": {"bbox": newBbox, "time": self._currentTime, "distance": None, "speed": None}, "updated": True})


        #show bBoxes
        for bBox in self._cars:
            if self._showCars:
                x1, y1, x2, y2 = bBox['new']['bbox']
                #id = bBox['new']['id']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                #cv2.putText(frame, "ID: {:6.2f}m".format(id), (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(100, 100, 255))
            bBox['updated'] = False

             


    #Returns the bboxes of the acceptedClasses found in the frame 
    def detect(self, model, frame):
        results = model.predict(source=frame, imgsz=(384,672), conf=self._yoloConfThresh, iou=self._yoloIouThresh, verbose=False)

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

        #Update and show detected cars
        self.updateCars(frame, newBboxes)

        #Update distances
        self.updateDist(frame.shape)
                     


    def updateDist(self, frameDim):
        
        for car in self._cars:
            car['new']['distance'] = self.getDistance(frameDim, car['new']['bbox'])




    def carInlane(self, x1,x2,y2, lx3, rx3, vpy, vpx, imgHeight):
    
        #if y2 is at the wrong height
        if(y2 > imgHeight or y2 < vpy):
            return False 
        

        #Detect if car is to the left, center, or right
        #center
        if(x1 < vpx and x2 > vpx):
            return True
        
        upPercent = (imgHeight-y2) / (imgHeight-vpy)
        if(upPercent < 0.42):
            return False

        boxWidth = x2-x1

        #left
        if(x2 < vpx):
            return (x1 + boxWidth*upPercent*0.9 > lx3)

        #right
        if(x1 > vpx):
            return (x2 - boxWidth*upPercent*0.9 < rx3)

        return False
    


    def getDistance(self, frameDim, bBox):

        imgHeight, imgWidth, _ = frameDim

        if(not self._sensorW):
            aspectRatio = imgWidth/imgHeight
            self._sensorW = (self._sensorDiag * aspectRatio)/(sqrt(1+1/pow(aspectRatio, 2)))
        
        x1, y1, x2, y2 = bBox
        
        #coordinates x of the lines at the car's height
        lx3, rx3 = self.getLinesCoords(int(imgWidth/2), y2, frameDim)

        if not lx3 or not rx3:
            return None

        roadWidthPx = rx3-lx3

        if roadWidthPx == 0:
            return None
        
        #if car is in lane
        #if(self.carInlane(x1,x2,y2, lx3, rx3, vpy, vpx, imgHeight)):
        #d = (self._f*self._roadWidth*imgWidth)/((rx3-lx3)*(self._sensorPixelW*imgWidth))
        d = (self._roadWidth * self._fReal)/(self._sensorW * (roadWidthPx/imgWidth))
        #d = d/1000

        return d

        #else:
        #    return None


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
                while self._laneMask[y][lx] == 1:
                    lx+=1
                lx3 = lx + 1

        #right line
        rx = x
        while rx < imgWidth and rx3 == None:
            if self._laneMask[y][rx] == 0:
                rx += (self._minLineWidth-1)
            else:
                while self._laneMask[y][rx] == 1:
                    rx-=1
                rx3 = rx - 1

        return (lx3, rx3)


                