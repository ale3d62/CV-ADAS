import cv2
from time import time
class CarDetector:

    def __init__(self, trackingIouThresh, camParams, showCars):
        self._cars = []
        self._trackingIouThresh = trackingIouThresh
        self._id = 0
        self._f = camParams["f"]
        self._sensorPixelW = camParams["sensorPixelW"]
        self._roadWidth = camParams["roadWidth"]
        self._showCars = showCars
        self._currentTime = None



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
                bBox['old'] = bBox['new']
                bBox['new'] = {"bbox": newBboxes[iMaxIou], "id": bBox['old']['id'], "time": self._currentTime}
                bBox['updated'] = True
                updatedBboxes.append(bBox)
                #remove from newBboxes list
                newBboxes.pop(iMaxIou)
        
        self._cars = updatedBboxes
                
        
        #Add remaining newBboxes
        for newBbox in newBboxes:
            self._cars.append({"old": None, "new": {"bbox": newBbox, "id": self.nextId(), "time": self._currentTime}, "updated": True})


        #show bBoxes
        for bBox in self._cars:
            if self._showCars:
                x1, y1, x2, y2 = bBox['new']['bbox']
                #id = bBox['new']['id']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                #cv2.putText(frame, "ID: {:6.2f}m".format(id), (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(100, 100, 255))
            bBox['updated'] = False

             


    #Returns the bboxes of the acceptedClasses found in the frame 
    def findCars(self, model, frame, acceptedClasses):
        results = model(frame, verbose=False, conf=0.2)[0]

        self._currentTime = time()*1000

        newBboxes = []
        #filter yolo results by class
        for result in results.boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = result
            
            if(class_id in acceptedClasses):

                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)

                bBox = (x1,y1,x2,y2)
                newBboxes.append(bBox)

        #Update detected cars
        self.updateCars(frame, newBboxes)
                     


    def updateDist(self, frameDim, bestLinePointsLeft, bestLinePointsRight):
        
        #distances = []

        for car in self._cars:
            car['new']['distance'] = self.getDistance(frameDim, car['new']['bbox'], bestLinePointsLeft, bestLinePointsRight)
            #if car['new']['distance']:
            #    distances.append(car['new'])

        #return distances



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
    


    def getDistance(self, frameDim, bBox, bestLinePointsLeft, bestLinePointsRight):

        #Get lines data
        lx1, lx2 = bestLinePointsLeft
        rx1, rx2 = bestLinePointsRight

        #if some lines data is missing
        if(not lx1 or not lx2 or not rx1 or not rx2):
            return None


        imgHeight, imgWidth, _ = frameDim
        halfImgHeight = int(imgHeight/2)

        #get lines equations (y = mx + b)
        lm = halfImgHeight/(lx2-lx1) # left
        lb = -lm * lx1
        rm = halfImgHeight/(rx2-rx1) # right
        rb = -rm * rx1

        #flip lines equations (as increasing means going down in the image)
        lm = -lm
        lb = -lb + imgHeight
        rm = -rm
        rb = -rb + imgHeight

        #get vanishing point coordinates
        vpx = (rb-lb) / (lm-rm)
        vpy = lm*vpx + lb


        x1, y1, x2, y2 = bBox
        
        #coordinates x of the lines at the car's height
        lx3 = (y2-lb)/lm
        rx3 = (y2-rb)/rm

        #if car is in lane
        #if(self.carInlane(x1,x2,y2, lx3, rx3, vpy, vpx, imgHeight)):
        d = (self._f*self._roadWidth*imgWidth)/((rx3-lx3)*(self._sensorPixelW*imgWidth))
        d = d/1000

        return d

        #else:
        #    return None



