import cv2
from time import time


class DetectorSingleTask:

    def __init__(self, yoloConfThresh, yoloIouThresh, trackingIouThresh, bBoxMinSize, f, sensorW, originalImgShape, paddingTop, showSettings, filterCarInLane, defaultBboxColor):
        self._cars = []
        self._yoloConfThresh = yoloConfThresh
        self._yoloIouThresh = yoloIouThresh
        self._bBoxMinSize = bBoxMinSize
        self._trackingIouThresh = trackingIouThresh
        self._id = 0

        #camera parameters
        self._f = f
        self._sensorW = sensorW

        self._originalImgWidth = originalImgShape[1]
        self._originalImgHeight = originalImgShape[0]
        self._paddingTop = paddingTop

        #camera estimations (these are set in setCameraEstimations())
        self._cameraPitch = 0
        self._cameraYaw = 0
        self._cameraHeight = 0

        pixelW = self._sensorW/self._originalImgWidth
        self._f_u = self._f_v = self._f/pixelW

        #Road lines
        self._linePointsLeft = (None, None)
        self._linePointsRight = (None, None)

        self._showCars = showSettings["cars"]
        self._showCarId = showSettings["carId"]
        self._showLanes = showSettings["lanes"]
        self._filterCarInLane = filterCarInLane
        self._currentTime = None
        self._bBoxColor = defaultBboxColor


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



    def setRoadLines(self, linePointsLeft, linePointsRight):
        self._linePointsLeft = linePointsLeft
        self._linePointsRight = linePointsRight



    def nextId(self):
        self._id += 1
        return self._id



    def showRoadLines(self, frame):
        imgHeight, _, _ = frame.shape
        halfImgHeight = int(imgHeight/2)

        if(self._linePointsLeft[0] and self._linePointsLeft[1]):
            cv2.line(frame, (self._linePointsLeft[1], halfImgHeight), (self._linePointsLeft[0], imgHeight), (0, 0, 255), 2)
        if(self._linePointsRight[0] and self._linePointsRight[1]):
            cv2.line(frame, (self._linePointsRight[1], halfImgHeight), (self._linePointsRight[0], imgHeight), (0, 0, 255), 2)

        return frame



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

        newBboxes = self.predict(model, frame)

        #show lanes
        if(self._showLanes):
            frame = self.showRoadLines(frame)

        #Update and show detected cars
        self.updateCars(frame, newBboxes)

        #Update distances
        self.updateDist(frame.shape)



    def predict(self, model, frame):

        results = model.predict(source=frame, imgsz=(384,672), conf=self._yoloConfThresh, iou=self._yoloIouThresh, verbose=False, device="cpu", stream=True)
        self._currentTime = time()*1000

        newBboxes = []

        # PROCESS BBOXES
        if results:
            frameH, frameW, _ = frame.shape
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

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



    def carInlane(self, x1, x2, y2, imgDim):
        if (self._linePointsLeft[0] == None
        or self._linePointsLeft[1] == None
        or self._linePointsRight[0] == None
        or self._linePointsRight[1] == None):
            return False

        imgH, imgW, _ = imgDim

        y_bottom = imgH
        y_top = imgH/2

        x_center = x1+(x2-x1)/2

        #if y2 is at the wrong height
        if(y2 > imgH or y2 < imgH * 0.25):
            return False

        def get_x_at_y(line, y):
            x_bottom, x_top = line

            dx = x_bottom - x_top
            dy = y_bottom - y_top

            #x - x_top = (dx / dy) * (y - y_top)
            return x_top + (dx / dy) * (y - y_top)

        x_line1 = get_x_at_y(self._linePointsLeft, y2)
        x_line2 = get_x_at_y(self._linePointsRight, y2)

        x_left = min(x_line1, x_line2)
        x_right = max(x_line1, x_line2)

        return x_left <= x_center <= x_right



    def calcDistance(self, u, v, frameDim):
        import numpy as np

        #H = 2.12
        imgHeight, imgWidth, _ = frameDim

        imgHeight = imgHeight-(self._paddingTop*2)

        v = v-self._paddingTop
        #print(f"previousU: {u}, previousV: {v}, paddingTop: {self._paddingTop}")
        v = v*self._originalImgHeight/imgHeight
        u = u*self._originalImgWidth/imgWidth

        #print(f"originalh: {self._originalImgHeight}, originalw: {self._originalImgWidth}, imgW: {imgWidth}, imgh: {imgHeight}, u: {u}, v: {v}")
        f_u = self._f_u
        f_v = self._f_v
        c_u = self._originalImgWidth/2
        c_v = self._originalImgHeight/2
        pitch = self._cameraPitch
        yaw = self._cameraYaw
        height = self._cameraHeight

        #10923
        #f_u = 2063.54
        #f_v = 2063.54
        #c_u = 956.07
        #c_v = 649.19
        #pitch = np.deg2rad(0.65)
        #yaw = np.deg2rad(-0.12)

        #10625
        #f_u = 2055.56
        #f_v = 2055.56
        #c_u = 939.65
        #c_v = 641.07
        #pitch = np.deg2rad(0.76)
        #yaw = np.deg2rad(0.34)

        #11199
        #f_u = 2079.67
        #f_v = 2079.67
        #c_u = 951.05
        #c_v = 656.09
        #pitch = np.deg2rad(0.14)
        #yaw = np.deg2rad(0.12)
        #---------------------------------#

        x_c = (u-c_u)/f_u
        y_c = (v-c_v)/f_v
        z_c = 1

        x_p = x_c
        y_p = y_c*np.cos(pitch)+z_c*np.sin(pitch)
        z_p = y_c*np.sin(pitch)+z_c*np.cos(pitch)

        x_w = x_p*np.cos(yaw)+z_p*np.sin(yaw)
        y_w = y_p
        z_w = -x_p*np.sin(yaw)+z_p*np.cos(yaw)

        S = height/y_w

        Z = S*z_w

        return Z



    def getDistance(self, frameDim, bBox):

        x1, y1, x2, y2 = bBox

        if(self.carInlane(x1,x2,y2,frameDim)):
            #considering straight camera
            #d = (2.12*2063.54)/(y2/0.3-649.19)

            #considering pitch and yaw
            #x2_s = x2/0.35
            #x1_s = x1/0.35
            #y2_s = (y2+32)/0.35
            #u = x1_s + ((x2_s-x1_s)/2)
            #v = y2_s
            u = x1+((x2-x1)/2)
            v = y2
            d = self.calcDistance(u,v, frameDim)

            return d
        else:
            return None
