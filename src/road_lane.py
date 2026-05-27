NoneType = type(None)
import cv2
from estimation_methods import EstimationMethods
import numpy as np
from auxFunctions import getBestLine


class RoadLane():

    def __init__(self,
                 _minLineWidth):
        self.linePointsLeft = (None, None)
        self.linePointsRight = (None, None)
        self.laneMask = None
        self.minLineWidth = _minLineWidth



    #Returns the closest pixel of the mask to both sides of the x,y point at y height in both sides
    def getLinesCoords(self, x, y, frameDim):
        imgHeight, imgWidth, _ = frameDim

        if y < 0 or y >= imgHeight:
            return (None, None)

        lx3 = rx3 = None

        #left line
        lx = x
        while lx > 0 and lx3 == None:
            if self.laneMask[y][lx] == 0:
                lx -= (self.minLineWidth-1)
            else:
                while lx < imgWidth and self.laneMask[y][lx] == 1:
                    lx+=1
                lx3 = lx + 1

        if(lx3 != None):
            lx -=1
            while(lx>0 and self.laneMask[y][lx] == 1):
                lx -=1
            lx3 = lx3 - ((lx3-lx)/2)

        #right line
        rx = x
        while rx > 0 and rx < imgWidth and rx3 == None:
            if self.laneMask[y][rx] == 0:
                rx += (self.minLineWidth-1)
            else:
                while self.laneMask[y][rx] == 1:
                    rx-=1
                rx3 = rx - 1

        if(rx3 != None):
            rx+=1
            while(rx < imgWidth and self.laneMask[y][rx] == 1):
                rx +=1
            rx3 = rx3 + ((rx-rx3)/2)

        return (lx3, rx3)



    def showLane(self, estimationMethod, frame):

        if(estimationMethod == EstimationMethods.roadWidthEstimation):
            frame[self.laneMask==1] = (30, 255, 15)

        elif(estimationMethod == EstimationMethods.inverseProjection):
            imgHeight, _, _ = frame.shape
            halfImgHeight = int(imgHeight / 2)

            if(self.linePointsLeft[0] and self.linePointsLeft[1]):
                cv2.line(frame,
                            (self.linePointsLeft[1], halfImgHeight),
                            (self.linePointsLeft[0], imgHeight),
                            (0, 0, 255), 2)

            if(self.linePointsRight[0] and self.linePointsRight[1]):
                cv2.line(frame,
                            (self.linePointsRight[1], halfImgHeight),
                            (self.linePointsRight[0], imgHeight),
                            (0, 0, 255), 2)

        return frame


    def getCleanLaneMask(self):
        mask2D = np.squeeze(self.laneMask)

        if np.max(mask2D) <= 1:
            mask2D = (mask2D * 255)

        thinnedMax = cv2.ximgproc.thinning(mask2D,
                                             thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        return thinnedMax



    def houghFiltering(self, cleanLaneMask):

        imgHeight, imgWidth = cleanLaneMask.shape
        halfImgHeight = int(imgHeight/2)
        halfImg = cleanLaneMask[halfImgHeight:imgHeight, 1:imgWidth]

        lines = cv2.HoughLinesP(halfImg, 1, np.pi/180, 30, minLineLength=50, maxLineGap=30)

        if(type(lines) == NoneType):
            return

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
            #rightLine
            if(m > 0):
                if(xCutBottom < imgWidth*0.55):
                    continue

                linesRight[0].append(xCutBottom)
                linesRight[1].append(xCutTop)

            #left line
            else:
                if(xCutBottom > imgWidth*0.45):
                    continue

                linesLeft[0].append(xCutBottom)
                linesLeft[1].append(xCutTop)

        newBestLinePointsLeft = getBestLine(linesLeft,
                                            30,
                                            max(2, int(len(linesRight))),
                                            False)

        newBestLinePointsRight = getBestLine(linesRight,
                                             30,
                                             max(2, int(len(linesRight))),
                                             False)

        if(newBestLinePointsLeft[0]):
            self.linePointsLeft = newBestLinePointsLeft

        if(newBestLinePointsRight[0]):
            self.linePointsRight = newBestLinePointsRight



    def estimateVanishingPoint(self):

        cleanLaneMask = self.getCleanLaneMask()
        self.houghFiltering(cleanLaneMask)

        return getVanishingPoint(self.linePointsLeft,
                                 self.linePointsRight,
                                 cleanLaneMask.shape[0])



def getVanishingPoint(lineLeft, lineRight, height):

    if(lineLeft[0] == None or lineRight[0] == None):
        return None

    x_bottom1, x_top1 = lineLeft
    x_bottom2, x_top2 = lineRight

    H = height - 1

    m1 = (x_bottom1 - x_top1) / H
    m2 = (x_bottom2 - x_top2) / H

    #Check that the lines are not parallel
    if abs(m1 - m2) < 1e-6:
        return None

    y_vp = (x_top2 - x_top1) / (m1 - m2)
    x_vp = x_top1 + m1 * y_vp

    return (x_vp, y_vp + height/2)
