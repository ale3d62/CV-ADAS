NoneType = type(None)
import cv2
from estimation_methods import EstimationMethods
import numpy as np
from auxFunctions import getBestLine


class RoadLane():

    def __init__(self,
                 _minLineWidth,
                 _originalImageShape,
                 _paddingTop):
        self.linePointsLeft = (None, None)
        self.linePointsRight = (None, None)
        self.laneMask = None
        self.minLineWidth = _minLineWidth
        self.originalImgH, self.originalImgW, _ = _originalImageShape
        self.paddingTop = _paddingTop



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

        thinnedMax = self.dynamicLaneMaskCleaning(thinnedMax)

        return thinnedMax



    #Clears most lines that do not belong to the road lane
    def dynamicLaneMaskCleaning(self, thinned_mask):

        height, width = thinned_mask.shape

        #Assign an ID to each pixel group
        num_labels, labels = cv2.connectedComponents(thinned_mask, connectivity=8)

        #Define ROI (Lower 10% and central 80%)
        y_start = int(height * 0.9)
        x_start = int(width * 0.2)
        x_end = int(width * 0.8)

        region = labels[y_start:height, x_start:x_end]

        validIds = np.unique(region)

        #Remove 0 from the list (since 0 represents the black background)
        validIds = validIds[validIds != 0]

        cleaned_mask = np.zeros_like(thinned_mask)

        if len(validIds) > 0:
            cleaned_mask[np.isin(labels, validIds)] = 255

        return cleaned_mask



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



    def estimateVanishingPoint(self, imgH):

        cleanLaneMask = self.getCleanLaneMask()
        self.houghFiltering(cleanLaneMask)

        return self.getVanishingPoint(imgH)



    def scaleVanishingPoint(self, vp_resized, target_h, target_w):
        x, y = vp_resized

        h, w = self.originalImgH, self.originalImgW
        target_ratio = target_w / target_h
        current_ratio = w / h

        #Vertical padding
        if current_ratio > target_ratio:
            #Height after padding
            new_h = w / target_ratio
            padding_total = new_h - h
            pad_top = padding_total / 2

            #Undo resize
            scale_y = new_h / target_h
            scale_x = w / target_w

            x_original = x * scale_x
            y_original = y * scale_y

            #Undo padding
            y_original -= pad_top

        #Center crop
        elif current_ratio < target_ratio:
            crop_h = w / target_ratio
            start_y = (h - crop_h) / 2

            #Undo resize
            scale_y = crop_h / target_h
            scale_x = w / target_w

            x_original = x * scale_x
            y_original = y * scale_y

            #Undo crop
            y_original += start_y

        #Same aspect ratio
        else:
            scale_x = w / target_w
            scale_y = h / target_h

            x_original = x * scale_x
            y_original = y * scale_y

        return (x_original, y_original)



    def scaleRoadLinePoints(self, linePoints, frameSize, target_w=672, target_h=384):

        if(linePoints[0] == None or linePoints[1] == None):
                return linePoints

        frameH, frameW = frameSize

        target_ratio = target_w / target_h
        current_ratio = frameW / frameH

        scale_x = target_w / frameW
        scale_y_inv = frameW / target_w

        if current_ratio > target_ratio:
            #Image too wide
            new_h = int(frameW / target_ratio)
            padding_total = new_h - frameH
            pad_top = padding_total // 2

            y_bottom_orig = target_h * scale_y_inv - pad_top
            y_top_orig = (target_h / 2) * scale_y_inv - pad_top

        elif current_ratio < target_ratio:
            #Image too tall
            crop_h = int(frameW / target_ratio)
            start_y = (frameH - crop_h) // 2

            y_bottom_orig = target_h * scale_y_inv + start_y
            y_top_orig = (target_h / 2) * scale_y_inv + start_y

        else:
            #Same aspect ratio
            y_bottom_orig = frameH
            y_top_orig = frameH / 2

        x_cut_bottom, x_cut_top = linePoints

        dx = x_cut_bottom - x_cut_top
        dy = frameH / 2  # h - h/2

        x_orig_bottom = x_cut_top + (dx / dy) * (y_bottom_orig - frameH / 2)
        x_orig_top = x_cut_top + (dx / dy) * (y_top_orig - frameH / 2)

        new_x_cut_bottom = int(x_orig_bottom * scale_x)
        new_x_cut_top = int(x_orig_top * scale_x)

        return (new_x_cut_bottom, new_x_cut_top)


    def getLineCoords(self, y, imgH):
        if(self.linePointsLeft[0] == None or self.linePointsRight[0] == None):
            return (None, None)

        y_bottom = imgH
        y_top = imgH / 2

        def lane_x_at_y(lane, y):
            x_bottom, x_top = lane

            t = (y - y_bottom) / (y_top - y_bottom)
            return x_bottom + t * (x_top - x_bottom)

        x_left = lane_x_at_y(self.linePointsLeft, y)
        x_right = lane_x_at_y(self.linePointsRight, y)

        return (x_left, x_right)



    def carInLane(self, x, y, imgH):
        y_bottom = imgH
        y_top = imgH / 2

        def lane_x_at_y(lane, y):
            x_bottom, x_top = lane

            t = (y - y_bottom) / (y_top - y_bottom)
            return x_bottom + t * (x_top - x_bottom)

        x_left = lane_x_at_y(self.linePointsLeft, y)
        x_right = lane_x_at_y(self.linePointsRight, y)

        return x_left <= x <= x_right



    def getVanishingPoint(self, height):

        if(self.linePointsLeft[0] == None or self.linePointsRight[0] == None):
            return None

        x_bottom1, x_top1 = self.linePointsLeft
        x_bottom2, x_top2 = self.linePointsRight

        H = height - 1

        m1 = (x_bottom1 - x_top1) / H
        m2 = (x_bottom2 - x_top2) / H

        #Check that the lines are not parallel
        if abs(m1 - m2) < 1e-6:
            return None

        y_vp = (x_top2 - x_top1) / (m1 - m2)
        x_vp = x_top1 + m1 * y_vp

        return (x_vp, y_vp + height/2)
