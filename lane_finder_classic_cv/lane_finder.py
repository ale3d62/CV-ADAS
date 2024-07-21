from types import NoneType
import cv2
import numpy as np
from auxFunctions import *



def show_lines(img, bestLinePointsLeft, bestLinePointsRight):
    imgHeight, _, _ = img.shape
    halfImgHeight = int(imgHeight/2)

    if(bestLinePointsLeft[0] and bestLinePointsLeft[1]):
        cv2.line(img, (bestLinePointsLeft[1], halfImgHeight), (bestLinePointsLeft[0], imgHeight), (0, 0, 255), 2)
    if(bestLinePointsRight[0] and bestLinePointsRight[1]):
        cv2.line(img, (bestLinePointsRight[1], halfImgHeight), (bestLinePointsRight[0], imgHeight), (0, 0, 255), 2)
    
    return img



def findLane(img, bestLinePointsLeft, bestLinePointsRight, showLines):

    linesUpdated = False
    #CROP TO HALF THE HEIGHT
    imgHeight, imgWidth, _ = img.shape
    halfImgHeight = int(imgHeight/2)
    halfImg = img[halfImgHeight:imgHeight, 1:imgWidth]


    #MASk
    vertices = np.array([[0, halfImgHeight], [round(imgWidth*0.3), 0], [round(imgWidth*0.7), 0], [imgWidth, halfImgHeight]], dtype=np.int32)
    mask = np.zeros_like(halfImg)
    cv2.fillPoly(mask, [vertices], (255, 255, 255))
    halfImg = cv2.bitwise_and(halfImg, mask)


    #LAB
    lab = np.zeros_like(halfImg)
    cv2.cvtColor(halfImg, cv2.COLOR_BGR2LAB, lab)

    #Channels: [Light, Green/Magenta, Blue/Yellow] 1-255 in all 3 channels
    lower_white = np.array([200, 1, 1])
    upper_white = np.array([255, 255, 255])

    mask = cv2.inRange(lab, lower_white, upper_white)
    
    colorMask = cv2.bitwise_and(halfImg,halfImg, mask= mask)


    #GAUSSIAN
    #blurred = cv2.GaussianBlur(src=colorMask, ksize=(3, 5), sigmaX=0.8) 


    #OPEN
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    colorMask = cv2.morphologyEx(colorMask, cv2.MORPH_OPEN, kernel)


    #CANNY
    t_lower = 50
    t_upper = 300

    edges = cv2.Canny(colorMask, t_lower, t_upper, apertureSize=3, L2gradient=True)


    #HOUGH
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=50, maxLineGap=30)

    if(type(lines) == NoneType):
        if(showLines):
            img = show_lines(img, bestLinePointsLeft, bestLinePointsRight)
        return (img, bestLinePointsLeft, bestLinePointsRight, linesUpdated)

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
        if((xCutBottom < imgWidth*0.01 and lineAngle > maskAngle-maskAngle*0.1 and lineAngle < maskAngle+maskAngle*0.1) or
            xCutBottom > imgWidth*0.99 and lineAngle > maskAngle-maskAngle*0.1 and lineAngle < maskAngle+maskAngle*0.1):
            continue

        #Classify left and right
        if(m > 0):  #rightLine
            if(xCutBottom < imgWidth*0.65):
                continue
            linesRight[0].append(xCutBottom)
            linesRight[1].append(xCutTop)
        else:       #left line
            if(xCutBottom > imgWidth*0.35):
                continue
            linesLeft[0].append(xCutBottom)
            linesLeft[1].append(xCutTop)


    newBestLinePointsLeft = getBestLine(linesLeft, 30, max(5, int(len(linesRight))), False)
    newBestLinePointsRight = getBestLine(linesRight, 30, max(5, int(len(linesRight))), False)

    if(newBestLinePointsLeft[0]):
        bestLinePointsLeft = newBestLinePointsLeft
    if(newBestLinePointsRight[0]):
        bestLinePointsRight = newBestLinePointsRight


    if(newBestLinePointsLeft[0] and newBestLinePointsRight[0]):
        linesUpdated = True

    if(showLines):
        img = show_lines(img, bestLinePointsLeft, bestLinePointsRight)

    return (img, bestLinePointsLeft,bestLinePointsRight, linesUpdated)