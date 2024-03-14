from types import NoneType
import cv2
import numpy as np
from time import time
from auxFunctions import *


def findLane(img):

    #CROP TO HALF THE HEIGHT
    #st = time() 135 480
    imgHeight, imgWidth, _ = img.shape
    halfImgHeight = int(imgHeight/2)
    img = img[halfImgHeight:imgHeight, 1:imgWidth]
    #print("Cropping time: " + str((time()-st)*1000) + "ms")


    #MASk
    #st = time()x1,y1,x2,y2
    vertices = np.array([[0, halfImgHeight], [round(imgWidth*0.3), 0], [round(imgWidth*0.7), 0], [imgWidth, halfImgHeight]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], (255, 255, 255))
    img = cv2.bitwise_and(img, mask)
    #print("Mask time: " + str((time()-st)*1000) + "ms")


    #LAB
    #st = time()
    lab = np.zeros_like(img)
    cv2.cvtColor(img, cv2.COLOR_BGR2LAB, lab)

    #Channels: [Light, Green/Magenta, Blue/Yellow] 1-255 in all 3 channels
    #print(np.mean(lab[:,:,0]))
    lower_white = np.array([200, 1, 1])
    upper_white = np.array([255, 255, 255])

    mask = cv2.inRange(lab, lower_white, upper_white)
    
    colorMask = cv2.bitwise_and(img,img, mask= mask)
    #print("LAB time: " + str((time()-st)*1000) + "ms")

    #colorMask = cv2.morphologyEx(lab[:,:,0], cv2.MORPH_TOPHAT, kernel)
    #mask = cv2.adaptiveThreshold(colorMask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)


    #GAUSSIAN
    #blurred = cv2.GaussianBlur(src=colorMask, ksize=(3, 5), sigmaX=0.8) 


    #OPEN
    #st = time()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    colorMask = cv2.morphologyEx(colorMask, cv2.MORPH_OPEN, kernel)
    #print("OPEN time: " + str((time()-st)*1000) + "ms")


    #CANNY
    #st = time()
    t_lower = 50
    t_upper = 300

    #edges = np.zeros((halfImgHeight, imgWidth), dtype=np.uint8)
    #cv2.Canny(colorMask, t_lower, t_upper, edges, apertureSize=3, L2gradient=True)
    edges = cv2.Canny(colorMask, t_lower, t_upper, apertureSize=3, L2gradient=True)
    #print("CANNY time: " + str((time()-st)*1000) + "ms")


    #HOUGH
    #st = time()
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=50, maxLineGap=30)
    #rgbEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    #print("HOUGH time: " + str((time()-st)*1000) + "ms")

    #st = time()
    if(type(lines) == NoneType):
        return ((None, None), (None,None))

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
            #edges = drawLine2(rgbEdges, m, b)
        else:       #left line
            if(xCutBottom > imgWidth*0.35):
                continue
            linesLeft[0].append(xCutBottom)
            linesLeft[1].append(xCutTop)
            #edges = drawLine2(rgbEdges, m, b)
    #print("LINE PROCESSING time: " + str((time()-st)*1000) + "ms")
    #print("time: " + str((time()-st)*1000) + "ms")
    #bestLinePointsLeft = getBestLine_Debug(rgbEdges, linesLeft, 30, max(5, int(len(linesRight))), False)
    #bestLinePointsRight = getBestLine_Debug(rgbEdges, linesRight, 30, max(5, int(len(linesRight))), False)
    #return rgbEdges
    #st = time()
    bestLinePointsLeft = getBestLine(linesLeft, 30, max(5, int(len(linesRight))), False)
    bestLinePointsRight = getBestLine(linesRight, 30, max(5, int(len(linesRight))), False)
    #print("BEST LINE time: " + str((time()-st)*1000) + "ms")

    return (bestLinePointsLeft,bestLinePointsRight)



    """
    roiXl = 863#575
    roiXr = 1050#700
    #575 415
    #700 415 
    #0 heigh
    #width height
    #WARP
    inputPts = np.float32([[roiXl, 83], [roiXr, 83], [0, halfImgHeight], [imgWidth, halfImgHeight]])

    warpedWidth = imgWidth
    warpedHeight = round(np.sqrt(pow(roiXl, 2) + pow(halfImgHeight, 2)))
    #print(warpedWidth)
    #print(warpedHeight)

    outputPts = np.float32([[0, 0], [warpedWidth, 0], [0, warpedHeight], [warpedWidth, warpedHeight]])

    M = cv2.getPerspectiveTransform(inputPts,outputPts)

    warpedImg = cv2.warpPerspective(edges, M, (warpedWidth, warpedHeight))

    return warpedImg
    """
    """
    #Parameters: (img, distance_resolution, angle_resolution, accumulator_threshold, )
    lines = cv2.HoughLines(edges, 1, np.pi/180, 30, min_theta=0.35, max_theta = 2.6)

    linesLeft = [[],[]]
    linesRight = [[],[]]

    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr

        #filter mask borders
        if(r <-580 or (r>256 and r<270)):
            continue
        
        #filter lines
        if(theta > 1.04 and theta < 2):
            continue

        if(theta > 1.5):
            linesRight[0].append(r)
            linesRight[1].append(theta)
            croppedImg = drawLine(croppedImg, r, theta)
        else:
            linesLeft[0].append(r)
            linesLeft[1].append(theta)
            croppedImg = drawLine(croppedImg, r, theta)
            
    
    try:
        #Left lines avg
        lR = sum(linesLeft[0]) / len(linesLeft[0])
        lTheta = sum(linesLeft[1]) / len(linesLeft[1])
        croppedImg = drawLine(croppedImg, lR, lTheta)

        #Right lines avg
        rR = sum(linesRight[0]) / len(linesRight[0])
        rTheta = sum(linesRight[1]) / len(linesRight[1])
        croppedImg = drawLine(croppedImg, rR, rTheta)
    except:
        return
    """
    
    return edges
    """
    #CROP BASED ON LINES
    xCutLeft = round((lR-halfImgHeight*np.sin(lTheta))/np.cos(lTheta))
    xCutRight = round((rR-halfImgHeight*np.sin(rTheta))/np.cos(rTheta))
    newWidth = xCutRight-xCutLeft


    #GET ROI
    y = -(((round(newWidth/5))-(rR/np.cos(rTheta))+(lR/np.cos(lTheta)))/(np.tan(rTheta)-np.tan(lTheta)))
    y = round(y)

    croppedImg2 = croppedImg[y:halfImgHeight, xCutLeft:xCutRight]
    croppedImg2Width = newWidth
    croppedImg2Height = halfImgHeight-y

    roiXl = round((lR - y*np.sin(lTheta))/np.cos(lTheta))-xCutLeft
    roiXr = round((rR - y*np.sin(rTheta))/np.cos(rTheta))-xCutLeft


    #WARP
    inputPts = np.float32([[roiXl, 0], [roiXr, 0], [0, croppedImg2Height], [croppedImg2Width, croppedImg2Height]])

    warpedWidth = croppedImg2Width
    warpedHeight = round(np.sqrt(pow(roiXl, 2) + pow(croppedImg2Height, 2)))
    #print(warpedWidth)
    #print(warpedHeight)

    outputPts = np.float32([[0, 0], [warpedWidth, 0], [0, warpedHeight], [warpedWidth, warpedHeight]])

    M = cv2.getPerspectiveTransform(inputPts,outputPts)

    #warpedImg = cv2.warpPerspective(croppedImg2, M, (warpedWidth, warpedHeight))
    """


if __name__ == "__main__":
    IMG_PATH = "test_images/test11.jpg";

    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    st = time()
    img = findLane(img)

    #RESULTS
    print("time: " + str((time()-st)*1000) + "ms")
    cv2.imshow("image", img)
    cv2.waitKey(0)

