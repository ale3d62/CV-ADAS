from types import NoneType
import cv2
import numpy as np
from time import time
from auxFunctions import *

def findLane(img):

    #CROP TO HALF THE HEIGHT
    imgHeight, imgWidth, _ = img.shape
    newImgHeight = round(imgHeight/2)
    croppedImg = img[newImgHeight:imgHeight, 1:imgWidth]

    #MASk
    vertices = np.array([[0, newImgHeight], [round(imgWidth*0.3), 0], [round(imgWidth*0.7), 0], [imgWidth, newImgHeight]], dtype=np.int32)
    mask = np.zeros_like(croppedImg)
    cv2.fillPoly(mask, [vertices], (255, 255, 255))
    masked_image = cv2.bitwise_and(croppedImg, mask)

    #LAB
    lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)

    #lower_white = np.array([0,0,200])
    #upper_white = np.array([360,50,255])

    #Channels: [Light, Green/Magenta, Blue/Yellow] 1-255 in all 3 channels
    #print(np.mean(lab[:,:,0]))
    lower_white = np.array([200, 1, 1])
    upper_white = np.array([255, 255, 255])

    mask = cv2.inRange(lab, lower_white, upper_white)
    colorMask = cv2.bitwise_and(masked_image,masked_image, mask= mask)

    #GAUSSIAN
    #blurred = cv2.GaussianBlur(src=colorMask, ksize=(3, 5), sigmaX=0.8) 


    #CANNY
    t_lower = 50
    t_upper = 300
    edges = cv2.Canny(colorMask, t_lower, t_upper, apertureSize=3, L2gradient=True)


    #HOUGH
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=50, maxLineGap=30)
    rgbEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    linesLeft = [[],[]]
    linesRight = [[],[]]

    if(type(lines) == NoneType):
        return edges

    for line in lines:
        arr = np.array(line[0], dtype=np.float64)
        x1,y1,x2,y2 = arr

        m = (y2-y1)/(x2-x1)
        b = y1 - m * x1


        #filter lines by angle
        lineAngle = abs(np.arctan(m))
    
        #not a line (60-120 deg or < 25 deg or > 155 deg)
        if((lineAngle > 1 and lineAngle < 2.2) or lineAngle < 0.43 or lineAngle > 2.7):
            continue


        xCutBottom = int((newImgHeight-b)/m)
        xCutTop = int(-b/m)
        
        #right line
        if(m < 0):
            linesRight[0].append(xCutBottom)
            linesRight[1].append(xCutTop)
            edges = drawLine2(rgbEdges, m, b)
        #left line
        else:
            linesLeft[0].append(xCutBottom)
            linesLeft[1].append(xCutTop)
            edges = drawLine2(rgbEdges, m, b)


    rightPoints = filterLinePoints(linesRight, 3)
    leftPoints = filterLinePoints(linesLeft, 3)


    try:
        xCoords,yCoords = zip(*leftPoints)
        lM,lB = np.polyfit(xCoords, yCoords, 1)
        edges = drawLine2(rgbEdges, lM, lB)

        xCoords,yCoords = zip(*rightPoints)
        rM,rB = np.polyfit(xCoords, yCoords, 1)
        edges = drawLine2(rgbEdges, rM, rB)
    except:
        return edges
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
    xCutLeft = round((lR-newImgHeight*np.sin(lTheta))/np.cos(lTheta))
    xCutRight = round((rR-newImgHeight*np.sin(rTheta))/np.cos(rTheta))
    newWidth = xCutRight-xCutLeft


    #GET ROI
    y = -(((round(newWidth/5))-(rR/np.cos(rTheta))+(lR/np.cos(lTheta)))/(np.tan(rTheta)-np.tan(lTheta)))
    y = round(y)

    croppedImg2 = croppedImg[y:newImgHeight, xCutLeft:xCutRight]
    croppedImg2Width = newWidth
    croppedImg2Height = newImgHeight-y

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

