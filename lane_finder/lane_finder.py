import cv2
import numpy as np
from time import time
from auxFunctions import *

IMG_PATH = "test_images/test4.jpg";

img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
st = time()

#CROP TO HALF THE HEIGHT
imgHeight, imgWidth, _ = img.shape
newImgHeight = round(imgHeight/2)
croppedImg = img[newImgHeight:imgHeight, 1:imgWidth]

#MASk
vertices = np.array([[0, newImgHeight], [round(imgWidth*0.3), 0], [round(imgWidth*0.7), 0], [imgWidth, newImgHeight]], dtype=np.int32)
mask = np.zeros_like(croppedImg)
cv2.fillPoly(mask, [vertices], (255, 255, 255))
masked_image = cv2.bitwise_and(croppedImg, mask)

#HLS
hls = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HLS)

#GRAYSCALE
gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY) 

#GAUSSIAN
blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.8) 

#CANNY
t_lower = 50
t_upper = 150

edges = cv2.Canny(blurred, t_lower, t_upper, apertureSize=3)


#HOUGH
lines = cv2.HoughLines(edges, 1, np.pi/180, 80, min_theta=0.35, max_theta = 2.6)
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
        

#Left lines avg
lR = sum(linesLeft[0]) / len(linesLeft[0])
lTheta = sum(linesLeft[1]) / len(linesLeft[1])
croppedImg = drawLine(croppedImg, lR, lTheta)

#Right lines avg
rR = sum(linesRight[0]) / len(linesRight[0])
rTheta = sum(linesRight[1]) / len(linesRight[1])
croppedImg = drawLine(croppedImg, rR, rTheta)


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


#RESULTS
print("time: " + str((time()-st)*1000) + "ms")
cv2.imshow("image", croppedImg2)
#cv2.imshow("borders", edges)
cv2.waitKey(0)