import cv2
import numpy as np

def drawLine(img, r, theta):
    imgHeight, _, _ = img.shape
    xCutBottom = int((r-imgHeight*np.sin(theta))/np.cos(theta))
    xCutTop = int(r/np.cos(theta))
    cv2.line(img, (xCutTop, 0), (xCutBottom, imgHeight), (0, 0, 255), 2)
    return img


def drawLine2(img, m, b):
    imgHeight, _, _ = img.shape
    xCutBottom = int((imgHeight-b)/m)
    xCutTop = int(-b/m)
    cv2.line(img, (xCutTop, 0), (xCutBottom, imgHeight), (0, 0, 255), 2)
    return img