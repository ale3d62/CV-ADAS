from enum import Enum
from math import sqrt
import numpy as np


class EstimationMethods(Enum):
    roadWidthEstimation = 1
    inverseProjection = 2


def roadWidthDistanceEstimation(frameDim,
                                roadLane,
                                bBox,
                                filterCarInLane,
                                roadWidth,
                                f,
                                sensorW,
                                heightCorrection,
                                cameraHeight):

    x1, y1, x2, y2 = bBox
    #use a bbox height 1% lower
    #y2 = int(y2*1.01)

    #coordinates x of the lines at the car's height
    imgHeight, imgWidth, _ = frameDim

    lx3, rx3 = roadLane.getLineCoords(y2, imgHeight)

    if not lx3 or not rx3:
        return None

    roadWidthPx = rx3-lx3

    if roadWidthPx == 0:
        return None

    if(filterCarInLane and not roadLane.carInLane(x1 + (x2-x1)/2, y2, imgHeight)):
        return None
    else:
        d = (roadWidth * f)/(sensorW * (roadWidthPx/imgWidth))

    #Apply height correction
    if(heightCorrection and d and d >= cameraHeight > 0 ):
        d = sqrt(d**2-cameraHeight**2)

    return d



def inverseProjectionDistanceEstimation(frameDim,
                                        roadLane,
                                        bBox,
                                        paddingTop,
                                        originalImgH,
                                        originalImgW,
                                        f_u,
                                        f_v,
                                        cameraPitch,
                                        cameraYaw,
                                        cameraHeight):

    x1, y1, x2, y2 = bBox
    imgHeight, imgWidth, _ = frameDim

    lx3, rx3 = roadLane.getLineCoords(y2, imgHeight)

    if not lx3 or not rx3:
        return None

    if(not roadLane.carInLane(x1 + (x2-x1)/2, y2, imgHeight)):
        return None
    else:
        u = x1+((x2-x1)/2)
        v = y2
        d = calcDistance(u,
                         v,
                         frameDim,
                         paddingTop,
                         originalImgH,
                         originalImgW,
                         f_u,
                         f_v,
                         cameraPitch,
                         cameraYaw,
                         cameraHeight)
        return d



def calcDistance(u,
                 v,
                 frameDim,
                 paddingTop,
                 originalImgH,
                 originalImgW,
                 f_u,
                 f_v,
                 cameraPitch,
                 cameraYaw,
                 cameraHeight):

    imgHeight, imgWidth, _ = frameDim

    imgHeight = imgHeight-(paddingTop*2)

    v = v-paddingTop
    v = v*originalImgH/imgHeight
    u = u*originalImgW/imgWidth

    f_u = f_u
    f_v = f_v
    c_u = originalImgW/2
    c_v = originalImgH/2
    pitch = cameraPitch
    yaw = cameraYaw
    height = cameraHeight

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

    d = S*z_w

    return d
