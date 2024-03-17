from types import NoneType
import cv2
import numpy as np
#from sklearn.neighbors import NearestNeighbors
import sys


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



def filterLinePoints(linePoints, threshold):
    
    if(len(linePoints[0]) == 1):
        return linePoints
    linePointsBottom = np.array(linePoints[0])
    linePointsTop = np.array(linePoints[1])
    
    bottomMean = np.mean(linePointsBottom, axis=0)
    topMean = np.mean(linePointsTop, axis=0)
    
    bottomStd = np.std(linePointsBottom, axis=0)
    topStd = np.std(linePointsTop, axis=0)

    bottomDistances = np.abs((linePointsBottom - bottomMean) / np.maximum(bottomStd, 1e-8))
    topDistances = np.abs((linePointsTop - topMean) / np.maximum(topStd, 1e-8))

    pointsToDelete = np.logical_or(bottomDistances >= threshold, topDistances >= threshold)

    filteredBottom = linePointsBottom[~pointsToDelete]
    filteredTop = linePointsTop[~pointsToDelete]
    print(len(filteredBottom), len(filteredTop))
    return zip(filteredBottom, filteredTop)



def knn(points, k):    
    distances = np.abs(points[:, np.newaxis] - points)
    distancesIndex = np.argsort(distances, axis=1)[:, 1:k+1]
    kDistances = distances[np.arange(distancesIndex.shape[0])[:, None], distancesIndex]
    distancesMean = np.mean(kDistances, axis=1)
    
    return distancesMean



def getBestPoint_Debug(points, threshold, k):
    neighbours = knn(points, k)

    filteredPoints = points[neighbours <= threshold]
    if(len(filteredPoints) == 0):
        return None

    return neighbours >= threshold



def getBestLine_Debug(rgbEdges, linePoints, threshold, k, rec):

    minPoints = k
    if(len(linePoints[0]) == 0):
        return rgbEdges
    elif(len(linePoints[0]) < minPoints):
        return rgbEdges


    linePointsBottom = np.array(linePoints[0])
    linePointsTop = np.array(linePoints[1])
    
    bestLinePointBottom = getBestPoint_Debug(linePointsBottom, threshold, k)
    bestLinePointTop = getBestPoint_Debug(linePointsTop, threshold, k)
    newLinePoints = [[],[]]
    imgHeight, _, _ = rgbEdges.shape
    if(type(bestLinePointBottom) != NoneType and type(bestLinePointTop) != NoneType):
        
        for i in range(len(bestLinePointBottom)):
            if(bestLinePointBottom[i] or bestLinePointTop[i]):
                #pass
                if(rec):
                    cv2.line(rgbEdges, (linePointsTop[i], 0), (linePointsBottom[i], imgHeight), (100, 255,100), 2)
                else:
                    cv2.line(rgbEdges, (linePointsTop[i], 0), (linePointsBottom[i], imgHeight), (255, 100,100), 2)
            else:
                if(rec):
                    cv2.line(rgbEdges, (linePointsTop[i], 0), (linePointsBottom[i], imgHeight), (100, 100,255), 2)
                else:
                    newLinePoints[0].append(linePointsBottom[i])
                    newLinePoints[1].append(linePointsTop[i])

    if(rec):
        return rgbEdges
    else:
        return getBestLine_Debug(rgbEdges, newLinePoints, threshold, max(5, len(newLinePoints[0])), True)





def getPointsMask(points, threshold, k):
    neighbours = knn(points, k)

    return neighbours <= threshold



def getBestLine(linePoints, threshold, k, rec):

    minPoints = k
    if(len(linePoints[0]) == 0):
        return [None, None]
    elif(len(linePoints[0]) < minPoints):
        return [None, None]


    linePointsBottom = np.array(linePoints[0])
    linePointsTop = np.array(linePoints[1])
    
    filteredMaskBottom = getPointsMask(linePointsBottom, threshold, k)
    filteredMaskTop = getPointsMask(linePointsTop, threshold, k)
    
    mask = np.bitwise_and(filteredMaskBottom, filteredMaskTop)
    if(np.any(mask == True)):
        
        if(rec):
            bestPointBottom = np.mean(linePointsBottom[mask])
            bestPointTop = np.mean(linePointsTop[mask])
            return [int(bestPointBottom), int(bestPointTop)]
        else:
            return getBestLine([linePointsBottom[mask].tolist(), linePointsTop[mask].tolist()], threshold, max(5, len(mask)-mask.sum()), True)
    else:
        return [None, None]




def carInlane(x1,x2,y2, lx3, rx3, vpy, vpx, imgHeight):
    
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
    