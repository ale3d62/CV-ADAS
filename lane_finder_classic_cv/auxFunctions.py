import numpy as np

NoneType = type(None)

#Returs true if the system can keep processing the input, false otherwise
def canProcessVideo(inputVideos, videoSource):
    if(videoSource == "screen" or videoSource == "camera"):
        return True
    elif(videoSource == "video"):
        return len(inputVideos) > 0
    else:
        return False
    



def knn(points, k):    
    distances = np.abs(points[:, np.newaxis] - points)
    distancesIndex = np.argsort(distances, axis=1)[:, 1:k+1]
    kDistances = distances[np.arange(distancesIndex.shape[0])[:, None], distancesIndex]
    distancesMean = np.mean(kDistances, axis=1)
    
    return distancesMean



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