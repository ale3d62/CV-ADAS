import cv2
import numpy as np



def alert():
    print("VEHICLE TOO CLOSE")



def resizeFrame(imagen, resizedFrameSize):
    target_h, target_w = resizedFrameSize
    h, w = imagen.shape[:2]

    target_ratio = target_w / target_h
    current_ratio = w / h

    #Image too wide: black bars top and bottom
    if current_ratio > target_ratio:

        new_h = int(w / target_ratio)

        padding_total = new_h - h
        pad_top = padding_total // 2
        pad_bottom = padding_total - pad_top

        imagen = cv2.copyMakeBorder(
            imagen,
            pad_top,
            pad_bottom,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    #Image too tall: center crop
    elif current_ratio < target_ratio:

        crop_h = int(w / target_ratio)

        start_y = (h - crop_h) // 2

        end_y = start_y + crop_h

        new_h = end_y-start_y
        pad_top = -start_y

        imagen = imagen[start_y:end_y, :]

    #Final resize
    imagen = cv2.resize(
        imagen,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR
    )

    pad_top = pad_top*target_h/new_h

    return imagen, pad_top



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
            return getBestLine([linePointsBottom[mask].tolist(), linePointsTop[mask].tolist()], threshold, max(k, len(mask)-mask.sum()), True)
    else:
        return [None, None]



def knn(points, k):
    distances = np.abs(points[:, np.newaxis] - points)
    distancesIndex = np.argsort(distances, axis=1)[:, 1:k+1]
    kDistances = distances[np.arange(distancesIndex.shape[0])[:, None], distancesIndex]
    distancesMean = np.mean(kDistances, axis=1)

    return distancesMean



def getPointsMask(points, threshold, k):
    neighbours = knn(points, k)

    return neighbours <= threshold
