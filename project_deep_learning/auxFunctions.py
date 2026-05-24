import cv2
import numpy as np



def alert():
    print("VEHICLE TOO CLOSE")



def resizeFrame(imagen, target_w=672, target_h=384):
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



def scaleRoadLinePoints(linePoints, frame, target_w=672, target_h=384):
    frameH, frameW, _ = frame.shape

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



def showRoadLines(frame, linePointsLeft, linePointsRight):
    imgHeight, _, _ = frame.shape
    halfImgHeight = int(imgHeight/2)

    if(linePointsLeft[0] and linePointsLeft[1]):
        cv2.line(frame, (linePointsLeft[1], halfImgHeight), (linePointsLeft[0], imgHeight), (0, 0, 255), 2)
    if(linePointsRight[0] and linePointsRight[1]):
        cv2.line(frame, (linePointsRight[1], halfImgHeight), (linePointsRight[0], imgHeight), (0, 0, 255), 2)

    return frame