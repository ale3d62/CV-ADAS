import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors



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



def getBestLine(linePoints, threshold):

    minPoints = 6
    if(len(linePoints[0]) == 0):
        return None
    elif(len(linePoints[0]) < minPoints):
        return None
        meanBottom = sum(linePoints[0]) / len(linePoints[0])
        meanTop = sum(linePoints[1]) / len(linePoints[1])
        return (int(meanBottom), int(meanTop))
    

    linePointsBottom = [[x] for x in linePoints[0]]
    linePointsTop = [[x] for x in linePoints[1]]

    # Entrena el modelo de detección de anomalías
    detector_anomalías = NearestNeighbors(n_neighbors=5) # Puedes ajustar el número de vecinos según tu necesidad
    detector_anomalías.fit(linePointsBottom)
    # Calcula las distancias a los k vecinos más cercanos para cada punto
    distancias, _ = detector_anomalías.kneighbors()
    # Calcula el promedio de las distancias a los k vecinos más cercanos
    distancias_promedio = distancias.mean(axis=1)
    # Encuentra los puntos que están por encima del umbral de anomalía
    puntos_anómalos = [linePointsBottom[i][0] for i, distancia_promedio in enumerate(distancias_promedio) if distancia_promedio > threshold]
    # Elimina los puntos anómalos de la lista de coordenadas
    coordenadas_filtradas = [x[0] for x in linePointsBottom if x[0] not in puntos_anómalos]
    if(len(coordenadas_filtradas) == 0):
        return None
    # El valor predicho será el promedio de los puntos restantes
    valor_predichobot = sum(coordenadas_filtradas) / len(coordenadas_filtradas)

    # Entrena el modelo de detección de anomalías
    detector_anomalías = NearestNeighbors(n_neighbors=5) # Puedes ajustar el número de vecinos según tu necesidad
    detector_anomalías.fit(linePointsTop)
    # Calcula las distancias a los k vecinos más cercanos para cada punto
    distancias, _ = detector_anomalías.kneighbors()
    # Calcula el promedio de las distancias a los k vecinos más cercanos
    distancias_promedio = distancias.mean(axis=1)
    # Encuentra los puntos que están por encima del umbral de anomalía
    puntos_anómalos = [linePointsTop[i][0] for i, distancia_promedio in enumerate(distancias_promedio) if distancia_promedio > threshold]
    # Elimina los puntos anómalos de la lista de coordenadas
    coordenadas_filtradas = [x[0] for x in linePointsTop if x[0] not in puntos_anómalos]
    if(len(coordenadas_filtradas) == 0):
        return None
    # El valor predicho será el promedio de los puntos restantes
    valor_predichotop = sum(coordenadas_filtradas) / len(coordenadas_filtradas)

    return (int(valor_predichobot), int(valor_predichotop))



    
