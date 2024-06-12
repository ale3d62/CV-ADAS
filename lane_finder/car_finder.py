import cv2
from concurrent.futures import ThreadPoolExecutor


#Returns the bboxes of the acceptedClasses found in the frame 
def findCars(model, frame, acceptedClasses, showBboxes):
    results = model(frame, verbose=False)[0]
    bBoxes = [] 
    #filter yolo results by class and confidence threshold
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if(class_id in acceptedClasses and score > 0.2):

            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            bBoxes.append((x1,y1,x2,y2))

            if(showBboxes):
                #draw car box in frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
    return bBoxes



#Updates the position of each bbox in bboxes
def findCarsPartial(model, frame, acceptedClasses, bBoxes, searchRegion):

    newBBoxes = []
    frameHeight, frameWidth, _ = frame.shape

    for bBox in bBoxes:
        x1,y1,x2,y2 = bBox

        #show original bBox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        bBoxW = x2-x1
        bBoxH = y2-y1

        #Get search region
        regionX1 = max(0, x1 - int(bBoxW*searchRegion))
        regionY1 = max(0, y1 - int(bBoxH*searchRegion))
        regionX2 = min(frameWidth, x2 + int(bBoxW*searchRegion))
        regionY2 = min(frameHeight, y2 + int(bBoxH*searchRegion))
        crop = frame[regionY1:regionY2, regionX1:regionX2]
        
        #show region bbox
        cv2.rectangle(frame, (regionX1, regionY1), (regionX2, regionY2), (150, 150, 150), 1)

    	#Search for the car in the region
        regionBoxes = findCars(model, crop, acceptedClasses, False)
        
        if(len(regionBoxes) < 1):
            continue

        #Convert the bBoxes coordinates from crop to frame
        for i in range(len(regionBoxes)):
            newX1,newY1,newX2,newY2 = regionBoxes[i]
            regionBoxes[i] = (newX1+regionX1, newY1+regionY1, newX2+regionX1, newY2+regionY1)

        #Filter if multiple cars found
        if(len(regionBoxes) <= 1):
            newBBoxes += regionBoxes
        else:
            #Choose the box closest to the original
            regionBoxesDistances = []
            for newBBox in regionBoxes:
                newX1,newY1,newX2,newY2 = newBBox
                regionBoxesDistances.append(abs(newX1-x1)+abs(newY1-y1)+abs(newX2-x2)+abs(newY2-y2))
            
            iClosestBbox = min(range(len(regionBoxesDistances)), key=regionBoxesDistances.__getitem__)
            newBBoxes.append(regionBoxes[iClosestBbox])

        #Show new bBox
        newX1, newY1, newX2, newY2 = regionBoxes[-1]
        cv2.rectangle(frame, (newX1, newY1), (newX2, newY2), (0, 0, 255), 1)

    return newBBoxes

