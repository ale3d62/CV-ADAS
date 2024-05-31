import cv2


#Returns the bboxes of the acceptedClasses found in the frame 
def findCars(model, frame, acceptedClasses):
    results = model(frame, verbose=False)[0]
    bBoxes = [] 
    #filter yolo results by class and confidence threshold
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if(class_id in acceptedClasses and score > 0.2):
            bBoxes.append((x1,y1,x2,y2))
            #draw car box in frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            
    return bBoxes



#Returns the bboxes of the acceptedClasses found near the given bBoxes
def findCarsPartial(model, frame, acceptedClasses, bBoxes, searchRegion):

    newBBoxes = []
    frameHeight, frameWidth, _ = frame.shape

    for bBox in bBoxes:
        x1,y1,x2,y2 = bBox
        bBoxW = x2-x1
        bBoxH = y2-y1

        #Get search region
        newX1 = max(0, x1 - (bBoxW*searchRegion))
        newY1 = max(0, y1 - (bBoxH*searchRegion))
        newX2 = min(frameWidth, x2 + (bBoxW*searchRegion))
        newY2 = min(frameHeight, y2 + (bBoxH*searchRegion))
        crop = frame[int(newY1):int(newY2), int(newX1):int(newX2)]

        #Search for cars
        newBBoxes += findCars(model, crop, acceptedClasses)

    return newBBoxes

