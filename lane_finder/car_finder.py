import cv2


def findCars(model, frame, acceptedClasses):
    results = model(frame, verbose=False)[0]
    bBoxes = [] 
    #filter yolo results by class and confidence threshold
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if(class_id in acceptedClasses and score > 0.2):
            bBoxes.append((x1,y2,x2,y2))
            #draw car box in frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            
    return bBoxes