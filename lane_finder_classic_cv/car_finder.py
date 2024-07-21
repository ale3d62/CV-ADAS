import cv2

#Returns the bboxes of the acceptedClasses found in the frame 
def findCars(model, frame, acceptedClasses):
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

            bBox = (x1,y1,x2,y2)
            bBoxes.append(bBox)

            #draw car box in frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
    return bBoxes