import cv2
from concurrent.futures import ThreadPoolExecutor

#Returns the bboxes of the acceptedClasses found in the frame 
def findCars(model, frame, acceptedClasses, doUpdateYoloRegions, yoloRegions, regionW, regionH):
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

            if(doUpdateYoloRegions):
                yoloRegions = updateYoloRegions(yoloRegions, bBox, regionW, regionH)

            #draw car box in frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
    return bBoxes, yoloRegions


def updateYoloRegions(yoloRegions, bBox, regionW, regionH):
    x1, y1, x2, y2 = bBox

    x1Region = int(x1 // regionW)
    y1Region = int(y1 // regionH)
    x2Region = int(x2 // regionW)
    y2Region = int(y2 // regionH)

    for i in range(x1Region, x2Region + 1):
        for j in range(y1Region, y2Region + 1):
            if 0 <= i < regionW and 0 <= j < regionH:
                yoloRegions[i][j] = True
    
    return yoloRegions




#Updates the position of each bbox in bboxes
def findCarsPartial(model, frame, acceptedClasses, bBoxes, yoloRegions, regionW, regionH):

    newBBoxes = []
    frameHeight, frameWidth, _ = frame.shape   
    
    def process_bbox(i, j):
        #Get search region
        regionX1 = int(frameWidth / len(yoloRegions)) * i
        regionX2 = regionX1+regionW
        regionY1 = int(frameHeight / len(yoloRegions[i])) * j
        regionY2 = regionY1+regionH
        crop = frame[regionY1:regionY2, regionX1:regionX2]

        #Search for cars in the region
        regionBoxes, _ = findCars(model, crop, acceptedClasses, False, yoloRegions, regionW, regionH)

        if(len(regionBoxes) == 0):
            yoloRegions[i][j] = False

        return regionBoxes


    with ThreadPoolExecutor() as executor:
        futures = []
        #For every region with a bbox
        for i in range(len(yoloRegions)):
            for j in range(len(yoloRegions[i])):
                if(yoloRegions[i][j] == True):
                    futures.append(executor.submit(process_bbox, i, j))
        for future in futures:
            newBBoxes.extend(future.result())
    
    return newBBoxes, yoloRegions