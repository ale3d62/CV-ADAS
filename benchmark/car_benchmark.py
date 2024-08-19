import json
import cv2
from ultralytics import YOLO
from progressBar import printProgress
import os.path

#-------------PARAMETERS--------------------------
DATASET_PATH = 'datasets/car/bdd10k/'
DATASET_JSON_NAME = 'bdd10k_labels_images_train.json'
MODEL_PATH = '../models/'
MODEL_NAME = 'yolov8n.pt'
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
IOU_COMPARE_THRESHOLD = 0.5
MODEL_IMG_SIZE = 640
PREVIEW_MODE = False
PREVIEW_TIME = 500 #ms
ACCEPTED_CLASSES = set(["car", "truck", "bus", "motorcycle"])
ACCEPTED_CLASSES_YOLO = set([2, 3, 4, 6, 7])
#CLASSES
#1: pedestrian
#2: rider
#3: car
#4: truck
#5: bus
#6: train
#7: motorcycle
#8: bicycle
#9: traffic light
#10: traffic sign
#-------------------------------------------------

def calculateIou(bbox1, bbox2):

        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        interArea = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1Area = (x2 - x1) * (y2 - y1)
        bbox2Area = (x4 - x3) * (y4 - y3)
        unionArea = bbox1Area + bbox2Area - interArea

        iou = interArea / unionArea

        return iou



#load model
model = YOLO(MODEL_PATH + MODEL_NAME)

#load labels
print("Loading labels...")
labels = json.load(open(DATASET_PATH + DATASET_JSON_NAME))

nImg = len(labels)
totalDet = 1
detHits = 1
newLabels = []

for iImg, label in enumerate(labels):

    img = cv2.imread(DATASET_PATH+"train/"+label['name'])
    pred = model.predict(source=img, imgsz=MODEL_IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    #filter real bboxes
    realBoxes = []
    for detection in label['labels']:
         if detection['category'] in ACCEPTED_CLASSES:
              box = detection['box2d']
              x1 = int(box['x1'])
              y1 = int(box['y1'])
              x2 = int(box['x2'])
              y2 = int(box['y2'])
              realBoxes.append((x1,y1,x2,y2))
    
    #load detected bboxes
    detectedBoxes = []
    for box in pred[0].boxes.data.tolist():

        x1, y1, x2, y2, score, class_id = box
        
        if(class_id in ACCEPTED_CLASSES_YOLO):

            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            detectedBoxes.append((x1,y1,x2,y2))
    
    #calculate hits
    for iBox, box in enumerate(realBoxes):
        ious = []

        if PREVIEW_MODE:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for detectedBox in detectedBoxes:
            ious.append(calculateIou(box, detectedBox))
    
        maxIou = max(ious) if len(ious) > 0 else 0
        
        if maxIou > IOU_COMPARE_THRESHOLD:
            detHits += 1

            iMaxIou = ious.index(maxIou)

            if PREVIEW_MODE:
                x1, y1, x2, y2 = detectedBoxes[iMaxIou]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            detectedBoxes.pop(iMaxIou)
        
        totalDet += 1
        realBoxes.pop(iBox)

    if(PREVIEW_MODE):
        cv2.imshow("Image Preview", img)
        cv2.waitKey(PREVIEW_TIME)
    
    printProgress(iImg, nImg)



print("")
print(f"Benchmark finished, [{detHits}/{totalDet}] {(detHits/totalDet)*100:.2f}%")    
