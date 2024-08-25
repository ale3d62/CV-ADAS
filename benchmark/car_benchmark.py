import json
import cv2
from progressBar import printProgress
import sys
import numpy as np
from time import time
import torch
import os
from yolopv2 import detect

#-------------PARAMETERS--------------------------
DATASET_PATH = 'datasets/car/bdd100k/'
DATASET_JSON_NAME = 'bdd100k_labels_images_val.json'
MODEL_PATH = '../models/'
MODEL_NAME = 'yolopv2.pt'
#DETECTION METHOD:
# - detection
# - multitask
# - ref_proyect
DETECTION_METHOD = "yolopv2"
#Limit the number of images
#set it to 0 to use all the images
NIMAGES = 0
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
IOU_COMPARE_THRESHOLD = 0.6
MODEL_IMG_SIZE = 640
PREVIEW_MODE = False
PREVIEW_TIME = 500 #ms
#BDD10k DATASET CLASSES
#pedestrian, rider, car, truck, bus, train, motorcycle
#bicycle, traffic light, traffic sign
ACCEPTED_CLASSES = set(["car", "truck", "bus", "motorcycle"])
ACCEPTED_CLASSES_YOLO = set([2, 3, 4, 6, 7])
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
if DETECTION_METHOD == "detection":
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH + MODEL_NAME)

elif DETECTION_METHOD == "multitask":
    sys.path.insert(0, './ultralytics_multitask')
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH + MODEL_NAME)

elif DETECTION_METHOD == "ref_proyect":
    sys.path.insert(0, './reference_proyect/CarND_Vehicle_Detection')
    from reference_proyect.CarND_Vehicle_Detection.car_finder import *

    with open('reference_proyect/CarND_Vehicle_Detection/classifier.p', 'rb') as f:
        data = pickle.load(f)

    scaler = data['scaler']
    cls = data['classifier']
    window_size = [64, 80, 96, 112, 128, 160]
    window_roi = [((200, 400),(1080, 550)), ((100, 400),(1180, 550)), ((0, 380),(1280, 550)),((0, 360),(1280, 550)), ((0, 360),(1280, 600)), ((0, 360),(1280, 670)) ]
    
    carFinder = CarFinder(64, hist_bins=128, small_size=20, orientations=12, pix_per_cell=8, cell_per_block=1, classifier=cls, scaler=scaler, window_sizes=window_size, window_rois=window_roi)

elif DETECTION_METHOD == "yolopv2":
    model = torch.jit.load(MODEL_PATH + MODEL_NAME)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    device = torch.device("cpu")

#load labels
print("Loading labels...")
labels = json.load(open(DATASET_PATH + DATASET_JSON_NAME))

nImg = len(labels) if NIMAGES == 0 else NIMAGES
totalDet = 0
detHits = 0
falseDet = 0
missingDet = 0
newLabels = []
st = time()

for iImg, label in enumerate(labels[:nImg]):

    img = cv2.imread(DATASET_PATH+"val/"+label['name'])

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
    if DETECTION_METHOD == "detection":
        pred = model.predict(source=img, imgsz=MODEL_IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        for box in pred[0].boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = box
            
            if(class_id in ACCEPTED_CLASSES_YOLO):

                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)

                detectedBoxes.append((x1,y1,x2,y2))

    elif DETECTION_METHOD == "multitask":
        pred = model.predict(source=img, imgsz=MODEL_IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        if not pred[0]:
            continue

        for bbox in pred[0][0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = bbox

            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            detectedBoxes.append((x1,y1,x2,y2))
    
    elif DETECTION_METHOD == "ref_proyect":
        carFinder.find_cars(img, reset=True)

        for car in carFinder.cars:
            detectedBoxes.append(car.filtered_bbox.output().astype(np.int32))
    
    elif DETECTION_METHOD == "yolopv2":
        pred = detect(img, model, device)
        print(pred)
        continue
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
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
            detectedBoxes.pop(iMaxIou)
        else:
            missingDet += 1

        totalDet += 1
    
    falseDet += len(detectedBoxes)


    if(PREVIEW_MODE):
        for detectedBox in detectedBoxes:
            x1, y1, x2, y2 = detectedBox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imshow("Image Preview", img)
        cv2.waitKey(PREVIEW_TIME)
    
    printProgress(iImg, nImg)

totalTime = (time()-st)*1000



#Detections
# +-------------------+--------------------+---------------+
# |         -         | Didn't Predict Car | Predicted Car |
# +-------------------+--------------------+---------------+
# | There were no car | -                  |  falseDet     |
# | There was a car   | missingDet         |  detHits      |
# +-------------------+--------------------+---------------+

if(totalDet > 0 and detHits+falseDet):
    print("")
    print("Benchmark finished!")
    print(f"Accuracy: {(detHits/totalDet)*100:.2f}% [{detHits}/{totalDet}]")   
    print(f"Recall: {detHits/(detHits+falseDet)*100:.2f}%")
    print(f"Time: Avg: {totalTime/nImg:.2f}ms --- Total: {totalTime/1000:.2f}sec")
