import json
import cv2
import torch
from progressBar import printProgress
import sys
import numpy as np
from time import time
import os
from yolopv2 import detect


#-------------PARAMETERS--------------------------
DATASET_PATH = 'datasets/lanes/tusimple/train_set/'
DATASET_JSON_NAME = 'label_data_0313.json'
MODEL_PATH = '../models/'
MODEL_NAME = 'yolopv2.pt'
#DETECTION METHOD:
# - multitask
# - classic_cv
# - yolopv2
DETECTION_METHOD = "yolopv2"
#Limit number of images
#set it to 0 to use all the images
NIMAGES = 0
CLASSIC_CV_LINE_WIDTH = 8 #px
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_IMG_SIZE = (384,672)
PREVIEW_MODE = False
PREVIEW_TIME = 500 #ms
#-------------------------------------------------




def getCenterLinesIndex(imgDim, label):

    imgH, imgW, _ = imgDim
    imgCenter = imgW/2

    #Get the h_sample closest to 3/4 of imgH
    refH_sample = min(range(len(label['h_samples'])), key=lambda i: abs(label['h_samples'][i] - imgH*(3/4)))

    #Get the lines closest to the center at refH_sample
    lanesSegmentsW = [label['lanes'][i][refH_sample] for i in range(len(label['lanes']))]

    iLeftLine = max([i for i in range(len(lanesSegmentsW)) if lanesSegmentsW[i] < imgCenter], key=lambda i: lanesSegmentsW[i])
    iRightLine = min([i for i in range(len(lanesSegmentsW)) if lanesSegmentsW[i] >= imgCenter], key=lambda i: lanesSegmentsW[i])

    return iLeftLine, iRightLine

# Given linepoints = (x1,x2), where (x1, imgH) and (x2, int(imgH/2)) are 
# in the same line, return the point that belongs to that line at height y
def getLinePoint(linePoints, y, imgDim):
    imgW, imgH, _ = imgDim
    halfImgHeight = int(imgH/2)

    x1,x2 = linePoints

    #get line equation (y = mx + b)
    m = halfImgHeight/(x2-x1)
    b = -m * x1

    #flip line equation (as increasing means going down in the image)
    m = -m
    b = -b + imgH

    return int(y-b / m)



#load model
if DETECTION_METHOD == "detection":
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH + MODEL_NAME)
elif DETECTION_METHOD == "multitask":
    sys.path.insert(0, './ultralytics_multitask')
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH + MODEL_NAME)
elif DETECTION_METHOD == "classic_cv":
    sys.path.insert(0, '../lane_finder_classic_cv')
    from lane_finder import *
elif DETECTION_METHOD == "yolopv2":
    if(MODEL_IMG_SIZE != (384, 672)):
        raise Exception("MODEL_IMG_SIZE has to be 384x672")
    model = torch.jit.load(MODEL_PATH + MODEL_NAME)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    device = torch.device("cpu")


#load labels
print("Loading labels...")
labels = [json.loads(line) for line in open(DATASET_PATH + DATASET_JSON_NAME).readlines()]


nImg = len(labels) if NIMAGES == 0 else NIMAGES

#Each line is composed of several segments, a hit is when a predicted segment overlaps wit a ground truth segment
totalSeg = 0
segHits = 0
st = time()


for iImg, label in enumerate(labels[:nImg]):

    img = cv2.imread(DATASET_PATH + label['raw_file'])
    imgH, imgW, _ = img.shape

    if DETECTION_METHOD == "multitask":
        pred = model.predict(source=img, imgsz=MODEL_IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        pred_mask = pred[-1][0].to(torch.uint8).cpu().numpy()

    elif DETECTION_METHOD == "classic_cv":
        _, bestLinePointsLeft, bestLinePointsRight, _ = findLane(img, None, None, False)
        pred_mask = np.zeros(img.shape, dtype=np.torch.uint8)
        if(bestLinePointsLeft):
            lx3 = getLinePoint(bestLinePointsLeft, 0, img.shape)
            cv2.line(pred_mask, (bestLinePointsLeft[0], imgH), (lx3, 0), color=255, thickness=CLASSIC_CV_LINE_WIDTH)
        if(bestLinePointsRight):
            rx3 = getLinePoint(bestLinePointsRight, 0, img.shape)
            cv2.line(pred_mask, (bestLinePointsRight[0], imgH), (rx3, 0), color=255, thickness=CLASSIC_CV_LINE_WIDTH)
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
    
    elif DETECTION_METHOD == "yolopv2":
        _, pred_mask = detect(img, model, device, imgsz=MODEL_IMG_SIZE)


    if(PREVIEW_MODE):
        if(np.any(pred_mask)):
            mask_bool = pred_mask != 0
            img[mask_bool] = np.array([30, 255, 15], dtype=np.uint8)
    
    try:
        #get index of center lanes
        iLeftLine, iRightLine = getCenterLinesIndex(img.shape, label)

        for i in range(len(label['h_samples'])):
            segY = label['h_samples'][i]

            #limit segment height
            if(segY < imgH*0.4 or segY > imgH*0.75):
                continue

            #left line
            lSegX = label['lanes'][iLeftLine][i]
            if(pred_mask[lSegX][segY]):
                segHits += 1
            if(lSegX != -2):
                totalSeg += 1

            #right line
            rSegX = label['lanes'][iRightLine][i]
            if(pred_mask[lSegX][segY]):
                segHits += 1
            if(rSegX != -2):
                totalSeg += 1

            if(PREVIEW_MODE):
                cv2.circle(img, (lSegX, segY), 2, (0, 0, 255), 2)
                cv2.circle(img, (rSegX, segY), 2, (0, 0, 255), 2)

        if(PREVIEW_MODE):
            cv2.imshow("Image Preview", img)
            cv2.waitKey(PREVIEW_TIME)

    except:
        pass

    printProgress(iImg, nImg)

totalTime = (time()-st)*1000


if(totalSeg > 0 and nImg > 0):
    print("")
    print("Benchmark finished!")
    print(f"Accuracy: {(segHits/totalSeg)*100:.2f}% [{segHits}/{totalSeg}]") 
    print(f"Time: Avg: {totalTime/nImg:.2f}ms --- Total: {totalTime/1000:.2f}sec")
    