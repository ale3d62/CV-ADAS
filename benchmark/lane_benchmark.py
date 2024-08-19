import json
import cv2
from ultralytics import YOLO
import sys
from torch import uint8


# PROGRESS BAR
def print_progress(iImg, nImg):
    bar_length = 40 
    progress = (iImg + 1) / nImg 
    block = int(round(bar_length * progress))
    
    progress_text = f"\rRunning Benchmark {iImg + 1}/{nImg} [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.1f}%  -  acc:{(segHits/totalSeg)*100:.2f}%"
    sys.stdout.write(progress_text)
    sys.stdout.flush()


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



DATASET_PATH = 'datasets/train_set/'
DATASET_JSON_NAME = 'label_data_0531.json'
MODEL_PATH = '../models/'
MODEL_NAME = 'v4n_lane_det.onnx'
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_IMG_SIZE = (384,672)
PREVIEW_MODE = True

#load model
model = YOLO(MODEL_PATH + MODEL_NAME)

#load labels
labels = [json.loads(line) for line in open(DATASET_PATH + DATASET_JSON_NAME).readlines()]


nImg = len(labels)
totalSeg = 1
segHits = 1

for iImg, label in enumerate(labels):
    
    print_progress(iImg, nImg)

    img = cv2.imread(DATASET_PATH + label['raw_file'])
    imgH, imgW, _ = img.shape

    pred = model.predict(source=img, imgsz=MODEL_IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    pred_mask = pred[-1][0].to(uint8).cpu().numpy()

    if(PREVIEW_MODE):
        img[pred_mask==1] = (30, 255, 15)
    
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
            cv2.waitKey(500)

    except:
        pass

print("")
print(f"Benchmark finished, [{segHits}/{totalSeg}] {(segHits/totalSeg)*100:.2f}%")        
    
    