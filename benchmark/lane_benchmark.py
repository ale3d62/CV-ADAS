import json
import cv2
from torch import uint8
from progressBar import printProgress


#-------------PARAMETERS--------------------------
DATASET_PATH = 'datasets/lanes/tusimple/train_set/'
DATASET_JSON_NAME = 'label_data_0531.json'
MODEL_PATH = '../models/'
MODEL_NAME = 'v4n_lane_det.onnx'
#ARCHITECTURE TYPES:
# - yoloP
# - multitask
ARCHITECTURE_TYPE = "multitask"
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_IMG_SIZE = (384,672)
PREVIEW_MODE = True
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




#load model
if ARCHITECTURE_TYPE == "detection":
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH + MODEL_NAME)
elif ARCHITECTURE_TYPE == "multitask":
    sys.path.insert(0, './ultralytics_multitask')
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH + MODEL_NAME)

#load labels
print("Loading labels...")
labels = [json.loads(line) for line in open(DATASET_PATH + DATASET_JSON_NAME).readlines()]


nImg = len(labels)

#Each line is composed of several segments, a hit is when a predicted segment overlaps wit a ground truth segment
totalSeg = 0
segHits = 0

for iImg, label in enumerate(labels):

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
            cv2.waitKey(PREVIEW_TIME)

    except:
        pass

    printProgress(iImg, nImg)


print("")
print("Benchmark finished!")
print(f"Benchmark finished, [{segHits}/{totalSeg}]")   
print(f"Accuracy: {(segHits/totalSeg)*100:.2f}%") 
    
    