import cv2
from ultralytics import YOLO
import torch
import numpy as np
from time import time

#Returns the bboxes of the acceptedClasses found in the frame 
model = YOLO("v4n.onnx")

ret = True
vid = cv2.VideoCapture("test9.mp4")

if(not vid):
    print("no vid")
    exit()

while(ret):
    #Get frame
    ret, frame = vid.read()

    if(not ret):
        exit()

    stl = time()
    frame = cv2.resize(frame,(1280, 720))

    #results = model(frame)


    results = model.predict(source=frame, stream=True, imgsz=(384,672), device="cpu",name='v4_daytime', save=True, conf=0.25, iou=0.45, verbose=True)
    
    if(results):
        #PROCESS LANES
        mask = results[2][0].to(torch.uint8).cpu().numpy()
        color_mask = np.stack([mask * 0, mask * 255, mask * 0], axis=-1)
        alpha = 0.5  # transparency factor
        #overlay
        frame[np.any(color_mask != [0, 0, 0], axis=-1)] = (1 - alpha) * frame[
                    np.any(color_mask != [0, 0, 0], axis=-1)] + alpha * color_mask[
                                                                     np.any(color_mask != [0, 0, 0], axis=-1)]

        #PROCESS BBOXES
        for bbox in results[0][0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = bbox

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        cv2.imshow('Frame',frame)
        cv2.waitKey(1)
    else:
        print("no results")
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
    
    cv2.imshow('Frame',frame)
    cv2.waitKey(1)
    print((time()-stl)*1000)
