import cv2
from ultralytics import YOLO
import torch
import numpy as np
from time import time

model = YOLO("v4n.onnx")

video_path = "../test_videos/"

# Indexes of the videos to use as input
inputVideos = [*range(8, 27)] 



while len(inputVideos) > 0:

    # Get input video
    vid = cv2.VideoCapture(video_path + 'test' + str(inputVideos[0]) + '.mp4')
    inputVideos.pop(0)
    ret = True

    while ret:
        # Get frame
        ret, frame = vid.read()

        if not ret:
            break

        stl = time()
        frame = cv2.resize(frame, (1280, 720))

        # results = model(frame)

        results = model.predict(
            source=frame, 
            stream=True, 
            imgsz=(384, 672), 
            device=[0], 
            name='v4_daytime', 
            save=True, 
            conf=0.25, 
            iou=0.45, 
            verbose=True
        )
        
        if results:
            # PROCESS LANES
            mask = results[2][0].to(torch.uint8).cpu().numpy()
            color_mask = np.stack([mask * 0, mask * 255, mask * 0], axis=-1)
            alpha = 0.5  # transparency factor
            # overlay
            frame[np.any(color_mask != [0, 0, 0], axis=-1)] = (
                (1 - alpha) * frame[np.any(color_mask != [0, 0, 0], axis=-1)] + 
                alpha * color_mask[np.any(color_mask != [0, 0, 0], axis=-1)]
            )

            # PROCESS BBOXES
            for bbox in results[0][0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = bbox

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
        else:
            print("no results")
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
        
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        print((time() - stl) * 1000)
