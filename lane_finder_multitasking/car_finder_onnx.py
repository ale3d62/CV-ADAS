import cv2
from ultralytics import YOLO
import torch
import numpy as np
from time import time
from time import sleep

model = YOLO("v4n_lane_det.onnx")

video_path = "../test_videos/"

# Indexes of the videos to use as input
inputVideos = [*range(1, 7)] 



while len(inputVideos) > 0:

    # Get input video
    vid = cv2.VideoCapture(video_path + 'test' + str(inputVideos[0]) + '.mp4')
    inputVideos.pop(0)

    ret = True
    stt = time()    
    nFrames = 1
    while ret:
        # Get frame
        ret, frame = vid.read()

        if not ret:
            break
        
        nFrames += 1

        stl = time()
        frame = cv2.resize(frame, (1280, 736))

        # results = model(frame)
        stp = time()
        """
        results = model.predict(
            source=frame, 
            stream=True, 
            imgsz=(384, 672), 
            device=[0], 
            name='v4_daytime', 
            save=True, 
            conf=0.25, 
            iou=0.45, 
        )
        """
        results = model.track(source=frame, conf=0.3, iou=0.5, verbose=False)
        print("Pred time: " + str((time() - stl) * 1000))
        exit()
        if results:
            """
            # PROCESS LANES
            mask = results[1][0].to(torch.uint8).cpu().numpy()
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
                """
            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
        else:
            print("no results")
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
        
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        
        #print("Total time: "+str((time() - stl) * 1000))
    print("Avg time: "+str((time() - stt) * 1000 / nFrames))
