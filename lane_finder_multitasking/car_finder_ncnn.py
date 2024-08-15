import cv2
from ncnn import ncnn
import torch
import numpy as np
from time import time

video_path = "../test_videos/"

# Indexes of the videos to use as input
inputVideos = [*range(8, 27)] 

net = ncnn.Net()
net.load_param("yolov8n.param")
net.load_model("yolov8n.bin")
ex = net.create_extractor()

width = 1280
height = 720

def preprocess(image, target_size):
    h, w, _ = image.shape
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    new_image[(target_size - nh) // 2: (target_size - nh) // 2 + nh,
              (target_size - nw) // 2: (target_size - nw) // 2 + nw, :] = resized_image
    new_image = new_image.astype(np.float32) / 255.0
    new_image = new_image.transpose(2, 0, 1)
    new_image = new_image[np.newaxis, :]
    return new_image, scale


def postprocess(detections, image_shape, scale, conf_threshold=0.5):
    h, w = image_shape
    boxes, scores, class_ids = [], [], []
    for det in detections:
        if det[4] >= conf_threshold:
            scores.append(det[4])
            class_ids.append(int(det[5]))
            box = det[:4] / scale
            boxes.append(box)
    return boxes, scores, class_ids

def draw_detections(image, boxes, scores):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


input_size = 640
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
        frame = cv2.resize(frame, (width, height))

        #input = ncnn.Mat.from_pixels(frame, ncnn.Mat.PixelType.PIXEL_BGR, width, height)
        

        preprocessed_image, scale = preprocess(frame, input_size)
        blob = ncnn.Mat(preprocessed_image)

        ex.input("images", blob)
        stl = time()
        outDetRet, outDet = ex.extract("output")
        #outSegRet, outSeg = ex.extract("677") #727
        detections = np.array(outDet)
        boxes, scores, class_ids = postprocess(detections, frame.shape[:2], scale)
        draw_detections(frame, boxes, scores)
        print("Pred time: " + str((time() - stl) * 1000))
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        continue






        segH = outSeg.h
        segW = outSeg.w
        
        if outSegRet == 0:

            # PROCESS LANES
            mask = np.array(outSeg).reshape((2, segH,segW))
            mask = mask[1,:,:]

            color_mask = np.stack([mask * 0, mask * 255, mask * 0], axis=-1)
            alpha = 0.5  # transparency factor
            # overlay
            frame[np.any(color_mask != [0, 0, 0], axis=-1)] = (
                (1 - alpha) * frame[np.any(color_mask != [0, 0, 0], axis=-1)] + 
                alpha * color_mask[np.any(color_mask != [0, 0, 0], axis=-1)]
            )
            """
            # PROCESS BBOXES
            for i in range (outDet.h):
                values = outDet.row(i)
                conf = values[1]
                if(conf > 0.5):
                    x1 = int(values[2]*width)
                    y1 = int(values[3]*height)
                    x2 = int(values[4]*width)
                    y2 = int(values[5]*height)

                    cv2.rectangle(frame, (x1,y1,x2,y2), (0, 255, 0), 1)
            """
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
