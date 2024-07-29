import cv2
from ncnn import ncnn
import torch
import numpy as np
from time import time

video_path = "../test_videos/"

# Indexes of the videos to use as input
inputVideos = [*range(8, 27)] 

net = ncnn.Net()
net.load_param("v4n_lane_det.param")
net.load_model("v4n_lane_det.bin")
ex = net.create_extractor()

width = 1280
height = 736

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

        input = ncnn.Mat.from_pixels(frame, ncnn.Mat.PixelType.PIXEL_BGR, width, height)
        
        ex.input("images", input)
        stl = time()
        outDetRet, outDet = ex.extract("detect")
        outSegRet, outSeg = ex.extract("727")

        print("Pred time: " + str((time() - stl) * 1000))

        segH = outSeg.h
        segW = outSeg.w
        
        if outDetRet == 0 and outSegRet == 0:

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
