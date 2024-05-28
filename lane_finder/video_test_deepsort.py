import cv2
from lane_finder import findLane
from auxFunctions import *
from time import time

#for screen capture
import numpy as np
from mss import mss
from PIL import Image

#YOLO
from ultralytics import YOLO

#deepsort
from tracker import Tracker

#for screen capture
bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
sct = mss()

#----------PARAMETERS----------------
modelName = "yolov8x.pt"
acceptedClasses = set([2, 3, 4, 6, 7])
showLines = True
resScaling = 1
#CAMERA PARAMETERS
f = 2.5
sensorPixelW = 0.008
roadWidth = 3600
#------------------------------------

#Load yolo model
model = YOLO(modelName)

#Instantiate deepsort tracker
tracker = Tracker()


for i in range(7,27):
#while True:
    
    vid = cv2.VideoCapture('test_videos/test'+str(i)+'.mp4')
    totalTime = 0
    totalFrames = 0
    bestLinePointsLeft = (None, None)
    bestLinePointsRight = (None, None)

    st = time()
    while(vid.isOpened()):
    #while True:

        # Capture frame-by-frame
        ret, frame = vid.read()
        #sct_img = sct.grab(bounding_box)
        #frame = np.array(sct_img)
        #frame = np.delete(frame, 3, axis=2)
        
        if ret == True:
            if(resScaling != 1):
                frame = cv2.resize(frame, (0,0), fx = resScaling, fy = resScaling, interpolation=cv2.INTER_NEAREST)
            totalFrames += 1
            
            #SCAN FOR CARS
            #Run the model on your images
            results = yolo_pipeline(images=frame)
            
            results = model(frame, verbose=False)[0]
            bBoxes = [] 
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result


            #for box in results.boxes[0]:
                #x1, y1, x2, y2  = box
                #score = result.scores
                
                if(class_id in acceptedClasses and score > 0.2):
                    bBoxes.append((x1,y2,x2,y2))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

            
            if(len(bBoxes) == 0):
                cv2.imshow('Frame',frame)
                cv2.waitKey(1)
                continue
            
            tracker.update(frame, bBoxes)

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
            
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            


            #SCAN FOR LINES
            imgHeight, imgWidth, _ = frame.shape
            halfImgHeight = int(imgHeight/2)
            
            #Find new lines
            newBestLinePointsLeft, newBestLinePointsRight = findLane(frame)

            #If new lines, update
            if(newBestLinePointsLeft[0]):
                bestLinePointsLeft = newBestLinePointsLeft
            if(newBestLinePointsRight[0]):
                bestLinePointsRight = newBestLinePointsRight

            #Show lines
            if(showLines):
                if(bestLinePointsLeft[0] and bestLinePointsLeft[1]):
                    cv2.line(frame, (bestLinePointsLeft[1], halfImgHeight), (bestLinePointsLeft[0], imgHeight), (0, 0, 255), 2)
                if(bestLinePointsRight[0] and bestLinePointsRight[1]):
                    cv2.line(frame, (bestLinePointsRight[1], halfImgHeight), (bestLinePointsRight[0], imgHeight), (0, 0, 255), 2)

            
            #GET DISTANCE TO CAR
            lx1, lx2 = bestLinePointsLeft
            rx1, rx2 = bestLinePointsRight
            if(not lx1 or not lx2 or not rx1 or not rx2):
                cv2.imshow('Frame',frame)
                cv2.waitKey(1)
                continue
            
            #get lines equations
            lm = halfImgHeight/(lx2-lx1)
            lb = -lm * lx1
            rm = halfImgHeight/(rx2-rx1)
            rb = -rm * rx1

            #flip lines equations (as increasing means going down in the image)
            lm = -lm
            lb = -lb + imgHeight
            rm = -rm
            rb = -rb + imgHeight

            #get vanishing point coordinates
            vpx = (rb-lb) / (lm-rm)
            vpy = lm*vpx + lb

            #process boxes for distances
            """
            for bBox in bBoxes:
                x1, y1, x2, y2 = bBox
                
                lx3 = (y2-lb)/lm
                rx3 = (y2-rb)/rm

                #if car is in lane
                if(carInlane(x1,x2,y2, lx3, rx3, vpy, vpx, imgHeight)):

                #if(y2 < imgHeight and y2 > halfImgHeight and ((lx3 < x1 and rx3 > x1) or (lx3 < x2 and rx3 > x2))):
                    d = (f*roadWidth*imgWidth)/((rx3-lx3)*(sensorPixelW*imgWidth))
                    d = d/1000
                    #cv2.line(frame, (int(lx3), int(y2)), (int(rx3), int(y2)), (255, 50, 50), 2)
                    #print((rx3-lx3))
                    cv2.putText(frame, "Distance: {:6.2f}m".format(d), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1, color=(100, 100, 255))
                    #print("distance: "+str(d))
            """

            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    

            
        
        # Break the loop
        else: 
            break
    totalTime += (time()-st)*1000
    if(totalFrames>0):
        pass
        #print("avg time: "+ str(totalTime/totalFrames) + "ms")