import cv2
from lane_finder import findLane
from auxFunctions import *
from time import time

from ultralytics import YOLO



#----------PARAMETERS----------------
modelName = "yolov8x.pt"
showLines = True
#CAMERA PARAMETERS
f = 2.5
sensorPixelW = 0.008
roadWidth = 3600
#------------------------------------

#Load yolo model
model = YOLO(modelName)


for i in range(8,23):
    vid = cv2.VideoCapture('test_videos/test'+str(i)+'.mp4')
    totalTime = 0
    totalFrames = 0
    resScaling = 0.75
    bestLinePointsLeft = (None, None)
    bestLinePointsRight = (None, None)

    st = time()
    while(vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()
        
        if ret == True:
            if(resScaling != 1):
                frame = cv2.resize(frame, (0,0), fx = resScaling, fy = resScaling, interpolation=cv2.INTER_NEAREST)
            totalFrames += 1

            #SCAN FOR CARS
            """
            results = model(frame, verbose=False)[0]
            bBoxes = [] 
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, _ = result

                if score > 0.2:
                    bBoxes.append((x1,y2,x2,y2))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

            
            if(len(bBoxes) == 0):
                cv2.imshow('Frame',frame)
                cv2.waitKey(1)
                continue
            """

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



            #GET DISTANCE OF CAR
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
            """
            for bBox in bBoxes:
                x1, y1, x2, y2 = bBox
                
                lx3 = ((imgHeight-y2)-lb)/lm
                rx3 = ((imgHeight-y2)-rb)/rm

                #if car is in lane
                if(y2 < imgHeight and y2 > halfImgHeight and ((lx3 < x1 and rx3 > x1) or (lx3 < x2 and rx3 > x2))):
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
        print("avg time: "+ str(totalTime/totalFrames) + "ms")