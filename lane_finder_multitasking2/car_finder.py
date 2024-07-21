import cv2
from ultralytics import YOLO
import torch
import numpy as np

#Returns the bboxes of the acceptedClasses found in the frame 
model = YOLO("v4.pt")

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

        
    frame = cv2.resize(frame,(1280, 720))

    #results = model(frame)
    print("predicting..")
    results = model.predict(source=frame, stream=True, imgsz=(384,672), device="cpu",name='v4_daytime', save=True, conf=0.25, iou=0.45, verbose=True)
    #print("results "+str(results[2]))
    print(results)
    # Dibujar las detecciones en la imagen
    continue
    if(results):
        for result in results[0]:
            # Obtener las coordenadas del bounding box
            """
            boxes = result.boxes.xyxy
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            """
            im_array = result.plot()  # plot a BGR numpy array of predictions
             # cv2.imwrite('predicted.png',im_array)

                        # Convert tensor to ndarray and remove the first dimension

            mask1 = results[1].to(torch.uint8).cpu().numpy()
            mask2 = results[2].to(torch.uint8).cpu().numpy()

            # Convert mask to RGB
            color_mask1 = np.stack([mask1 * 0, mask1 * 255, mask1 * 0], axis=-1)
            color_mask2 = np.stack([mask2 * 255, mask2 * 0, mask2 * 0], axis=-1)

            alpha = 0.5  # transparency factor

            # Overlay masks on im0 with transparency
            frame[np.any(color_mask1 != [0, 0, 0], axis=-1)] = (1 - alpha) * frame[
                np.any(color_mask1 != [0, 0, 0], axis=-1)] + alpha * color_mask1[
                                                                 np.any(color_mask1 != [0, 0, 0], axis=-1)]
            frame[np.any(color_mask2 != [0, 0, 0], axis=-1)] = (1 - alpha) * frame[
                np.any(color_mask2 != [0, 0, 0], axis=-1)] + alpha * color_mask2[
                                                                 np.any(color_mask2 != [0, 0, 0], axis=-1)]

            cv2.namedWindow("asd", cv2.WINDOW_NORMAL)  
            cv2.imshow('asd',im_array)
            cv2.waitKey(1)
    else:
        print("no results")
        cv2.imshow('Detecciones YOLOv8', frame)
        cv2.waitKey(1)

# Mostrar la imagen con las detecciones
#cv2.imshow('Detecciones YOLOv8', image)
#cv2.waitKey(0)
