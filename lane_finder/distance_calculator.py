def carInlane(x1,x2,y2, lx3, rx3, vpy, vpx, imgHeight):
    
    #if y2 is at the wrong height
    if(y2 > imgHeight or y2 < vpy):
        return False 
    

    #Detect if car is to the left, center, or right
    #center
    if(x1 < vpx and x2 > vpx):
        return True
    
    upPercent = (imgHeight-y2) / (imgHeight-vpy)
    if(upPercent < 0.42):
        return False

    boxWidth = x2-x1

    #left
    if(x2 < vpx):
        return (x1 + boxWidth*upPercent*0.9 > lx3)

    #right
    if(x1 > vpx):
        return (x2 - boxWidth*upPercent*0.9 < rx3)

    return False



def getDistances(frame, bBoxes, bestLinePointsLeft, bestLinePointsRight, roadWidth, sensorPixelW, f):
    
    distances = []

    #Get lines data
    lx1, lx2 = bestLinePointsLeft
    rx1, rx2 = bestLinePointsRight

    #if some lines data is missing
    if(not lx1 or not lx2 or not rx1 or not rx2):
        return distances


    imgHeight, imgWidth, _ = frame.shape
    halfImgHeight = int(imgHeight/2)

    #get lines equations (y = mx + b)
    lm = halfImgHeight/(lx2-lx1) # left
    lb = -lm * lx1
    rm = halfImgHeight/(rx2-rx1) # right
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
    for bBox in bBoxes:
        x1, y1, x2, y2 = bBox
        
        #coordinates x of the lines at the car's height
        lx3 = (y2-lb)/lm
        rx3 = (y2-rb)/rm

        #if car is in lane
        if(carInlane(x1,x2,y2, lx3, rx3, vpy, vpx, imgHeight)):
            d = (f*roadWidth*imgWidth)/((rx3-lx3)*(sensorPixelW*imgWidth))
            d = d/1000
            distances.append((d, x1, y1, x2))

    
    return distances
