from label_classes import parse_kitti_tracklets
import os

sequencePath = "./2011_09_26/2011_09_26_DRIVE_0015_sync"
veloToCamDistance = 0.27

if __name__ == "__main__":
    labelsFile = f"{sequencePath}/tracklet_labels.xml"
    imagesPath = f"{sequencePath}/image_00/data"

    nFrames = len(os.listdir(imagesPath))

    tracklets = parse_kitti_tracklets(labelsFile)
    nVehicles = len(tracklets)

    distances = [["-" for iFrame in range(nFrames)] for iVehicle in range(nVehicles)]

    for iVehicle, tracklet in enumerate(tracklets):
        iFrame = tracklet.first_frame

        for pose in tracklet.poses:
            vehicleLength = tracklet.l
            vehicleX = pose.tx #distance
            vehicleY = pose.ty #left-right offset
            #Filter vehicles in other lanes
            if(vehicleY < 2 and vehicleY > -2):
                distance = vehicleX-(vehicleLength/2)-veloToCamDistance
                distances[iVehicle][iFrame] = distance
            iFrame += 1


    #print distances
    for iVehicle in range(len(distances)):

        if any(distance != "-" for distance in distances[iVehicle]):
            print(f"Distances for vehicle {iVehicle}")

            for distance in distances[iVehicle]:
                print(distance)
