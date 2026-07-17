from waymo_extractor_utils import camera_parameters_extract
import pandas as pd

def extract_front_car_distance(lidarBoxFile, cameraCalibFile):

    #Get camera x offset
    cameraParameters = camera_parameters_extract.extract_camera_parameters(cameraCalibFile)
    cameraXOffset = cameraParameters.cameraXOffset

    cols = [
        'key.segment_context_name',
        'key.frame_timestamp_micros',
        '[LiDARBoxComponent].type',
        'key.laser_object_id',
        '[LiDARBoxComponent].box.size.x',
        '[LiDARBoxComponent].box.center.x',
        '[LiDARBoxComponent].box.center.y',
        '[LiDARBoxComponent].box.center.z'
    ]

    #Load bounding boxes file
    try:
        dfLidar = pd.read_parquet(lidarBoxFile, columns=cols)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    if dfLidar.empty:
        print(f"Bounding box file is empty.")
        return

    #Sort frames by timestamp
    totalTimestamps = sorted(dfLidar['key.frame_timestamp_micros'].unique())
    print(f"There's a total of {len(totalTimestamps)} frames in the sequence")

    #Filter only vehicles (type == 1)
    dfVehicles = dfLidar[dfLidar['[LiDARBoxComponent].type'] == 1].copy()

    #Filter front vehicles (X > 0) and same lane (-1.75m to 1.75m)
    dfFront = dfVehicles[
        (dfVehicles['[LiDARBoxComponent].box.center.x'] > 0) &
        (dfVehicles['[LiDARBoxComponent].box.center.y'] > -1.75) &
        (dfVehicles['[LiDARBoxComponent].box.center.y'] < 1.75)
    ].copy()

    if dfFront.empty:
        print("No front vehicle was found in the same lane for this sequence")
        return

	#Dictionary of distances for each vehicle
    vehicleIds = dfVehicles['key.laser_object_id'].unique()
    distances = {vehicleId: {} for vehicleId in vehicleIds}

    #For each frame, get the index corresponding to the closest vehicle
    idxClosestVehicles = dfFront.groupby('key.frame_timestamp_micros')['[LiDARBoxComponent].box.center.x'].idxmin()

    #Get the rows that correspond to the closest vehicle for each frame
    dfResult = dfFront.loc[idxClosestVehicles].sort_values('key.frame_timestamp_micros')

    for index, row in dfResult.iterrows():
        ts = row['key.frame_timestamp_micros']
        carId = row['key.laser_object_id']
        dist = row['[LiDARBoxComponent].box.center.x']

        boxSize = row['[LiDARBoxComponent].box.size.x']
        finalDistance = dist-(boxSize/2)-cameraXOffset

        distances[carId][ts] = finalDistance

	#Print results
    for vehicle, distancesByTimestamp in distances.items():
        if len(distancesByTimestamp) > 0:
            print(f"\Vehicle ID: {vehicle}")

            for ts in totalTimestamps:
                if ts in distancesByTimestamp:
                    print(f"{distancesByTimestamp[ts]:.2f}")
                else:
                    #If the vehicle is not in the same lane for that frame
                    print("-")
