from waymo_extractor_utils import distance_extract

#SEQUENCEID = "10923963890428322967_1445_000_1465_000"
SEQUENCEID = "10625026498155904401_200_000_220_000"
CAMERA_CALIB_PATH = "./camera_calibration/"
LIDAR_BOX_PATH = "./lidar_box/"


if __name__ == "__main__":
    cameraCailbFilePath = f"{CAMERA_CALIB_PATH}/{SEQUENCEID}.parquet"
    lidarBoxFilePath = f"{LIDAR_BOX_PATH}/{SEQUENCEID}.parquet"

    distance_extract.extract_front_car_distance(lidarBoxFilePath, cameraCailbFilePath)
