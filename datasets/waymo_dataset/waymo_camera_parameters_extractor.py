from waymo_extractor_utils import camera_parameters_extract

#SEQUENCEID = "10923963890428322967_1445_000_1465_000"
#SEQUENCEID = "11199484219241918646_2810_030_2830_030"
SEQUENCEID = "10625026498155904401_200_000_220_000"
CAMERA_CALIB_PATH = "./camera_calibration/"


if __name__ == "__main__":

    cameraCailbFilePath = f"{CAMERA_CALIB_PATH}/{SEQUENCEID}.parquet"
    cameraParameters = camera_parameters_extract.extract_camera_parameters(cameraCailbFilePath)

    print(f"Camera height: {cameraParameters.cameraHeight}")
    print(f"Yaw (deg): {cameraParameters.yaw}, pitch (deg): {cameraParameters.pitch}, roll (deg): {cameraParameters.roll}")
    print(f"f_u: {cameraParameters.f_u}, f_v: {cameraParameters.f_v}")
    print(f"c_u: {cameraParameters.c_u}, c_v: {cameraParameters.c_v}")

    sensorWidth = 36 #Estimated from full frame camera standard

    f = cameraParameters.f_u * (sensorWidth/cameraParameters.imgW)
    print(f"Estimated focal length: {f}mm")
