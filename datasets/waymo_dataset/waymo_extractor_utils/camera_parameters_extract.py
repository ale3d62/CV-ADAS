import pandas as pd
from scipy.spatial.transform import Rotation


def matrix_to_euler(translationMatrix):

    # 2. extract 3x3 rotation submatrix (first 3 rows and cols)
    rotationSubmatrix = translationMatrix[0:3, 0:3].copy()

    r = Rotation.from_matrix(rotationSubmatrix)

    #Euler angles
    #'zyx' (Yaw, Pitch, Roll)
    yaw, pitch, roll = r.as_euler('zyx', degrees=True)
    return (yaw, pitch, roll)


class CameraParameters:
    def __init__(self, f_u, f_v, c_u, c_v, yaw, pitch, roll, cameraXOffset, cameraHeight, imgW):
        self.f_u = f_u
        self.f_v = f_v
        self.c_u = c_u
        self.c_v = c_v
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.cameraXOffset = cameraXOffset
        self.cameraHeight = cameraHeight
        self.imgW = imgW


def extract_camera_parameters(cameraCalibFilePath):

    cols = ['[CameraCalibrationComponent].extrinsic.transform',
            '[CameraCalibrationComponent].intrinsic.f_u',
            '[CameraCalibrationComponent].intrinsic.f_v',
            '[CameraCalibrationComponent].intrinsic.c_u',
            '[CameraCalibrationComponent].intrinsic.c_v',
            '[CameraCalibrationComponent].width']

    #Cargar el archivo de los parámetros de la cámara
    print(f"Reading camera calibration for file: {cameraCalibFilePath}...")
    try:
        dfCalib = pd.read_parquet(cameraCalibFilePath, columns=cols)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()

    translationMatrix = dfCalib['[CameraCalibrationComponent].extrinsic.transform'][0].reshape(4,4)
    yaw, pitch, roll = matrix_to_euler(translationMatrix)
    cameraXOffset = translationMatrix[0][3]
    cameraHeight = translationMatrix[2][3]

    f_u = dfCalib['[CameraCalibrationComponent].intrinsic.f_u'][0]
    f_v = dfCalib['[CameraCalibrationComponent].intrinsic.f_v'][0]
    c_u = dfCalib['[CameraCalibrationComponent].intrinsic.c_u'][0]
    c_v = dfCalib['[CameraCalibrationComponent].intrinsic.c_v'][0]
    imgW = dfCalib['[CameraCalibrationComponent].width'][0]

    return CameraParameters(f_u, f_v, c_u, c_v, yaw, pitch, roll, cameraXOffset, cameraHeight, imgW)
