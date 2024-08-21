#USE THIS FOR EXTRACTING REGULAR DETECTION MODELS

from ultralytics import YOLO

model = YOLO('../../yolov8n.pt',task='detect')
model.export(format='OpenVINO')