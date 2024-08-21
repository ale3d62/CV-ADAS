import sys

sys.path.insert(0, "/home/xianyang/Desktop/YOLOv8-multi-task-main")

from ultralytics import YOLO

number = 12  # input how many tasks in your work
# model = YOLO('runs/multi/yolopm11/weights/best.pt')  # Validate the model
# model.export(format='onnx')
#
#
model = YOLO('v4n.engine')
model.predict(source=r"test.jpg", imgsz=(384,672), device=[0],
              name='v4_daytime', save=True, show=False,conf=0.5, iou=0.5, show_labels=True)
