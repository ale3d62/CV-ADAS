# -*- coding: utf-8 -*-
# @Time    : 2024/7/16 17:21
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : 1.py
# ------❤❤❤------ #

#
from ultralytics import YOLO

model = YOLO('yolov8n.pt',task='detect')
model.export(format='onnx', imgsz=672)

#YOU CAN RUN THIS WITH PYTHON 3.7 AND THE VENV AT THE PARENT FOLDER
#RUN THIS WITH WITH THE SPECIFIC MODEL AS THIS ONLY EXTRACT WITH SIZE 374x640 (OR WHATEVER THE SIZE WAS), REGARDLESS OF IMGSZ