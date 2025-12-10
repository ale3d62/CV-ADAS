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