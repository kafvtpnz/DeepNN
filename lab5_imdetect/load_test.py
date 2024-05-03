# -*- coding: utf-8 -*-

import ultralytics
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

ultralytics.checks()

model = YOLO("yolov8s.pt") 
cap = cv2.VideoCapture('')


if(model and cap):
    print('')
    print('OK!')

