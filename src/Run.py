import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid
import os
import time
from flask import Flask, request, jsonify
from train import trainFunc

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom',
    path= os.getenv('path_module'),
    force_reload=True)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    results = model(frame)

    cv2.imshow("YOLO", np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
