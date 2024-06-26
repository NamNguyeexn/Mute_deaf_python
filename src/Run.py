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
                       path='D:\\Tai_lieu_ki_1_nam_4\\PT_HTTM\\Mute_deaf_python\\yolov5\\runs\\train\\exp28\\weights\\last.pt',
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
