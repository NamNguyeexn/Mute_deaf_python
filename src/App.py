from flask import Flask, render_template, Response
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np
import uuid
import os
import time

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='D:\\Tai_lieu_ki_1_nam_4\\PT_HTTM\\Mute_deaf_python\\yolov5\\runs\\train\\exp27\\weights'
                            '\\last.pt',
                       force_reload=True)


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # results = model_hand(frame)
        # lastResult = model_sign(results.render())
        results = model(frame)

        # cv2.imshow("YOLO", np.squeeze(lastResult.render())[1])
        cv2.imshow("YOLO", np.squeeze(results.render()))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
