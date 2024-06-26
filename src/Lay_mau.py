import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid
import os
import time

model = torch.hub.load("ultralytics/yolov5", "yolov5s")


label = "hand"
number_imgs = 100

IMAGES_PATH = (
    "D:\Tai_lieu_ki_1_nam_4\PT_HTTM\Mute_deaf_python\data_canh_tay\images"
)

cap = cv2.VideoCapture(0)
print("Collecting images for {}".format(label))
time.sleep(3)
for img_num in range(number_imgs):
    print("Collecting images for {}, image number {}".format(label, img_num))

    # Webcam feed
    ret, frame = cap.read()

    # Naming out image path
    imgname = os.path.join(IMAGES_PATH, label + "." + str(uuid.uuid1()) + ".jpg")

    # Writes out image to file
    cv2.imwrite(imgname, frame)

    # Render to the screen
    cv2.imshow("Image Collection", frame)

    # 2 second delay between captures
    time.sleep(2)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
