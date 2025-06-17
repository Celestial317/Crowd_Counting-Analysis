import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolo11s.pt")
img = cv2.imread("crowd_wala_dataset/train_data/images/IMG_1.jpg")
results = model(img)

annotated_frame = results[0].plot()
annotated_frame = cv2.resize(annotated_frame, (640, 320))
cv2.imshow("camera", annotated_frame)
cv2.waitKey(0)
