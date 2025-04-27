import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import opencv2 as cv2

import numpy as np
import os
import time

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


