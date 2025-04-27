import os

import cv2 as cv
import mediapipe as mp



def extract_test(datasets_path):
    dirs = []
    if datasets_path is None:
        raise ValueError("Dataset path cannot be None")
    for file in os.listdir(datasets_path):
        if file.startswith("."):
            continue
        if file.endswith(".jpg"):
            path = os.join.paths(datasets_path, file)
            dirs.append(path)
            
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.2)
    
    X = []

    for path in dirs:
        image = cv.imread(path)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    X.append([landmark.x, landmark.y, landmark.z])
                    
    return X       
            
