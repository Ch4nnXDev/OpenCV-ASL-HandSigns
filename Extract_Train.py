import os
import cv2 as cv
import mediapipe as mp

def get_paths(dataset_path):
    if dataset_path is None:
        raise ValueError("Dataset path cannot be None")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
    
    return [os.path.join(dataset_path, dir) for dir in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, dir)) and not dir.startswith(".")]

def get_image_paths(dir_list):
    image_paths = []
    for dir in dir_list:
        for file in os.listdir(dir):
            if file.startswith("."):
                continue
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(dir, file))
    return image_paths

def extract_landmarks():
    X = []
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    dataset_path = "./asl_alphabet_train"
    dir_list = get_image_paths(get_paths(dataset_path))
    
    for img in dir_list:
        image = cv.imread(img)
        if image is None:
            print(f"Warning: Unable to read image {img}")
            continue
        result = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    X.append([landmark.x, landmark.y, landmark.z])
        else:
            print(f"Warning: No hand landmarks found in image {img}")
    return X    
        

extract_landmarks()