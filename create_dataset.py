import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

hand_dirs = ['right_hand', 'left_hand']

data = []
labels = []
num_landmarks = 21

for hand_dir in hand_dirs:
    hand_path = os.path.join(DATA_DIR, hand_dir)
    for class_dir in os.listdir(hand_path):
        class_path = os.path.join(hand_path, class_dir)
        for img_path in os.listdir(class_path):
            data_aux = []

            x_ = []
            y_ = []

            try:
                img = cv2.imread(os.path.join(class_path, img_path))
                if img is None:
                    raise FileNotFoundError(f"Image {img_path} not found.")
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    if len(x_) < num_landmarks:
                        diff = num_landmarks - len(x_)
                        data_aux.extend([0] * diff * 2)

                    data.append(data_aux)
                    labels.append(f"{hand_dir}_{class_dir}")
            except Exception as e:
                print(f"Skipping image {img_path} due to error: {e}")
                continue

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)