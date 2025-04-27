import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os

labels_dict = {}
count = 0

for hand in ['right_hand', 'left_hand']:
    for i in range(26):
        labels_dict[count] = f"{hand}_{chr(65 + i)}"
        count += 1

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

labels = np.array([list(labels_dict.values()).index(label) for label in labels])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None, min_samples_split=2)

required_dirs = ['./data/right_hand', './data/left_hand']
for dir_path in required_dirs:
    if not os.path.exists(dir_path) or len(os.listdir(dir_path)) == 0:
        print(f"Warning: No images found in {dir_path}. Skipping training for that hand.")
        continue

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)