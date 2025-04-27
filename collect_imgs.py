import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

hands_dirs = {'right': os.path.join(DATA_DIR, 'right_hand'), 'left': os.path.join(DATA_DIR, 'left_hand')}
for hand_dir in hands_dirs.values():
    if not os.path.exists(hand_dir):
        os.makedirs(hand_dir)

dataset_size = 100

print("Choose the hand for which you want to collect images:")
print("1. Right Hand")
print("2. Left Hand")
choice = int(input("Enter your choice (1 or 2): "))

if choice == 1:
    selected_hand = 'right'
elif choice == 2:
    selected_hand = 'left'
else:
    print("Invalid choice. Exiting...")
    exit()

progress_file = f"last_label_{selected_hand}.txt"

def get_last_completed_index():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            last_label = f.read().strip()
            if last_label and last_label.isalpha():
                return ord(last_label.upper()) - 64  # 'A' = 65
    return 0  # Start from A

start_index = get_last_completed_index()
cap = cv2.VideoCapture(0)

# Loop through A-Z (65 to 90 ASCII)
for j in range(start_index, 26):
    letter = chr(65 + j)  # A to Z
    class_dir = os.path.join(hands_dirs[selected_hand], letter)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    img_count = len(os.listdir(class_dir))
    if img_count >= dataset_size:
        print(f"Skipping '{letter}' - already has {img_count} images.")
        continue

    print(f'Press "Q" to start capturing for {selected_hand} hand, class {letter}')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        cv2.putText(frame, f'Hand: {selected_hand.capitalize()}, Alphabet: {letter}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, 'Press "Q" to start capture', (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            print(f"Starting image capture for {selected_hand} hand, Alphabet: {letter}")
            break

    counter = img_count
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        cv2.putText(frame, f'Capturing: {selected_hand.capitalize()}, Alphabet: {letter}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        img_name = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_name, frame)
        print(f"Image {counter} captured for {selected_hand} hand, class {letter}")
        counter += 1

    with open(progress_file, "w") as f:
        f.write(letter)

cap.release()
cv2.destroyAllWindows()
print("Data collection completed!")
