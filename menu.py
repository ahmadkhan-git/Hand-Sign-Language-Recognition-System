import tkinter as tk
from tkinter import ttk
import cv2
import pickle
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import time
import pyttsx3

class GestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.cap = None
        self.model = None
        self.hands = None

        self.labels_dict = {}
        count = 0
        for hand in ['right_hand', 'left_hand']:
            for i in range(26):
                self.labels_dict[count] = f"{hand}_{chr(65 + i)}"
                count += 1

        self.last_recognized_time = time.time()
        self.displayed_character = ""
        self.last_display_time = time.time()
        self.is_recognizing = False
        self.frame_counter = 0

        self.setup_gui()
        self.setup_model()

    def setup_gui(self):
        self.root.title("Hand Gesture Recognition")
        self.root.configure(bg="#2b2b2b")
        window_width = 900
        window_height = 650
        x_offset = 200
        y_offset = 100
        self.root.geometry(f"{window_width}x{window_height}+{x_offset}+{y_offset}")

        webcam_frame = tk.Frame(self.root, bg="#333333", padx=10, pady=10, bd=5, relief="ridge")
        webcam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.webcam_label = tk.Label(webcam_frame, bg="#333333")
        self.webcam_label.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(self.root, bg="#2b2b2b", padx=20, pady=20, bd=5, relief="ridge")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        title_label = tk.Label(control_frame, text="Recognized Text", font=("Arial", 18), bg="#2b2b2b", fg="#ffffff")
        title_label.pack(pady=10)

        self.recognized_text_box = tk.Text(control_frame, height=15, width=30, font=("Arial", 14), bg="#4a4a4a", fg="#ffffff", wrap=tk.WORD, bd=3, relief="sunken")
        self.recognized_text_box.pack(pady=10)

        self.create_rounded_button(control_frame, "Start Recognition", self.start_recognition, 20, 2, "#4caf50")
        self.create_rounded_button(control_frame, "Text to Speech", self.text_to_speech, 20, 2, "#2196f3")
        self.create_rounded_button(control_frame, "Clear Text", self.clear_text_box, 20, 2, "#f44336")
        self.create_rounded_button(control_frame, "Exit", self.exit_application, 20, 2, "#808080")

    def setup_model(self):
        try:
            model_dict = pickle.load(open('./model.p', 'rb'))
            self.model = model_dict['model']
        except Exception as e:
            print(f"Error loading model: {e}")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            model_complexity=1
        )

    def create_rounded_button(self, parent, text, command, width, height, color, font_size=12):
        button = tk.Button(parent, text=text, command=command, relief="flat",
                           width=width, height=height, font=("Arial", font_size),
                           fg="white", bg=color, bd=0, highlightthickness=0)
        button.pack(pady=10, padx=10, fill=tk.X)
        return button

    def start_recognition(self):
        if not self.is_recognizing:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
                return

            self.is_recognizing = True
            self.update_frame()

    def update_frame(self):
        if not self.is_recognizing:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        self.frame_counter += 1
        if self.frame_counter % 2 != 0:
            self.root.after(10, self.update_frame)
            return

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

            current_time = time.time()
            if current_time - self.last_recognized_time >= 0.25:
                self.last_recognized_time = current_time
                data_aux = []
                x_ = []
                y_ = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                if len(data_aux) > 0:
                    try:
                        prediction = self.model.predict([np.asarray(data_aux)])
                        prediction_confidence = self.model.predict_proba([np.asarray(data_aux)]).max()

                        if prediction_confidence >= 0.7:
                            full_label = self.labels_dict.get(int(prediction[0]), "Unknown")
                            predicted_character = full_label.split('_')[-1] if full_label != "Unknown" else "Unknown"

                            if current_time - self.last_display_time >= 2:
                                self.displayed_character = predicted_character
                                self.last_display_time = current_time
                                self.recognized_text_box.insert(tk.END, predicted_character)
                                self.recognized_text_box.see(tk.END)
                    except Exception as e:
                        print(f"Error in prediction: {e}")

        if self.displayed_character:
            cv2.putText(frame, self.displayed_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_image = ImageTk.PhotoImage(frame_image)

        self.webcam_label.config(image=frame_image)
        self.webcam_label.image = frame_image

        self.root.after(10, self.update_frame)

    def stop_recognition(self):
        if self.cap:
            self.cap.release()
        self.is_recognizing = False
        cv2.destroyAllWindows()

    def clear_text_box(self):
        self.recognized_text_box.delete(1.0, tk.END)

    def text_to_speech(self):
        text = self.recognized_text_box.get(1.0, tk.END).strip()
        if text:
            tts_engine = pyttsx3.init()
            tts_engine.say(text)
            tts_engine.runAndWait()
        else:
            print("Text box is empty. Nothing to convert to speech.")

    def exit_application(self):
        if self.is_recognizing:
            self.stop_recognition()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureRecognitionApp(root)
    root.mainloop()