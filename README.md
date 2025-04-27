# Hand Sign Language Recognition System

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues Welcome](https://img.shields.io/badge/Issues-Welcome-brightgreen.svg)](https://github.com/your-username/hand-sign-language-recognition-system/issues)
[![Made with Love](https://img.shields.io/badge/Made%20with-Love-red.svg)](https://github.com/your-username/hand-sign-language-recognition-system)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Overview

The **Hand Sign Language Recognition System** is a complete pipeline that allows users to **collect hand gesture images**, **train a machine learning model**, and **recognize gestures in real-time** through a webcam feed.  
It aims to bridge communication gaps by providing a simple and practical hand sign recognition system.

---

## Features

- Collect A–Z hand gesture images for both **right** and **left** hands.
- Landmark detection using **MediaPipe Hands**.
- Train a **Random Forest Classifier** to predict gestures.
- **Real-time** hand gesture recognition through webcam.
- User-friendly **Tkinter GUI**.
- **Text-to-Speech** functionality for recognized letters.
- Supports **individual hand training** for improved accuracy.

---

## Project Structure

| File Name            | Purpose |
|----------------------|---------|
| `collect_imgs.py`     | Collect images for each hand gesture class (A-Z). |
| `create_dataset.py`   | Extract landmarks and generate dataset from collected images. |
| `train_classifier.py` | Train and save the machine learning model. |
| `menu.py`             | Launch the GUI for real-time gesture recognition. |
| `data/`               | Folder where collected images are saved. |
| `data.pickle`         | Dataset containing landmark features and labels. |
| `model.p`             | Saved trained Random Forest model. |

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/hand-sign-language-recognition-system.git
   cd hand-sign-language-recognition-system
   ```

2. **Install Required Packages:**
   ```bash
   pip install opencv-python mediapipe scikit-learn numpy pillow pyttsx3
   ```

---

## How to Use

1. **Collect Images:**
   ```bash
   python collect_imgs.py
   ```
   - Choose Right or Left hand.
   - Capture images for each alphabet A-Z.

2. **Create Dataset:**
   ```bash
   python create_dataset.py
   ```
   - Generates a `.pickle` file with extracted landmarks.

3. **Train Classifier:**
   ```bash
   python train_classifier.py
   ```
   - Trains the Random Forest Classifier and saves the model.

4. **Launch GUI:**
   ```bash
   python menu.py
   ```
   - Start recognition, view recognized text, and convert text to speech!

---

## Technical Details

- **Hand Landmark Extraction:**  
  Using **MediaPipe Hands** to detect and extract 21 keypoints per hand.

- **Model Architecture:**  
  Random Forest Classifier from Scikit-learn.

- **Threshold for Prediction:**  
  Displays predictions only if confidence ≥ 70%.

- **GUI Framework:**  
  Built using **Tkinter** for simplicity and ease of use.

---

## Future Improvements

- Extend system to recognize **words** or **sentences**.
- Integrate **Deep Learning models** (CNNs, LSTMs).
- Support for more complex **dynamic gestures**.
- Improve the user interface and recognition speed.
- Add **language options** for text-to-speech.

---

## Acknowledgements

- **OpenCV** for real-time computer vision.
- **MediaPipe** for easy and efficient hand tracking.
- **Scikit-learn** for model training and evaluation.
- **Tkinter** for creating a clean and simple GUI.
- **pyttsx3** for offline text-to-speech support.

---

## License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute this project for personal and commercial purposes.

[Read the License here.](https://opensource.org/licenses/MIT)

---

> **Maintainer:** Ahmad Akeel Khan  
> Contributions, suggestions, and issue reports are warmly welcome!  
> Let's build more accessible technologies together!

---
