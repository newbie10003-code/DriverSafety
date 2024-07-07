import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import threading
import pygame
from playsound import playsound

# Load the pre-trained Keras model with error handling
def load_model_safe(filepath):
    try:
        model = load_model(filepath)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model_safe('./VGG16Drowsiness.h5')

if model is None:
    st.stop()  # Stop the app if the model fails to load

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_frame(frame):
    """Preprocess the frame for the model and return eye coordinates."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(eyes) == 0:
        return None, None

    x, y, w, h = eyes[0]  # Use the first detected eye
    eye = gray[y:y+h, x:x+w]
    eye = cv2.resize(eye, (64, 64))
    eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
    eye = img_to_array(eye)
    eye = np.expand_dims(eye, axis=0)
    eye /= 255.0
    return eye, (x, y, w, h)

def classify_eye(frame):
    """Classify the eye using the pre-trained model and return eye coordinates."""
    processed_eye, eye_coords = preprocess_frame(frame)
    if processed_eye is None:
        return 0, None  # Assume awake if no eye is detected
    predictions = model.predict(processed_eye)
    class_idx = 1 if predictions[0][0] >= 0.5 else 0
    return class_idx, eye_coords

is_playing = False

def play_alarm():
    global is_playing
    if not is_playing:
        pygame.mixer.music.play()
        is_playing = True

def stop_alarm():
    global is_playing
    if is_playing:
        pygame.mixer.music.stop()
        is_playing = False

# Streamlit UI
st.title("Driver Drowsiness Detection System")

run = st.checkbox('Run Webcam', key='run_webcam')
stop_alarm_button = st.button('Stop Alarm', key='stop_alarm_button')
FRAME_WINDOW = st.image([])

alarm_active = False
alarm_thread = None

camera = cv2.VideoCapture(0)

sleep_count = 0
start_time = None

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Failed to capture image from webcam. Please check your camera.")
        break

    frame = cv2.flip(frame, 1)
    
    # Classify the current frame
    class_idx, eye_coords = classify_eye(frame)

    if class_idx == 1:  # Assuming class 1 means sleeping
        if start_time is None:
            start_time = time.time()
        else:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 3:
                sleep_count += 1
                start_time = None
                play_alarm()
    else:
        start_time = None
        sleep_count = 0
        stop_alarm()

    # Draw a box around the detected eye
    if eye_coords is not None:
        x, y, w, h = eye_coords
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame and the status
    status = "Awake" if class_idx == 0 else "Sleeping"
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    FRAME_WINDOW.image(frame, channels="BGR")

    # Stop alarm button
    if stop_alarm_button:
        stop_alarm()

else:
    st.write('Stopped')

if camera.isOpened():
    camera.release()

# Ensure the alarm thread is stopped when the app exits
stop_alarm()
