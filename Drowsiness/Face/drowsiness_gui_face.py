import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import threading
import playsound
import pygame

# Load the pre-trained Keras model
model = load_model('CustomCNNDrowsiness.h5')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_frame(frame):
    """Preprocess the frame for the model."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (64, 64))
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame /= 255.0
    return frame

def classify_frame(frame):
    """Classify the frame using the pre-trained model."""
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    class_idx = np.argmax(predictions)
    return class_idx

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
    
    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # Classify the face region
        class_idx = classify_frame(face)

        if class_idx == 1:  # Assuming class 1 means drowsiness detected
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

        # Draw a rectangle around the face
        color = (0, 255, 0) if class_idx == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Display the frame and the status
    status = "Awake" if sleep_count == 0 else "Drowsy"
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
