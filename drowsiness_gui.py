import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import threading
import playsound

# Load the pre-trained Keras model
model = load_model('your_model.h5')

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_frame(frame):
    """Preprocess the frame for the model."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no eyes are detected, return the original frame preprocessed
    if len(eyes) == 0:
        eye = gray
    else:
        # Assume the first detected eye is the driver's eye
        x, y, w, h = eyes[0]
        eye = gray[y:y+h, x:x+w]

    eye = cv2.resize(eye, (64, 64))
    eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    eye = img_to_array(eye)
    eye = np.expand_dims(eye, axis=0)
    eye /= 255.0
    return eye

def classify_eye(frame):
    """Classify the eye using the pre-trained model."""
    processed_eye = preprocess_frame(frame)
    predictions = model.predict(processed_eye)
    class_idx = np.argmax(predictions)
    return class_idx

def play_alarm():
    """Play alarm sound."""
    global alarm_active
    while alarm_active:
        playsound.playsound('alarm.wav')

# Streamlit UI
st.title("Driver Drowsiness Detection System")

run = st.checkbox('Run Webcam')
FRAME_WINDOW = st.image([])

alarm_active = False
alarm_thread = None

def stop_alarm():
    global alarm_active
    alarm_active = False
    if alarm_thread is not None:
        alarm_thread.join()

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
    class_idx = classify_eye(frame)

    if class_idx == 1:  # Assuming class 1 means eyes closed
        if start_time is None:
            start_time = time.time()
        else:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 3:
                sleep_count += 1
                start_time = None
                if not alarm_active:
                    alarm_active = True
                    alarm_thread = threading.Thread(target=play_alarm)
                    alarm_thread.start()
    else:
        start_time = None
        sleep_count = 0
        stop_alarm()

    # Display the frame and the status
    status = "Awake" if class_idx == 0 else "Sleeping"
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    FRAME_WINDOW.image(frame, channels="BGR")

    # Stop alarm button
    if st.button('Stop Alarm'):
        stop_alarm()

else:
    st.write('Stopped')

if camera.isOpened():
    camera.release()

# Ensure the alarm thread is stopped when the app exits
stop_alarm()
