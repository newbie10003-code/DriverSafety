import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import pygame

pygame.mixer.init()

# Load the pre-trained Keras models
model1 = load_model('CustomCNNDrowsiness.h5')
model2 = load_model('ResNet50Drowsiness.h5')
model3 = load_model('VGG16Drowsiness.h5')

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_frame(frame):
    eyes = eye_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
    if len(eyes) == 0:
        return None, None

    x, y, w, h = eyes[0]  # Use the first detected eye
    eye = frame[y:y+h, x:x+w]
    eye = cv2.resize(eye, (80, 80))
    eye = img_to_array(eye)
    eye = np.expand_dims(eye, axis=0)
    eye /= 255.0
    return eye, (x, y, w, h)

def extract_eyes_to_array(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None

    for (fx, fy, fw, fh) in faces:
        roi_gray = gray_image[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:
            return None, None
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            resized_eye_img = cv2.resize(eye_img, (80, 80))
            sharpened_image = enhance_image_quality(resized_eye_img)
            sharpened_image = cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2BGR)
            sharpened_image = np.expand_dims(sharpened_image, axis=0)
            sharpened_image = sharpened_image / 255.0
            return sharpened_image, (fx + ex, fy + ey, ew, eh)

def enhance_image_quality(image):
    alpha = 1.0  # Contrast control (1.0-3.0)
    beta = 50   # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)
    return sharpened

def classify_eye(frame):
    processed_eye, eye_coords = extract_eyes_to_array(frame)
    if processed_eye is None:
        return None, None  # Assume awake if no eye is detected
    prediction1 = model1.predict(processed_eye)
    prediction2 = model2.predict(processed_eye)
    prediction3 = model3.predict(processed_eye)
    
    avg_prediction = (prediction1[0][0] + prediction2[0][0] + prediction3[0][0]) / 3.0
    return 1 if avg_prediction >= 0.5 else 0, eye_coords

is_playing = False

def play_alarm():
    pygame.mixer.music.load('alarm.wav')
    global is_playing
    if not is_playing:
        pygame.mixer.music.play()
        is_playing = True

def stop_alarm():
    global is_playing
    if is_playing:
        pygame.mixer.music.stop()
        is_playing = False

st.title("Driver Drowsiness Detection System")

run = st.checkbox('Run Webcam', key='run_webcam')
stop_alarm_button = st.button('Stop Alarm', key='stop_alarm_button')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

sleep_count = 0
start_time = None

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Failed to capture image from webcam. Please check your camera.")
        break

    frame = cv2.flip(frame, 1)
    
    res = classify_eye(frame)
    if res is None:
        continue

    class_idx, eye_coords = res

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

    if eye_coords is not None:
        x, y, w, h = eye_coords
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    status = "Awake" if class_idx == 0 else "Sleeping"
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    FRAME_WINDOW.image(frame, channels="BGR")

    if stop_alarm_button:
        stop_alarm()
        continue

else:
    st.write('Stopped')

if camera.isOpened():
    camera.release()

stop_alarm()
