import streamlit as st
import cv2
import numpy as np
import tensorflow.keras as keras  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

# Load the pre-trained Keras model
model = load_model('VGG16Distraction.h5')
categories = ["Safe Driving", "Texting - Right", "Talking on the phone - Right", "Texting - Left", "Talking on the phone - Left", "Operating the Radio", "Drinking", "Reaching Behind", "Hair and Makeup", "Talking to Passenger"]

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_frame(frame):
    """Preprocess the frame for the model."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no face is detected, return the original frame preprocessed
    if len(faces) == 0:
        face = gray
    else:
        # Assume the first detected face is the driver's face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (64, 64))
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face /= 255.0
    return face

def classify_frame(frame):
    """Classify the frame using the pre-trained model."""
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    class_idx = np.argmax(predictions)
    return categories[class_idx]

# Streamlit UI
st.title("Driver Distraction Detection")

run = st.checkbox('Run Webcam')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Failed to capture image from webcam. Please check your camera.")
        break

    frame = cv2.flip(frame, 1)
    
    # Classify the current frame
    predicted_class = classify_frame(frame)

    # Display the frame and the prediction
    cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    FRAME_WINDOW.image(frame, channels="BGR")

else:
    st.write('Stopped')

if camera.isOpened():
    camera.release()