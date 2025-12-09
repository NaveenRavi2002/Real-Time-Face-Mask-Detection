import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load Model
model = load_model("mask_detector.h5")

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("Face Mask Detection App 😷")

# ================================
# FUNCTION: Detect mask in a frame
# ================================
def detect_mask(image):

    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))
        face_resized = face_resized.astype("float32") / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        pred = model.predict(face_resized)[0][0]

        label = "Mask" if pred < 0.5 else "No Mask"
        color = (0, 255, 0) if pred < 0.5 else (0, 0, 255)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return img


# ================================
# INTERFACE — Two Buttons
# ================================
start = st.button("Open Camera")
stop = st.button("Close Camera")

cam = None

if start:
    cam = cv2.VideoCapture(0)
    st.write("Camera Started...")

    frame_window = st.image([])

    while True:
        ret, frame = cam.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        result = detect_mask(img)

        frame_window.image(result)

        if stop:
            break

    cam.release()
    st.write("Camera Stopped")

