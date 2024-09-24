import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import threading
import logging
import winsound  # For Windows, use a different library for other OS

cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("crime_detection_model.h5")

lm_list = []
label = "normal"
confidence = 0.0

logging.basicConfig(filename='crime_detection.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame


def draw_class_on_image(label, confidence, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 30)
    fontScale = 1
    thickness = 2

    if label == "normal":
        color = (0, 255, 0)  # Green
    elif label == "suspicious":
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red

    cv2.putText(img, f"{label} ({confidence:.2f})", org, font, fontScale, color, thickness)
    return img


def detect(model, lm_list):
    global label, confidence
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    pred_index = np.argmax(result[0])
    if pred_index == 0:
        label = "normal"
    elif pred_index == 1:
        label = "suspicious"
    else:
        label = "violent"
    confidence = result[0][pred_index]
    return label, confidence


def sound_alarm():
    frequency = 2500  # Set frequency To 2500 Hertz
    duration = 1000  # Set duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)


i = 0
warm_up_frames = 60

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    i += 1
    if i > warm_up_frames:
        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == 20:
                t1 = threading.Thread(target=detect, args=(model, lm_list))
                t1.start()
                lm_list = []

            frame = draw_landmark_on_image(mpDraw, results, frame)
            frame = draw_class_on_image(label, confidence, frame)

            if label != "normal":
                logging.info(f"Detected: {label} with confidence {confidence:.2f}")
                if label == "violent":
                    threading.Thread(target=sound_alarm).start()

    cv2.imshow("Crime Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()