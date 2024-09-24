import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import pandas as pd
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1,
                   enable_segmentation=True)
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "normal2"  # Change this "violent" when collecting different data
no_of_frames = 2000


def make_landmark_timestep(results):
    c_lm = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm


def draw_landmark_on_image(mpDraw, results, frame):
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for lm in results.pose_landmarks.landmark:
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame


# Create a named window
cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)

while len(lm_list) <= no_of_frames:
    # Capture the screen
    screen = np.array(pyautogui.screenshot())
    frame = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

    # Process the frame with MediaPipe
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = make_landmark_timestep(results)
        lm_list.append(lm)
        frame = draw_landmark_on_image(mpDraw, results, frame)

    cv2.putText(frame, f"Collecting {label} data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Frames: {len(lm_list)}/{no_of_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame in the named window
    cv2.imshow("Screen Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Add a small delay to reduce CPU usage
    time.sleep(0.01)

df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")

cv2.destroyAllWindows()