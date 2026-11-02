import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Model setup (the Tasks API needs a .task model file; legacy solutions bundled it)
# ---------------------------------------------------------------------------
MODEL_PATH = "pose_landmarker_lite.task"  # swap for _full / _heavy for more accuracy
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
if not os.path.exists(MODEL_PATH):
    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Landmark indices (same as the old mp_pose.PoseLandmark enum)
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

# Skeleton connections for drawing (replaces mp_pose.POSE_CONNECTIONS / drawing_utils)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(image, p, 3, (0, 0, 255), -1)


options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,  # synchronous, keeps the loop structure the same
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

left_data_list = []
time_list = []

cap = cv2.VideoCapture(0)

start_time = time.time()

saved_frame = None

class PunchTracker:

    def update(wristPos, shoulderPos):
        print(wristPos)


left_tracker = PunchTracker()

state = "idle"

guard_pos = None
min_ext = 0.8

guard_window = 150 # last 5 seconds at 30fps

wrist_pos_window = 5 # shorter for velocity

recent_guard_wrist_rel = []
recent_wrist_rel = []

min_throwing_speed = 7

max_punch_ext = 0
max_punch_speed = 0
peak_t = 0
peak_pos = None

recover_t = 0.1

throwing_timeout = 1
recover_timeout = 2

throwing_start = 0
recover_start = 0

punch_peaks = []

def average_velocity(data):
    times, positions = zip(*data)
    t = np.array(times)
    t = t - t[0]   # make time 0 the time of the first data point
    return np.polyfit(t, np.array(positions), 1)[0]  # vx vy vz slopes

def draw_state_text(image):
    # 2. Define text and parameters
    text = "State = " + state
    coordinates = (50, 100)  # (X, Y) position of the bottom-left corner of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)      # BGR format (Green)
    thickness = 2

    # 3. Add text to the image
    cv2.putText(image, text, coordinates, font, font_scale, color, thickness, cv2.LINE_AA)

def pos_to_pixels(x, y, image):
    h, w, _ = image.shape
    return (int(x * w), int(y * h))

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t = time.time() - start_time

        # Convert BGR to RGB and wrap in a MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs a monotonically increasing timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Keep the BGR frame for display
        image = frame.copy()

        # results.pose_landmarks is a list of poses; each pose is a list of landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]

            # Get shoulder and wrist positions
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            left_wrist = landmarks[LEFT_WRIST]

            l_wrist = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
            l_shoulder = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
            
            r_shoulder = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])

            mid = (l_shoulder + r_shoulder) / 2
            scale = np.linalg.norm(r_shoulder - l_shoulder)        # shoulder width
            wrist_rel = (l_wrist - mid) / scale

            recent_wrist_rel.append((t, wrist_rel))
            if len(recent_wrist_rel) > wrist_pos_window:
                recent_wrist_rel.pop(0)

            extension = np.linalg.norm(wrist_rel - guard_pos) if guard_pos is not None else 0

            recent_wrist_vel = average_velocity(recent_wrist_rel) if len(recent_wrist_rel) == wrist_pos_window else [0,0,0]# as a vector
            recent_speed = np.linalg.norm(recent_wrist_vel)

            #print(recent_speed, extension)
           
            if state == "idle":

                if extension > min_ext and recent_speed > min_throwing_speed and len(recent_guard_wrist_rel) == guard_window:
                    state = "throwing"
                    max_punch_ext = extension
                    max_punch_speed = recent_speed

                    peak_pos = pos_to_pixels(left_wrist.x, left_wrist.y, image)
                    
                    throwing_start = t

                elif recent_speed < 3 and extension < min_ext * 0.5:
                    recent_guard_wrist_rel.append(wrist_rel)
                    if len(recent_guard_wrist_rel) > guard_window:
                        recent_guard_wrist_rel.pop(0)

                    guard_pos = np.median(recent_guard_wrist_rel, axis=0)
                            
            elif state == "throwing":
                if extension > max_punch_ext: # record peak info
                    max_punch_ext = extension
                    peak_t = t

                    peak_pos = pos_to_pixels(left_wrist.x, left_wrist.y, image)

                max_punch_speed = max(max_punch_speed, recent_speed)

                if extension < max_punch_ext * 0.9 and recent_speed < max_punch_speed * 0.75:
                    punch_peaks.append((peak_t, peak_pos))
                    print("punch thrown at " + str(round(t, 2)))
                    state = "recovering"
                    recover_start = t

                elif t - throwing_start > throwing_timeout:     # break out of stale throwing state
                    state = "recovering"
                    recover_start = t

            elif state == "recovering":
                if (extension < min_ext * 0.75 and t - peak_t > recover_t) or t - recover_start > recover_timeout:
                    state = "idle"
            

            # Draw pose landmarks on the image
            draw_landmarks(image, results.pose_landmarks[0])

        for punch in punch_peaks:
            punch_t = punch[0]

            if t - punch_t < 0.5:
                position = punch[1]

                image = cv2.circle(image, position, 30, (0, 255, 0), 3)

        # Mirror the image horizontally
        image = cv2.flip(image, 1)


        draw_state_text(image)


        cv2.imshow('Pose Detection', image)
            

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()