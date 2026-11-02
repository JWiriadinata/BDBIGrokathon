import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
left_data_list = []
time_list = []

cap = cv2.VideoCapture(0)

start_time = time.time()
last_time_entry = 0.0
current_extension_left = 0.0  # Track latest extensionLeft value

# For live peak detection
last_value = None
peak_cooldown = 0  # Prevent detecting multiple peaks too quickly

# settings
vel_history_window = 4
start_threshold = 1.5
end_threshold = 0.8

# For velocity-based detection
wrist_history = []
is_punching = False  # Track if we're in a punch motion
punch_start_extension = None  # Extension at start of punch

last_wrist_pos = None
last_ts = 0

punch_dir = None
signed_speed_history = []

peak_vel = 0

start_time = time.perf_counter()

punch_cooldown_timer = 0

hitX, hitY = 0, 0

max_newer = 0

def average_velocity(data):
    times, positions = zip(*data)
    t = np.array(times)
    t = t - t[0]   # make time 0 the time of the first data point
    return np.polyfit(t, np.array(positions), 1)[0]  # vx vy vz slopes

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t = time.perf_counter() - start_time
        dt = t - last_ts

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame
        results = pose.process(image)

        # Convert back to BGR for display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks and results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark

            wrist_screen = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

            # Get wrist positions
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            left_wrist_pos = np.array([left_wrist.x, left_wrist.y, left_wrist.z])

            
            # check signed average vel for turn around

            # if average velocity of last few frames is over a threshold then start punch
            # if punching, check if average falls below low threshold - if yes: detect punch

            if last_wrist_pos is not None and dt > 0:

                #curr_vel = calculate_wrist_vel(left_wrist_pos, dt)

                wrist_history.append((t, left_wrist_pos))

                if len(wrist_history) > vel_history_window:
                    wrist_history.pop(0)

                if len(wrist_history) == vel_history_window:

                    avg_older = average_velocity(wrist_history[:-2])
                    avg_newer = average_velocity(wrist_history[-3:])

                    older_mag = np.linalg.norm(avg_older)

                    if older_mag > 0.1: # filter out noise
                        avg_dir = avg_older / older_mag

                        h, w, _ = image.shape
                                                
                        wristX = int(wrist_screen.x * w)
                        wristY = int(wrist_screen.y * h) 

                        endX = int(avg_dir[0] * 100 * older_mag + wristX)
                        endY = int(avg_dir[1] * 100 * older_mag + wristY)

                        cv2.arrowedLine(image, (wristX, wristY), (endX, endY), (255, 0, 0), max(3, int(avg_dir[2] * older_mag * 3)))

                        signed_speed = np.dot(avg_newer, avg_dir)

                        newer_speed = np.linalg.norm(avg_newer)

                        
                        max_newer = max(newer_speed, max_newer)

                        if not is_punching and punch_cooldown_timer <= 0:
                            if newer_speed > start_threshold:
                                is_punching = True

                                print("punch started")
                        
                        if is_punching:
                            if signed_speed <= -0.1:
                                is_punching = False
                                punch_cooldown_timer = 0.5
                        
                                h, w, _ = image.shape
                        
                                hitX = int(wrist_screen.x * w)
                                hitY = int(wrist_screen.y * h)           
                        
                                print("Punch detected")

                if punch_cooldown_timer > 0:
                    image = cv2.circle(image, (hitX, hitY), 30, (0, 255, 0))
            else:
                wrist_history.clear()
                is_punching = False
                punch_cooldown_timer = 0

            punch_cooldown_timer -= dt
            # Print positions every frame
            # print(f"Left Shoulder:  x={left_shoulder.x:.3f}, y={left_shoulder.y:.3f}, z={left_shoulder.z:.3f}")
            # print(f"Right Shoulder: x={right_shoulder.x:.3f}, y={right_shoulder.y:.3f}, z={right_shoulder.z:.3f}")
            # print(f"Left Wrist:     x={left_wrist.x:.3f}, y={left_wrist.y:.3f}, z={left_wrist.z:.3f}")
            # print(f"Right Wrist:    x={right_wrist.x:.3f}, y={right_wrist.y:.3f}, z={right_wrist.z:.3f}")
            # print("-" * 50)

            last_wrist_pos = left_wrist_pos
            last_ts = t
            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Mirror the image horizontally
        image = cv2.flip(image, 1)
        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(max_newer)