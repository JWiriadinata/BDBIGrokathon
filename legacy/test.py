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

PUNCH_CLASSES = {0: 'hook', 1: 'jab', 2: 'uppercut'}

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
last_time_entry = 0.0
current_extension_left = 0.0  # Track latest extensionLeft value

# For live peak detection
detected_peaks = []  # Store detected peak times to avoid duplicates
last_value = None
peak_cooldown = 0  # Prevent detecting multiple peaks too quickly

punch_timeout = 1
punch_start_time = 0

# For velocity-based detection
extension_history = []  # Store recent extensions for smoothing
is_punching = False  # Track if we're in a punch motion
punch_start_extension = None  # Extension at start of punch

# Buffer to store last 15 frames of coordinates for classification
coords_buffer = []  # Each entry: [x1,y1,x2,y2,x3,y3,x4,y4] = RIGHT_WRIST, LEFT_WRIST, RIGHT_ELBOW, LEFT_ELBOW

saved_frame = None


hitX, hitY = 0, 0

def average_velocity(data):
    times, extensions = zip(*data)
    t = np.array(times)
    t = t - t[0]   # make time 0 the time of the first data point
    return np.polyfit(t, np.array(extensions), 1)[0]  # vx vy vz slopes

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB and wrap in a MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs a monotonically increasing timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Keep the BGR frame for display
        image = frame.copy()

        # results.pose_landmarks is a list of poses; each pose is a list of landmarks
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks[0]

            # Get shoulder and wrist positions
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            left_wrist = landmarks[LEFT_WRIST]
            right_wrist = landmarks[RIGHT_WRIST]
            left_elbow = landmarks[LEFT_ELBOW]
            right_elbow = landmarks[RIGHT_ELBOW]

            left_wrist_array = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
            left_shoulder_array = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])

            right_shoulder_array = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])

            shoulder_width = abs(np.linalg.norm(left_shoulder_array - right_shoulder_array))
            
            extension_left = abs(np.linalg.norm(left_wrist_array - left_shoulder_array))
            current_extension_left = extension_left  # Update current value

            # Check if 0.1 seconds has passed since last time entry
            elapsed = time.time() - start_time
            while elapsed >= last_time_entry + 0.1:
                last_time_entry += 0.1
                time_list.append(round(last_time_entry, 1))
                left_data_list.append(current_extension_left)

            # Live peak detection using velocity (rate of change)
            if peak_cooldown > 0:
                image = cv2.circle(image, (hitX, hitY), 30, (0, 255, 0))
                peak_cooldown -= 1

            # Store extension history for smoothing
            extension_history.append((elapsed, extension_left))
            if len(extension_history) > 10:
                extension_history.pop(0)

            # Calculate velocity (change in extension)
            if len(extension_history) >= 5:
                # velocity = extension_left - last_value
                # velocity_history.append(velocity)
                # if len(velocity_history) > 10:
                #     velocity_history.pop(0)

                avg_velocity = average_velocity(extension_history)

                #print(avg_velocity)

                # max_recent_velocity = max(velocity_history[-5:]) if velocity_history[-5:] else 0

                # Velocity threshold - needs significant outward movement to start punch detection
                velocity_threshold = 0.4  # Adjust this based on testing

                # Start tracking a punch when we see fast outward movement
                if not is_punching and avg_velocity > velocity_threshold and peak_cooldown <= 0:
                    is_punching = True
                    punch_start_extension = extension_left
                    punch_start_time = elapsed

                # If we're in a punch, look for the peak (velocity goes from positive to negative)
                if is_punching:
                    # Check if we've reached the peak (velocity becomes negative after being positive)
                    if avg_velocity < -0.002 and extension_left > punch_start_extension * 1.3:
                        # This is a real punch peak!
                        current_time = round(time.time() - start_time, 1)

                        # Classify the punch using last 15 frames
                        print(f"\n*** PUNCH detected at {current_time}s! Extension: {extension_left:.4f} ***")

                        detected_peaks.append(('PUNCH', current_time, extension_left))

                        saved_frame = image  # reference, so the skeleton drawn below also appears on it

                        is_punching = False
                        punch_start_extension = None
                        peak_cooldown = 30  # Cooldown to avoid duplicate detections

                    # Reset if extension dropped below start (aborted punch or arm returned)
                    elif extension_left < punch_start_extension - 0.02 or elapsed - punch_start_time > punch_timeout:
                        is_punching = False
                        punch_start_extension = None

            # Draw pose landmarks on the image
            draw_landmarks(image, results.pose_landmarks[0])

        h, w, _ = image.shape

        thickness = 10
        color = (0, 255, 0) if is_punching else (0, 0, 255)

        image = cv2.rectangle(image, (0, 0), (w, h), color, thickness)

        # Mirror the image horizontally saved_frame is not None
        if False:
            cv2.imshow('Pose Detection', saved_frame)
        else:
            image = cv2.flip(image, 1)
            cv2.imshow('Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\n=== LIVE DETECTED PEAKS ===")
for peak_type, t, v in detected_peaks:
    print(f"  Time: {t}s, Extension: {v:.4f} ({peak_type})")

print("\nTime list:", time_list)
print("Left data list:", left_data_list)
df = pd.DataFrame({"extension": left_data_list}, index=time_list)

# Calculate dynamic threshold based on data spread
data_std = np.std(left_data_list)
data_median = np.median(left_data_list)
prominence_threshold = data_std * 1.5  # Only detect peaks that stand out significantly

# Find prominent local maxima
maxima, max_props = find_peaks(df["extension"], prominence=prominence_threshold, distance=5)
# Find prominent local minima (invert the signal)
minima, min_props = find_peaks(-df["extension"], prominence=prominence_threshold, distance=5)

# Filter maxima: only keep those significantly ABOVE median (true punch extensions)
# Filter minima: only keep those significantly BELOW median (true retractions)
valid_maxima = [i for i in maxima if left_data_list[i] > data_median + data_std]
valid_minima = [i for i in minima if left_data_list[i] < data_median - data_std]

# Combine and sort by time
all_peaks = sorted(valid_maxima + valid_minima)
peak_times = [time_list[i] for i in all_peaks]
peak_values = [left_data_list[i] for i in all_peaks]

print(f"\nMedian: {data_median:.4f}, Std: {data_std:.4f}")
print(f"Max threshold: > {data_median + data_std:.4f}")
print(f"Min threshold: < {data_median - data_std:.4f}")
print("\nPunch peaks detected at times:")
for t, v in zip(peak_times, peak_values):
    peak_type = "MAX" if v > data_median else "MIN"
    print(f"  Time: {t}s, Extension: {v:.4f} ({peak_type})")

df.plot(title="DataFrame Plot")
# Plot maxima in red, minima in blue
max_times = [time_list[i] for i in valid_maxima]
max_values = [left_data_list[i] for i in valid_maxima]
min_times = [time_list[i] for i in valid_minima]
min_values = [left_data_list[i] for i in valid_minima]

plt.scatter(max_times, max_values, color='r', s=100, zorder=5, label='Maxima')
plt.scatter(min_times, min_values, color='b', s=100, zorder=5, label='Minima')
plt.axhline(y=data_median, color='gray', linestyle='-', alpha=0.5, label='Median')
plt.axhline(y=data_median + data_std, color='r', linestyle='--', alpha=0.3)
plt.axhline(y=data_median - data_std, color='b', linestyle='--', alpha=0.3)
plt.legend()
plt.show()