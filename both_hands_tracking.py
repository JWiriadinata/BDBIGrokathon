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

# ---------------------------------------------------------------------------
# Detection tuning constants (shared by both arms)
# ---------------------------------------------------------------------------
MIN_EXT = 0.8               # extension from guard needed to start a punch
MIN_THROWING_SPEED = 7      # wrist speed needed to start a punch
GUARD_MAX_SPEED = 3         # wrist must be slower than this to update the guard
GUARD_WINDOW = 150          # frames of guard samples (5 seconds at 30fps)
SCALE_WINDOW = 150
WRIST_POS_WINDOW = 5        # frames used for the velocity fit
RECOVER_T = 0.1             # min seconds after the peak before returning to idle
THROWING_TIMEOUT = 1        # seconds before a stuck throw is dropped
RECOVER_TIMEOUT = 2         # seconds before a stuck recovery is forced to idle
PEAK_DISPLAY_T = 0.5        # seconds a detected-punch circle stays on screen

scale_buffer = []

def draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(image, p, 3, (0, 0, 255), -1)


def lm_to_array(lm):
    return np.array([lm.x, lm.y, lm.z])


def pos_to_pixels(x, y, image):
    h, w, _ = image.shape
    return (int(x * w), int(y * h))


def average_velocity(data):
    times, positions = zip(*data)
    t = np.array(times)
    t = t - t[0]   # make time 0 the time of the first data point
    return np.polyfit(t, np.array(positions), 1)[0]  # vx vy vz slopes


class PunchTracker:
    """Detects punches for a single wrist."""

    def __init__(self, name, wrist_id, shoulder_id, color=(0, 255, 0)):
        self.name = name
        self.wrist_id = wrist_id
        self.shoulder_id = shoulder_id
        self.color = color  # BGR, used for the punch circle and state text

        self.state = "idle"

        self.guard_pos = None
        self.recent_guard_wrist_rel = []
        self.recent_wrist_rel = []  # (t, wrist_rel) samples for velocity

        self.max_punch_ext = 0
        self.max_punch_speed = 0
        self.peak_t = 0
        self.peak_pos = None

        self.throwing_start = 0
        self.recover_start = 0

        self.last_ext = 0
        self.last_speed = 0

        self.punch_peaks = []  # (peak_t, peak_screen_pos)

    def update(self, t, landmarks, scale, image):
        wrist = landmarks[self.wrist_id]
        shoulder = landmarks[self.shoulder_id]
        wrist_screen_pos = pos_to_pixels(wrist.x, wrist.y, image)

        wrist_rel = (lm_to_array(wrist) - lm_to_array(shoulder)) / scale

        self.recent_wrist_rel.append((t, wrist_rel))
        if len(self.recent_wrist_rel) > WRIST_POS_WINDOW:
            self.recent_wrist_rel.pop(0)

        extension = (
            np.linalg.norm(wrist_rel - self.guard_pos)
            if self.guard_pos is not None else 0
        )

        if len(self.recent_wrist_rel) == WRIST_POS_WINDOW:
            wrist_vel = average_velocity(self.recent_wrist_rel)  # as a vector
        else:
            wrist_vel = np.zeros(3)
        speed = np.linalg.norm(wrist_vel)

        self.last_ext = extension
        self.last_speed = speed

        if self.state == "idle":
            guard_ready = len(self.recent_guard_wrist_rel) == GUARD_WINDOW

            if extension > MIN_EXT and speed > MIN_THROWING_SPEED and guard_ready:
                self.state = "throwing"
                self.max_punch_ext = extension
                self.max_punch_speed = speed

                # Initialize peak info so a quick drop-off can't emit a stale/None peak
                self.peak_t = t
                self.peak_pos = wrist_screen_pos

                self.throwing_start = t

                other = [tr for tr in trackers if tr is not self][0]
                print(f"{self.name} start: ext {extension:.2f} speed {speed:.1f} | "
      f"other ext {other.last_ext:.2f} speed {other.last_speed:.1f} | "
      f"vis {landmarks[other.wrist_id].visibility:.3f}")

            elif speed < GUARD_MAX_SPEED and extension < MIN_EXT * 0.5:
                self.recent_guard_wrist_rel.append(wrist_rel)
                if len(self.recent_guard_wrist_rel) > GUARD_WINDOW:
                    self.recent_guard_wrist_rel.pop(0)

                self.guard_pos = np.median(self.recent_guard_wrist_rel, axis=0)

        elif self.state == "throwing":
            if extension > self.max_punch_ext:  # record peak info
                self.max_punch_ext = extension
                self.peak_t = t
                self.peak_pos = wrist_screen_pos

            self.max_punch_speed = max(self.max_punch_speed, speed)

            if extension < self.max_punch_ext * 0.9 and speed < self.max_punch_speed * 0.75:
                self.punch_peaks.append((self.peak_t, self.peak_pos))
                #print(f"{self.name} punch thrown at {t:.2f}")
                self.state = "recovering"
                self.recover_start = t

            elif t - self.throwing_start > THROWING_TIMEOUT:  # break out of stale throwing state
                self.state = "recovering"
                self.recover_start = t

        elif self.state == "recovering":
            if ((extension < MIN_EXT * 0.75 and t - self.peak_t > RECOVER_T)
                    or t - self.recover_start > RECOVER_TIMEOUT):
                self.state = "idle"

    def is_idle(self):
        return self.state == "idle"

    def draw_peaks(self, image, t):
        # Drop old peaks so the list doesn't grow forever
        self.punch_peaks = [p for p in self.punch_peaks if t - p[0] < PEAK_DISPLAY_T]
        for _, position in self.punch_peaks:
            cv2.circle(image, position, 30, self.color, 3)

    def draw_state_text(self, image, y):
        text = f"{self.name}: {self.state}"
        cv2.putText(image, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, self.color, 2, cv2.LINE_AA)


options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,  # synchronous, keeps the loop structure the same
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

start_time = time.time()

# MediaPipe's "left"/"right" are the person's own left/right.
trackers = [
    PunchTracker("L", LEFT_WRIST, LEFT_SHOULDER, color=(0, 255, 0)),
    PunchTracker("R", RIGHT_WRIST, RIGHT_SHOULDER, color=(0, 165, 255)),
]

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
        timestamp_ms = int(t * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Keep the BGR frame for display
        image = frame.copy()

        # results.pose_landmarks is a list of poses; each pose is a list of landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]

            # Shoulder reference shared by both arms


            if all(tr.is_idle() for tr in trackers):
                l_shoulder = lm_to_array(landmarks[LEFT_SHOULDER])
                r_shoulder = lm_to_array(landmarks[RIGHT_SHOULDER])
                currScale = np.linalg.norm(r_shoulder - l_shoulder)  # shoulder width
                
                scale_buffer.append(currScale)

                if len(scale_buffer) > SCALE_WINDOW:
                    scale_buffer.pop(0)

            if len(scale_buffer) > 0:
                scale = np.median(scale_buffer)

                for tracker in trackers:
                    tracker.update(t, landmarks, scale, image)

            # Draw pose landmarks on the image
            draw_landmarks(image, landmarks)

        for tracker in trackers:
            tracker.draw_peaks(image, t)

        # Mirror the image horizontally
        image = cv2.flip(image, 1)

        for i, tracker in enumerate(trackers):
            tracker.draw_state_text(image, 100 + 40 * i)

        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()