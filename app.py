#!/usr/bin/env python3
"""
Hand movement + face detection (no Ultralytics).
Uses MediaPipe Tasks API: hand landmarker + face detector.
Opens webcam and shows: hand skeleton, hand direction (UP/DOWN/LEFT/RIGHT), face boxes.
Press 'q' to quit.
"""

import os
import time
import urllib.request

import cv2
import numpy as np

# MediaPipe Tasks (works with mediapipe >= 0.10.31 where solutions was removed)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model URLs and local paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")

MOVE_THRESHOLD = 15
SMOOTH_FRAMES = 5


def download_model(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        return
    print(f"Downloading model to {path} ...")
    urllib.request.urlretrieve(url, path)
    print("Done.")


def main():
    download_model(HAND_MODEL_URL, HAND_MODEL_PATH)
    download_model(FACE_MODEL_URL, FACE_MODEL_PATH)

    # Hand landmarker (VIDEO mode for webcam)
    hand_base = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    # Face detector (VIDEO mode)
    face_base = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    face_options = vision.FaceDetectorOptions(
        base_options=face_base,
        min_detection_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    face_detector = vision.FaceDetector.create_from_options(face_options)

    # Hand connections for drawing (must be Connection objects, not tuples)
    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    hand_positions = []
    last_direction = ""
    direction_start_time = 0
    frame_timestamp_ms = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_contiguous = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contiguous)

            # Face detection
            face_result = face_detector.detect_for_video(mp_image, frame_timestamp_ms)
            if face_result.detections:
                for detection in face_result.detections:
                    b = detection.bounding_box
                    x1 = max(0, b.origin_x)
                    y1 = max(0, b.origin_y)
                    x2 = min(w, b.origin_x + b.width)
                    y2 = min(h, b.origin_y + b.height)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame, "Face",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2,
                    )

            # Hand detection and movement
            hand_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            current_direction = ""

            if hand_result.hand_landmarks:
                for hand_landmarks in hand_result.hand_landmarks:
                    vision.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        hand_connections,
                        vision.drawing_styles.get_default_hand_landmarks_style(),
                        vision.drawing_styles.get_default_hand_connections_style(),
                    )
                    # Use wrist (index 0) for movement
                    wrist = hand_landmarks[0]
                    palm_x = int(wrist.x * w)
                    palm_y = int(wrist.y * h)
                    hand_positions.append((palm_x, palm_y))
            else:
                hand_positions.clear()

            if len(hand_positions) > SMOOTH_FRAMES:
                hand_positions.pop(0)

            if len(hand_positions) >= 2:
                x_old, y_old = hand_positions[0]
                x_new, y_new = hand_positions[-1]
                dx = x_new - x_old
                dy = y_new - y_old
                if abs(dx) > MOVE_THRESHOLD or abs(dy) > MOVE_THRESHOLD:
                    if abs(dy) >= abs(dx):
                        current_direction = "DOWN" if dy > 0 else "UP"
                    else:
                        current_direction = "RIGHT" if dx > 0 else "LEFT"
                    last_direction = current_direction
                    direction_start_time = time.time()

            if current_direction or (
                time.time() - direction_start_time < 1.0 and last_direction
            ):
                display_dir = current_direction or last_direction
                (tw, th), _ = cv2.getTextSize(
                    display_dir, cv2.FONT_HERSHEY_SIMPLEX, 2, 3
                )
                tx = (w - tw) // 2
                ty = 80
                cv2.putText(
                    frame, display_dir,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 255), 3,
                )
                cv2.putText(
                    frame, "Hand movement",
                    (tx, ty - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2,
                )

            cv2.putText(
                frame,
                "Hands: green  |  Face: blue  |  Press 'q' to quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

            cv2.imshow("Hand & Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_timestamp_ms += 33  # ~30 fps
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_detector.close()


if __name__ == "__main__":
    main()
