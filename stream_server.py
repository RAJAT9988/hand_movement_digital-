#!/usr/bin/env python3
"""
Web server: streams webcam (with hand/face detection) to a webpage.
Run this script, then open http://127.0.0.1:5000 in your browser.
"""

import os
import time
import urllib.request
from threading import Lock

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_from_directory

# MediaPipe Tasks
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")

MOVE_THRESHOLD = 15
SMOOTH_FRAMES = 5

app = Flask(__name__, template_folder=os.path.join(SCRIPT_DIR, "templates"))

# Global camera and detectors (lazy init)
_camera = None
_hand_landmarker = None
_face_detector = None
_hand_connections = None
_lock = Lock()
_hand_positions = []
_last_direction = ""
_direction_start_time = 0
_frame_timestamp_ms = 0
_face_detected = False


def download_model(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        return
    print(f"Downloading model to {path} ...")
    urllib.request.urlretrieve(url, path)
    print("Done.")


def init_detectors():
    global _hand_landmarker, _face_detector, _hand_connections
    with _lock:
        if _hand_landmarker is not None:
            return
        download_model(HAND_MODEL_URL, HAND_MODEL_PATH)
        download_model(FACE_MODEL_URL, FACE_MODEL_PATH)
        hand_base = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        _hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        face_base = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        face_options = vision.FaceDetectorOptions(
            base_options=face_base,
            min_detection_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        _face_detector = vision.FaceDetector.create_from_options(face_options)
        _hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS


def get_camera():
    global _camera
    with _lock:
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(0)
        return _camera


def generate_frames():
    global _hand_positions, _last_direction, _direction_start_time, _frame_timestamp_ms, _face_detected
    init_detectors()
    cap = get_camera()
    if not cap.isOpened():
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"\r\n")
        return

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
        face_result = _face_detector.detect_for_video(mp_image, _frame_timestamp_ms)
        _face_detected = bool(face_result.detections)
        if face_result.detections:
            for detection in face_result.detections:
                b = detection.bounding_box
                x1 = max(0, b.origin_x)
                y1 = max(0, b.origin_y)
                x2 = min(w, b.origin_x + b.width)
                y2 = min(h, b.origin_y + b.height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame, "Face", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
                )

        # Hand detection and movement
        hand_result = _hand_landmarker.detect_for_video(mp_image, _frame_timestamp_ms)
        current_direction = ""
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, _hand_connections,
                    vision.drawing_styles.get_default_hand_landmarks_style(),
                    vision.drawing_styles.get_default_hand_connections_style(),
                )
                wrist = hand_landmarks[0]
                palm_x = int(wrist.x * w)
                palm_y = int(wrist.y * h)
                _hand_positions.append((palm_x, palm_y))
        else:
            _hand_positions.clear()

        if len(_hand_positions) > SMOOTH_FRAMES:
            _hand_positions.pop(0)
        if len(_hand_positions) >= 2:
            x_old, y_old = _hand_positions[0]
            x_new, y_new = _hand_positions[-1]
            dx, dy = x_new - x_old, y_new - y_old
            if abs(dx) > MOVE_THRESHOLD or abs(dy) > MOVE_THRESHOLD:
                if abs(dy) >= abs(dx):
                    current_direction = "DOWN" if dy > 0 else "UP"
                else:
                    current_direction = "RIGHT" if dx > 0 else "LEFT"
                _last_direction = current_direction
                _direction_start_time = time.time()

        if current_direction or (time.time() - _direction_start_time < 1.0 and _last_direction):
            display_dir = current_direction or _last_direction
            (tw, th), _ = cv2.getTextSize(display_dir, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            tx, ty = (w - tw) // 2, 80
            cv2.putText(frame, display_dir, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            cv2.putText(frame, "Hand movement", (tx, ty - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        _frame_timestamp_ms += 33
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/person.glb")
def serve_person_glb():
    path = os.path.join(SCRIPT_DIR, "person.glb")
    if not os.path.isfile(path):
        return "person.glb not found", 404
    return send_from_directory(SCRIPT_DIR, "person.glb", mimetype="model/gltf-binary")


@app.route("/api/face")
def api_face():
    return jsonify(faceDetected=_face_detected)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print("Starting server. Open http://127.0.0.1:5000 in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
