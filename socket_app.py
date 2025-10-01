
import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import base64
import io

# --- Path and Model Loading ---

# Set base_dir to the directory of this file, which is the project root
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from src.cv.mediapipe_to_kinect_match import evaluate_pose_fit

# Function to load model and scaler
def load_resources(model_path, scaler_path):
    try:
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        print(f"Failed to load resources: {e}")
        return None, None

# Specific resources for this page
PART_NAME = "neck_left"
models_dir = os.path.join(base_dir, 'src', 'cv', 'models')
model_path = os.path.join(models_dir, f'{PART_NAME}_autoencoder_model.h5')
scaler_path = os.path.join(models_dir, f'{PART_NAME}_scaler.pkl')

model, scaler = load_resources(model_path, scaler_path)

# --- MediaPipe and Font Setup ---

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
font_path = os.path.join(base_dir, 'src', 'cv', 'NanumGothic.ttf')
font = ImageFont.truetype(font_path, 40)

# --- FastAPI and Socket.IO Setup ---

app = FastAPI()

@app.get("/")
async def read_index():
    return FileResponse('index.html')

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pose Processing Function ---

def process_image(img, model, scaler):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    annotated_image = img.copy()

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        if model and scaler:
            fit_score, reconstruction_error = evaluate_pose_fit(
                results.pose_landmarks.landmark, model, scaler
            )
            
            min_error = 70.0
            max_error = 150.0
            if (max_error - min_error) > 0:
                normalized_error = (reconstruction_error - min_error) / (max_error - min_error)
            else:
                normalized_error = 0
            
            score = (1 - normalized_error) * 100
            score = max(0.0, min(100.0, score))
            
            if score >= 90:
                score_text = "참 잘했어요"
            elif score >= 70:
                score_text = "잘했어요"
            else:
                score_text = "자세를 확인하세요"
            
            pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            draw.text((10, 60), score_text, font=font, fill=(0, 255, 0, 255))
            annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    else:
        pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text((10, 30), "No pose detected", font=font, fill=(0, 0, 255, 255))
        annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return annotated_image

# --- Socket.IO Event Handlers ---

@sio.event
async def connect(sid, environ):
    print('connect ', sid)

@sio.event
async def image(sid, data):
    # Decode the base64 image
    img_data = base64.b64decode(data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Process the image
    processed_frame = process_image(frame, model, scaler)

    # Encode the processed image back to base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    # Send the processed image back to the client
    await sio.emit('response', 'data:image/jpeg;base64,' + jpg_as_text, to=sid)

@sio.event
async def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    uvicorn.run(socket_app, host='0.0.0.0', port=8000)
