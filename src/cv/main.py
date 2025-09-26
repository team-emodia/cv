
import asyncio
import base64
import logging
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np
import socketio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from cv.mediapipe_to_kinect_match import KINECT_JOINTS, KINECT_TO_MEDIAPIPE

# -----------------------
# 1) Initialization
# -----------------------
# FastAPI app
app = FastAPI()

# Socket.IO server
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session data
session_data: Dict[str, Dict] = {}


# -----------------------
# 2) Helper Functions
# -----------------------
def mediapipe_landmark_to_xyz(landmark, image_w, image_h):
    return np.array(
        [
            landmark.x * image_w,
            landmark.y * image_h,
            landmark.z * 1000.0,  # Scale z for numeric stability
        ]
    )


def process_pose_frame(image_data: str, sid: str):
    try:
        # Decode image
        img_decoded = base64.b64decode(image_data.split(",")[1])
        img_np = np.frombuffer(img_decoded, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("Received empty frame")
            return None

        image_h, image_w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return {"landmarks": None, "connections": None}

        lm = results.pose_landmarks.landmark
        landmarks_for_client = [
            {"x": l.x, "y": l.y, "z": l.z, "visibility": l.visibility} for l in lm
        ]

        # Kinect matching logic (simplified for real-time)
        left_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        chest_x = ((left_sh.x + right_sh.x) / 2.0) * image_w
        chest_y = ((left_sh.y + right_sh.y) / 2.0) * image_h
        chest_z = ((left_sh.z + right_sh.z) / 2.0) * 1000.0
        chest_center = np.array([chest_x, chest_y, chest_z])

        kinect_joints_data = {}
        for j in KINECT_JOINTS:
            mp_idx = KINECT_TO_MEDIAPIPE.get(j)
            if mp_idx is not None:
                xyz = mediapipe_landmark_to_xyz(lm[mp_idx], image_w, image_h)
                xyz_norm = xyz - chest_center
                kinect_joints_data[j] = xyz_norm.tolist()

        return {
            "landmarks": landmarks_for_client,
            "connections": list(mp_pose.POSE_CONNECTIONS),
            "kinect_joints": kinect_joints_data,
        }

    except Exception as e:
        logger.error(f"Error processing frame for sid {sid}: {e}", exc_info=True)
        return None


# -----------------------
# 3) FastAPI routes
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Pose Estimation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
            h1 { text-align: center; }
            #container { display: flex; justify-content: center; gap: 20px; }
            #video-container, #canvas-container { border: 1px solid #ccc; background: #fff; }
            video, canvas { display: block; }
        </style>
    </head>
    <body>
        <h1>Real-time Pose Estimation with FastAPI and MediaPipe</h1>
        <div id="container">
            <div id="video-container">
                <video id="video" width="640" height="480" autoplay playsinline></video>
            </div>
            <div id="canvas-container">
                <canvas id="canvas" width="640" height="480"></canvas>
            </div>
        </div>
        <pre id="kinect-data"></pre>

        <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const kinectDataEl = document.getElementById('kinect-data');

            const socket = io(window.location.origin);

            socket.on('connect', () => {
                console.log('Connected to server', socket.id);
            });

            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                    video.addEventListener('loadeddata', () => {
                        setInterval(() => {
                            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                            const dataURL = canvas.toDataURL('image/jpeg', 0.5);
                            socket.emit('frame', dataURL);
                        }, 100); // Send frame every 100ms
                    });
                })
                .catch(err => console.error("Error accessing camera:", err));

            socket.on('pose_data', (data) => {
                // Redraw video frame
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                if (data && data.landmarks) {
                    // Draw landmarks
                    ctx.fillStyle = '#00FF00';
                    data.landmarks.forEach(lm => {
                        if (lm.visibility > 0.5) {
                            ctx.beginPath();
                            ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
                            ctx.fill();
                        }
                    });

                    // Draw connections
                    if (data.connections) {
                        ctx.strokeStyle = '#FF0000';
                        ctx.lineWidth = 2;
                        data.connections.forEach(conn => {
                            const start = data.landmarks[conn[0]];
                            const end = data.landmarks[conn[1]];
                            if (start.visibility > 0.5 && end.visibility > 0.5) {
                                ctx.beginPath();
                                ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
                                ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
                                ctx.stroke();
                            }
                        });
                    }
                }
                if (data && data.kinect_joints) {
                    kinectDataEl.textContent = JSON.stringify(data.kinect_joints, null, 2);
                }
            });

            socket.on('disconnect', () => {
                console.log('Disconnected from server');
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# -----------------------
# 4) Socket.IO events
# -----------------------
@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    session_data[sid] = {"last_frame_time": asyncio.get_event_loop().time()}


@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    session_data.pop(sid, None)


@sio.on("frame")
async def handle_frame(sid, data):
    loop = asyncio.get_event_loop()
    try:
        # Offload CPU-bound work to a thread pool
        pose_data = await loop.run_in_executor(None, process_pose_frame, data, sid)
        if pose_data:
            await sio.emit("pose_data", pose_data, to=sid)
    except Exception as e:
        logger.error(f"Error in handle_frame for sid {sid}: {e}", exc_info=True)

