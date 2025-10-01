
import asyncio
import base64
import csv
import logging
import os
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np
import socketio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from cv.mediapipe_to_kinect_match import KINECT_JOINTS, KINECT_TO_MEDIAPIPE, evaluate_pose_fit



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

# Output directory for CSV files
OUTPUT_DIR = "output_csvs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

        pose_fit_score, _ = evaluate_pose_fit(lm) 
        
        print(f"Debug - Pose Fit Score: {pose_fit_score}")


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

        # Save kinect data to CSV
        if sid in session_data and "csv_writer" in session_data[sid]:
            row_data = []
            for joint in KINECT_JOINTS:
                joint_data = kinect_joints_data.get(joint, [None, None, None])
                row_data.extend(joint_data)
            
            # Ensure the writer and file are still valid before writing
            csv_writer = session_data[sid].get("csv_writer")
            csv_file = session_data[sid].get("csv_file")
            if csv_writer and csv_file and not csv_file.closed:
                csv_writer.writerow(row_data)


        return {
            "landmarks": landmarks_for_client,
            "connections": list(mp_pose.POSE_CONNECTIONS),
            "kinect_joints": kinect_joints_data,
            "pose_fit_score": pose_fit_score,
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

        <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            const poseFitScoreEl = document.createElement('p');
            poseFitScoreEl.style.textAlign = 'center';
            poseFitScoreEl.style.fontWeight = 'bold';
            document.body.appendChild(poseFitScoreEl);

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

                if (data && data.pose_fit_score !== undefined) {
                    const score = (data.pose_fit_score * 100).toFixed(2);
                    poseFitScoreEl.textContent = `Pose Fit Score: ${score}%`;
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
    
    # Prepare CSV file for the session
    try:
        file_path = os.path.join(OUTPUT_DIR, f"{sid}.csv")
        csv_file = open(file_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        
        # Write header
        header = []
        for joint in KINECT_JOINTS:
            header.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])
        csv_writer.writerow(header)
        
        session_data[sid] = {
            "last_frame_time": asyncio.get_event_loop().time(),
            "csv_file": csv_file,
            "csv_writer": csv_writer,
        }
        logger.info(f"CSV file created for session {sid} at {file_path}")
    except Exception as e:
        logger.error(f"Failed to create CSV file for session {sid}: {e}", exc_info=True)


@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    if sid in session_data:
        csv_file = session_data[sid].get("csv_file")
        if csv_file and not csv_file.closed:
            csv_file.close()
            logger.info(f"CSV file for session {sid} closed.")
        session_data.pop(sid, None)


@sio.on("frame")
async def handle_frame(sid, data):
    loop = asyncio.get_event_loop()
    try:
        # Offload CPU-bound work to a thread pool
        pose_data = await loop.run_in_executor(None, process_pose_frame, data, sid)
        if pose_data:
            if pose_data.get('landmarks') is not None:
                # landmarks는 리스트 형태이므로, MediaPipe 객체로 재변환 필요
                # 이는 비효율적이므로 process_pose_frame에서 점수를 계산하는 것이 더 좋음
                pass # 이 부분은 다음 단계에서 수정
            await sio.emit("pose_data", pose_data, to=sid)
    except Exception as e:
        logger.error(f"Error in handle_frame for sid {sid}: {e}", exc_info=True)
