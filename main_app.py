import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av
import numpy as np
import os
import pickle
import sys
from tensorflow.keras.models import load_model

# --- Path and Model Loading ---

# Set base_dir to the directory of this file, which is the project root
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from src.cv.mediapipe_to_kinect_match import evaluate_pose_fit

# Function to load model and scaler, with caching for performance
@st.cache_resource
def load_resources(model_path, scaler_path):
    try:
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Failed to load resources: {e}")
        return None, None

# Specific resources for this page
PART_NAME = "neck_left"
models_dir = os.path.join(base_dir, 'src', 'cv', 'models')
model_path = os.path.join(models_dir, f'{PART_NAME}_autoencoder_model.h5')
scaler_path = os.path.join(models_dir, f'{PART_NAME}_scaler.pkl')

model, scaler = load_resources(model_path, scaler_path)

# --- MediaPipe and Streamlit UI Setup ---

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.set_page_config(page_title="Left Neck Evaluation", page_icon="脖")
st.markdown(f"# {PART_NAME.replace('_', ' ').title()} 실시간 평가")
st.write("웹캠을 통해 실시간으로 자세를 평가합니다. 자세를 취하면 점수가 표시됩니다.")

# CSS 수정 (확대 방지: object-fit contain)
st.markdown(
    """
    <style>
    video {
        object-fit: contain !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- WebRTC Video Processing Class ---

class PoseEvaluator(VideoTransformerBase):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.reconstruction_error = 0.0
        self.min_error = 70.0  # Fixed value for a "good" pose
        self.max_error = 150.0 # Fixed value for a "bad" pose

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Horizontally flip the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        annotated_image = img.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            if self.model and self.scaler:
                fit_score, self.reconstruction_error = evaluate_pose_fit(
                    results.pose_landmarks.landmark, self.model, self.scaler
                )
            
            # Remap the error to a 0-100 score based on the min and max error settings
            if (self.max_error - self.min_error) > 0:
                normalized_error = (self.reconstruction_error - self.min_error) / (self.max_error - self.min_error)
            else:
                normalized_error = 0
            
            score = (1 - normalized_error) * 100
            score = max(0.0, min(100.0, score)) # Clamp score between 0 and 100
            
            score_text = f"Score: {score:.1f}"
            cv2.putText(annotated_image, score_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(annotated_image, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return annotated_image

# --- Main App Logic ---

if model and scaler:
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        webrtc_streamer(
            key=f"{PART_NAME}-evaluation",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: PoseEvaluator(model=model, scaler=scaler),
            media_stream_constraints={
                "video": {
                    # 해상도는 카메라 네이티브(1280x720 또는 1920x1080) 권장
                    "width": {"ideal": 1920, "max": 1920},
                    "height": {"ideal": 1080, "max": 1080},
                    "aspectRatio": 16/9,  # 센서 기본 비율
                    "frameRate": {"ideal": 30, "max": 30},
                    "advanced": [
                        {"focusMode": "manual"},
                        {"autoFocus": False},
                        {"exposureMode": "manual"},
                        {"whiteBalanceMode": "manual"},
                        {"zoom": 1.0}
                    ]
                },
                "audio": False
            },
            async_processing=True,
        )

else:
    st.error("모델 또는 스케일러를 로드하지 못했습니다. 앱을 실행할 수 없습니다.")
