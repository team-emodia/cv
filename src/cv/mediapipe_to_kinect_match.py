"""
mediapipe_to_kinect_match.py

기능:
 - MediaPipe Pose로부터 3D 관절 좌표 추출
 - 학습된 모델을 사용하여 자세 적합도 평가
"""

import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
L = mp_pose.PoseLandmark

# 9개의 주요 랜드마크 정의
PRIMARY_LANDMARKS = {
    'nose': L.NOSE,
    'left_shoulder': L.LEFT_SHOULDER,
    'right_shoulder': L.RIGHT_SHOULDER,
    'left_elbow': L.LEFT_ELBOW,
    'right_elbow': L.RIGHT_ELBOW,
    'left_wrist': L.LEFT_WRIST,
    'right_wrist': L.RIGHT_WRIST,
    'left_hip': L.LEFT_HIP,
    'right_hip': L.RIGHT_HIP,
}

def get_coords(landmarks, landmark_index):
    """Helper to get coordinates from the landmark list."""
    lm = landmarks[landmark_index]
    return np.array([lm.x, lm.y, lm.z])

def calculate_distance_features(landmarks):
    """Calculates 4 custom distance features from MediaPipe landmarks."""
    def get_dist(p1_idx, p2_idx):
        p1 = get_coords(landmarks, p1_idx)
        p2 = get_coords(landmarks, p2_idx)
        return np.linalg.norm(p1 - p2)

    # Note: These distance feature names from the CSV were ambiguous.
    # Using the most likely interpretation.
    dist_lw_lh = get_dist(L.LEFT_WRIST, L.LEFT_HIP)
    dist_rw_rh = get_dist(L.RIGHT_WRIST, L.RIGHT_HIP)
    dist_sh_sh = get_dist(L.LEFT_SHOULDER, L.RIGHT_SHOULDER)
    
    left_hip_coords = get_coords(landmarks, L.LEFT_HIP)
    right_hip_coords = get_coords(landmarks, L.RIGHT_HIP)
    hip_mid = (left_hip_coords + right_hip_coords) / 2
    nose_coords = get_coords(landmarks, L.NOSE)
    dist_nose_hip = np.linalg.norm(nose_coords - hip_mid)

    return np.array([dist_lw_lh, dist_rw_rh, dist_sh_sh, dist_nose_hip])

def evaluate_pose_fit(landmarks, model, scaler):
    """
    Evaluates the pose fit using the model and scaler, ensuring a 34-feature vector.
    """
    if not landmarks or not model or not scaler:
        return 0.0, 0.0

    try:
        # 1. Extract coordinates for the 9 primary landmarks (27 features)
        pose_features = []
        for lm_name in PRIMARY_LANDMARKS.keys():
            lm_index = PRIMARY_LANDMARKS[lm_name]
            lm = landmarks[lm_index]
            pose_features.extend([lm.x, lm.y, lm.z])

        # 2. Calculate and add the 'spine_shoulder' midpoint (3 features)
        left_shoulder_coords = get_coords(landmarks, L.LEFT_SHOULDER)
        right_shoulder_coords = get_coords(landmarks, L.RIGHT_SHOULDER)
        spine_shoulder_coords = (left_shoulder_coords + right_shoulder_coords) / 2
        pose_features.extend(spine_shoulder_coords)

        # 3. Calculate the 4 custom distance features
        distance_features = calculate_distance_features(landmarks)

        # 4. Combine into the final 34-feature vector
        all_features = np.concatenate([pose_features, distance_features]).reshape(1, -1)

        # 5. Normalize with the scaler
        normalized_data = scaler.transform(all_features)

        # 6. Evaluate with the model
        reconstructed_data = model.predict(normalized_data)
        reconstruction_error = np.mean(np.square(normalized_data - reconstructed_data))

        max_error = 2.0
        fit_score = max(0.0, 1.0 - (reconstruction_error / max_error))

        return fit_score, reconstruction_error

    except Exception as e:
        print(f"Error during pose fit evaluation: {e}")
        return 0.0, 0.0
