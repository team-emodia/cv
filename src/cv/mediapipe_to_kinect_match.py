"""
mediapipe_to_kinect_match.py

요구: pip install opencv-python mediapipe numpy pandas scipy

기능:
 - MediaPipe Pose로부터 3D(approx) 관절 좌표 추출 (x,y in normalized image coords, z relative)
 - Kinect dataset joint set으로 근사 매핑
 - chest 기준으로 정규화 (Kinect README 방식에 맞춘 shift)
 - velocity 계산 (v[n] = (x[n] - x[n-5]) / 5)
 - dataset CSV (POSITION_JOINT.csv / VELOCITY_JOINT.csv / label.csv) 와 매칭 (시간 보간 및 RMSE/DTW 계산)

주의:
 - MediaPipe 좌표는 정규화된 이미지 좌표(x,y in [0,1]) + z (relative). 실제 미터 단위가 아니므로,
   비교를 위해 동일 스케일로 맞추거나 상대 거리(예: 관절간 거리 비율)로 평가하는 것이 현실적입니다.
"""

import math
from collections import deque

import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

mp_pose = mp.solutions.pose

# -----------------------
# 1) MediaPipe -> Kinect 관절 매핑 (근사)
# -----------------------
# README에 있는 Kinect 관절 이름 목록 중 핵심만 매핑 (추후 필요시 확장)
KINECT_JOINTS = [
    "Head",
    "Neck",
    "SpineShoulder",
    "SpineMid",
    "SpineBase",
    "ShoulderRight",
    "ShoulderLeft",
    "HipRight",
    "HipLeft",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HandTipRight",
    "ThumbRight",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "HandTipLeft",
    "ThumbLeft",
    "KneeRight",
    "AnkleRight",
    "FootRight",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
]

# MediaPipe landmark indices (mp.solutions.pose.PoseLandmark) for convenience
L = mp_pose.PoseLandmark

# 근사 매핑: Kinect joint -> MediaPipe landmark (or interpolation of multiple)
KINECT_TO_MEDIAPIPE = {
    "Head": L.NOSE,  # approximate
    "Neck": L.LEFT_SHOULDER,  # MediaPipe doesn't have explicit neck; use shoulder midpoint later
    "SpineShoulder": L.LEFT_SHOULDER,  # approx
    "SpineMid": L.LEFT_HIP,  # rough approx (we'll compute midpoint)
    "SpineBase": L.RIGHT_HIP,  # use hips midpoint for base
    "ShoulderRight": L.RIGHT_SHOULDER,
    "ShoulderLeft": L.LEFT_SHOULDER,
    "HipRight": L.RIGHT_HIP,
    "HipLeft": L.LEFT_HIP,
    "ElbowRight": L.RIGHT_ELBOW,
    "WristRight": L.RIGHT_WRIST,
    "HandRight": L.RIGHT_INDEX,  # index finger base ~ hand
    "HandTipRight": L.RIGHT_INDEX,
    "ThumbRight": L.RIGHT_THUMB,
    "ElbowLeft": L.LEFT_ELBOW,
    "WristLeft": L.LEFT_WRIST,
    "HandLeft": L.LEFT_INDEX,
    "HandTipLeft": L.LEFT_INDEX,
    "ThumbLeft": L.LEFT_THUMB,
    "KneeRight": L.RIGHT_KNEE,
    "AnkleRight": L.RIGHT_ANKLE,
    "FootRight": L.RIGHT_FOOT_INDEX,
    "KneeLeft": L.LEFT_KNEE,
    "AnkleLeft": L.LEFT_ANKLE,
    "FootLeft": L.LEFT_FOOT_INDEX,
}


# -----------------------
# 2) Helper functions
# -----------------------
def mediapipe_landmark_to_xyz(landmark, image_w, image_h):
    """
    MediaPipe landmark -> approximate Cartesian coordinates (x,y,z)
    x,y normalized to image pixels; z is relative (negative is forward)
    We'll return normalized [x,y,z] where x,y in pixels, z in meters-ish (keep as provided).
    """
    return np.array(
        [landmark.x * image_w, landmark.y * image_h, landmark.z * 1000.0]
    )  # scale z for numeric stability


# -----------------------
# 3) Load Kinect dataset trial CSV
# -----------------------
def load_kinect_position_csv(path):
    """
    Expect CSV format: Column order per README: x,y,z,timestamp
    For a single joint file: position_joints.csv (one joint).
    Return numpy array (T,3) and times (T,)
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 4:
        raise ValueError("Expected at least 4 columns: x,y,z,timestamp")
    coords = df.iloc[:, 0:3].to_numpy(dtype=float)
    times = df.iloc[:, 3].to_numpy(dtype=float)
    return coords, times


def compute_velocity_from_positions(positions, method="kinect_formula"):
    """
    positions: ndarray (T,3)
    Returns velocities ndarray (T,3) using formula v[n] = (x[n] - x[n-5]) / 5
    For first few frames where index<5, fill with zeros or NaN
    """
    T = positions.shape[0]
    v = np.full_like(positions, np.nan)
    if method == "kinect_formula":
        for n in range(T):
            if n >= 5:
                v[n] = (positions[n] - positions[n - 5]) / 5.0
            else:
                v[n] = np.zeros(3)
    else:
        # numerical derivative
        dt = 1.0
        v[1:] = (positions[1:] - positions[:-1]) / dt
        v[0] = np.zeros(3)
    return v


# -----------------------
# 4) Time-sync / interpolation utilities
# -----------------------
def interp_sequence_to_times(seq_arr, seq_times, target_times):
    """
    seq_arr: dict joint -> (T,3)
    seq_times: (T,) ms
    target_times: (M,) ms
    Return dict joint->(M,3)
    """
    out = {}
    for j, arr in seq_arr.items():
        # for each coordinate x,y,z interpolate separately
        # handle NaNs by linear interpolation/extrapolate
        interp_coords = np.zeros((len(target_times), 3), dtype=float)
        for k in range(3):
            y = arr[:, k]
            t = seq_times
            # if all NaN -> fill NaN
            if np.all(np.isnan(y)):
                interp_coords[:, k] = np.nan
                continue
            # mask NaNs
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                # not enough to interp
                interp_coords[:, k] = np.nan
            else:
                f = interp1d(
                    t[valid],
                    y[valid],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                interp_coords[:, k] = f(target_times)
        out[j] = interp_coords
    return out


# -----------------------
# 5) Matching metrics: RMSE, per-joint and overall, and simple DTW
# -----------------------
def per_joint_rmse(seqA, seqB):
    """
    seqA/B: dict joint -> (T,3)
    Assume same length T
    Return dict joint->rmse and overall_mean_rmse
    """
    joints = seqA.keys()
    rmses = {}
    vals = []
    for j in joints:
        a = seqA[j]
        b = seqB[j]
        mask = ~np.isnan(a).any(axis=1) & ~np.isnan(b).any(axis=1)
        if mask.sum() == 0:
            rmses[j] = np.nan
            continue
        diff = a[mask] - b[mask]
        mse = np.mean(np.sum(diff ** 2, axis=1))
        rmses[j] = math.sqrt(mse)
        vals.append(rmses[j])
    overall = np.nanmean(vals) if len(vals) > 0 else np.nan
    return rmses, overall


def dtw_distance(ts_a, ts_b):
    """
    Simple DTW between two 1D sequences
    """
    n, m = len(ts_a), len(ts_b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(ts_a[i - 1] - ts_b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw[n, m]


def joint_dtw(seqA, seqB, joint):
    """
    Compute DTW on Euclidean norm time-series for a single joint
    seqA/B[joint] -> (T,3)
    """
    a = seqA[joint]
    b = seqB[joint]
    # norms (skip NaNs)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    # remove NaNs by simple masking (this is crude but OK for example)
    mask_a = ~np.isnan(na)
    mask_b = ~np.isnan(nb)
    na = na[mask_a]
    nb = nb[mask_b]
    if len(na) == 0 or len(nb) == 0:
        return np.nan
    return dtw_distance(na, nb)