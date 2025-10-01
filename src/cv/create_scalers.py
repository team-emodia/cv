import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Define base paths
base_dir = os.path.dirname(os.path.abspath(__file__))
answer_csv_dir = os.path.join(base_dir, 'answer_csv')
models_dir = os.path.join(base_dir, 'models')

os.makedirs(models_dir, exist_ok=True)

# Define the feature columns based on the model's expected input (34 features)
# 9 landmarks * 3 coords + 1 calculated landmark * 3 coords + 4 distances
landmark_cols_from_csv = [
    'nose_x', 'nose_y', 'nose_z',
    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
    'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
    'left_elbow_x', 'left_elbow_y', 'left_elbow_z',
    'right_elbow_x', 'right_elbow_y', 'right_elbow_z',
    'left_wrist_x', 'left_wrist_y', 'left_wrist_z',
    'right_wrist_x', 'right_wrist_y', 'right_wrist_z',
    'left_hip_x', 'left_hip_y', 'left_hip_z',
    'right_hip_x', 'right_hip_y', 'right_hip_z',
]

distance_cols = [
    'dist_left_wrist_hip',
    'dist_right_wrist_hip',
    'dist_shoulder_shoulder',
    'dist_nose_hip_mid'
]

parts = ['neck_left', 'neck_right', 'shoulder_left', 'shoulder_right']

for part in parts:
    csv_path = os.path.join(answer_csv_dir, f'{part}_labeled.csv')
    scaler_path = os.path.join(models_dir, f'{part}_scaler.pkl')

    print(f"Processing {csv_path}...")

    if not os.path.exists(csv_path):
        print(f"  -> Error: CSV file not found at {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # --- Feature Engineering: Create the 34 features ---
    
    # 1. Start with the 27 landmark coordinates from the CSV
    features_df = df[landmark_cols_from_csv].copy()

    # 2. Calculate and add the 'spine_shoulder' midpoint (3 features)
    features_df['spine_shoulder_x'] = (df['left_shoulder_x'] + df['right_shoulder_x']) / 2
    features_df['spine_shoulder_y'] = (df['left_shoulder_y'] + df['right_shoulder_y']) / 2
    features_df['spine_shoulder_z'] = (df['left_shoulder_z'] + df['right_shoulder_z']) / 2

    # 3. Add the 4 distance features from the CSV
    for col in distance_cols:
        features_df[col] = df[col]

    # Now we have 27 + 3 + 4 = 34 features
    print(f"  -> Engineered {features_df.shape[1]} features.")

    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(features_df)

    print(f"  -> Scaler fitted successfully.")

    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"  -> Scaler saved to {scaler_path}")

print("\nAll scalers have been created.")
