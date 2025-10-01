import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import os

# Define paths
CSV_PATH = "src/cv/youtube_pose_data_normalized.csv"
MODELS_DIR = "src/cv/models"
SCALER_PATH = os.path.join(MODELS_DIR, "youtube_pose_data_normalize.pkl")

def create_scaler():
    """
    Reads pose data from CSV, fits a MinMaxScaler on 103 features, 
    and saves the scaler to a pickle file.
    """
    print(f"Reading data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found.")
        return

    # Select landmark columns (first 99) and the 4 distance columns
    landmark_columns = df.columns[:99]
    distance_columns = [
        'dist_LeftWrist_LeftAnkle',
        'dist_RightWrist_RightAnkle',
        'dist_Shoulder_Shoulder',
        'dist_Nose_HipMid'
    ]
    feature_columns = list(landmark_columns) + distance_columns
    
    # Check if all columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from the CSV file: {missing_cols}")
        return

    feature_data = df[feature_columns]
    print(f"Found {len(feature_columns)} feature columns for scaling.")

    print("Fitting MinMaxScaler...")
    scaler = MinMaxScaler()
    scaler.fit(feature_data)
    print("Scaler fitted successfully.")

    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Saving scaler to {SCALER_PATH}...")
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Scaler saved successfully!")
    print("The scaler now includes 103 features.")
    print(f"Please ensure your 'youtube_autoencoder_model.h5' file is in the '{MODELS_DIR}' directory and restart the main application.")

if __name__ == "__main__":
    create_scaler()