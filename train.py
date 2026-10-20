import os
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import h5py

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# Point to the actual user-collected dataset inside data
LOG_PATH = os.path.join(DATA_DIR, "driving_log.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")

INPUT_W, INPUT_H = 64, 32  # Preprocessed resolution for feature vector

def preprocess_image(img):
    """
    Preprocess image for self-driving model:
    1. Crop sky (top 30%) and car hood (bottom 15%)
    2. Resize to 64x32
    3. Convert BGR to YUV color space
    4. Flatten and normalize features to range [-1.0, 1.0]
    """
    if img is None:
        return None
    h, w, _ = img.shape
    crop_top = int(h * 0.3)
    crop_bottom = int(h * 0.85)
    img_cropped = img[crop_top:crop_bottom, :]
    
    img_resized = cv2.resize(img_cropped, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    img_yuv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV)
    
    # Flatten & normalize
    features = (img_yuv.astype(np.float32) / 127.5) - 1.0
    return features.flatten()

def save_mlp_to_h5(mlp, filepath):
    """
    Export trained Multi-Layer Perceptron Neural Network architecture, 
    weights, biases, and metadata to standard HDF5 (.h5) format.
    """
    with h5py.File(filepath, 'w') as h5f:
        h5f.attrs['model_type'] = 'MLPRegressor'
        h5f.attrs['activation'] = mlp.activation
        h5f.attrs['input_shape'] = (INPUT_H, INPUT_W, 3)
        h5f.attrs['num_layers'] = len(mlp.coefs_)
        
        coefs_grp = h5f.create_group('coefs')
        for i, weight in enumerate(mlp.coefs_):
            coefs_grp.create_dataset(f'w_{i}', data=weight)
            
        intercepts_grp = h5f.create_group('intercepts')
        for i, bias in enumerate(mlp.intercepts_):
            intercepts_grp.create_dataset(f'b_{i}', data=bias)
            
    print(f"\n[Model Exporter] Successfully saved trained neural network model to: {filepath}")

def train():
    if not os.path.exists(LOG_PATH):
        print(f"Error: Dataset CSV log not found at {LOG_PATH}")
        print("Please run `python generate_dataset.py` first.")
        return
        
    print(f"Loading dataset telemetry from {LOG_PATH}...")
    # Read the dataset and handle both cases (header and no header)
    cols = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    df = pd.read_csv(LOG_PATH, names=cols)
    if df.iloc[0]['steering'] == 'steering':
        df = df.iloc[1:]
    print(f"Total dataset samples: {len(df)}")
    
    X = []
    y = []
    
    correction = 0.25 # Steering adjustment to recover back to center
    for idx, row in df.iterrows():
        steering_center = float(row['steering'])
        
        cameras = ['center', 'left', 'right']
        steering_angles = [
            steering_center, 
            steering_center + correction, 
            steering_center - correction
        ]
        
        for i in range(3):
            img_path = str(row[cameras[i]]).strip()
            if not os.path.isabs(img_path):
                img_path = os.path.join(DATA_DIR, img_path)
                
            img = cv2.imread(img_path)
            angle = steering_angles[i]
            
            if img is not None:
                features = preprocess_image(img)
                X.append(features)
                y.append(angle)
                
                # Data Augmentation: Horizontal Flip with inverted steering
                img_flipped = cv2.flip(img, 1)
                X.append(preprocess_image(img_flipped))
                y.append(-angle)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Augmented dataset size: {len(X)} samples. Feature vector size: {X.shape[1]}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Deep Neural Network Steering Angle Predictor (MLP Architecture: 128 -> 64 -> 16)...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 16),
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        max_iter=100,
        random_state=42,
        verbose=True
    )
    
    mlp.fit(X_train, y_train)
    
    val_preds = mlp.predict(X_val)
    val_mse = mean_squared_error(y_val, val_preds)
    print(f"\nModel Validation Loss (MSE): {val_mse:.5f}")
    
    save_mlp_to_h5(mlp, MODEL_PATH)

if __name__ == "__main__":
    train()
