import os
import csv
import math
import numpy as np
import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMG_DIR = os.path.join(DATA_DIR, "IMG")
LOG_PATH = os.path.join(DATA_DIR, "driving_log.csv")

def ensure_directories():
    os.makedirs(IMG_DIR, exist_ok=True)

def draw_track_frame(road_offset, curve_factor, width=320, height=160):
    """
    Generates a synthetic camera view looking down a track.
    - road_offset: lateral displacement of car from center of lane (-1.0 left, +1.0 right)
    - curve_factor: road curvature (-1.0 sharp left, 0.0 straight, +1.0 sharp right)
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 60  # Dark grass/background
    
    # Sky region
    cv2.rectangle(img, (0, 0), (width, height // 2), (180, 130, 90), -1) # Sky color (BGR)
    
    # Road horizon line
    horizon_y = height // 2
    
    # Road perspective points
    center_bottom = int(width // 2 + road_offset * 120)
    road_width_bottom = 220
    road_width_top = 24
    
    center_top = int(width // 2 + curve_factor * 90)
    
    left_bottom = (center_bottom - road_width_bottom // 2, height)
    right_bottom = (center_bottom + road_width_bottom // 2, height)
    left_top = (center_top - road_width_top // 2, horizon_y)
    right_top = (center_top + road_width_top // 2, horizon_y)
    
    # Draw Road surface (Asphalt gray)
    road_poly = np.array([left_bottom, left_top, right_top, right_bottom], dtype=np.int32)
    cv2.fillPoly(img, [road_poly], (80, 80, 80))
    
    # Draw Yellow Center Lane Line
    num_segments = 8
    for i in range(num_segments):
        t1 = i / num_segments
        t2 = (i + 0.5) / num_segments
        
        y1 = int(horizon_y + t1 * (height - horizon_y))
        y2 = int(horizon_y + t2 * (height - horizon_y))
        
        x1 = int((1 - t1) * center_top + t1 * center_bottom)
        x2 = int((1 - t2) * center_top + t2 * center_bottom)
        
        cv2.line(img, (x1, y1), (x2, y2), (0, 215, 255), max(1, int(t1 * 4))) # Yellow line
        
    # Draw White Side Boundary Lines
    cv2.line(img, left_bottom, left_top, (240, 240, 240), 3)
    cv2.line(img, right_bottom, right_top, (240, 240, 240), 3)
    
    # Add subtle random texture noise
    noise = np.random.randint(-5, 5, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def generate_synthetic_dataset(num_samples=1200):
    ensure_directories()
    print(f"Generating {num_samples} synthetic driving dataset samples...")
    
    records = []
    
    for i in range(num_samples):
        # Vary road curvature and car lateral offset
        curve = math.sin(i * 0.05) * 0.8 + np.random.uniform(-0.1, 0.1)
        offset = np.random.uniform(-0.3, 0.3)
        
        # Steering angle needed to compensate for curve and offset
        steering = float(curve * 0.75 - offset * 0.4)
        steering = max(-1.0, min(1.0, steering))
        
        frame = draw_track_frame(road_offset=offset, curve_factor=curve)
        
        img_name = f"center_{i:05d}.png"
        img_path = os.path.join(IMG_DIR, img_name)
        cv2.imwrite(img_path, frame)
        
        rel_img_path = os.path.join("IMG", img_name).replace("\\", "/")
        speed = float(np.random.uniform(15.0, 30.0))
        throttle = float(max(0.2, 1.0 - abs(steering) * 0.8))
        brake = 0.0
        
        # Standard Udacity style CSV record format
        records.append([rel_img_path, rel_img_path, rel_img_path, steering, throttle, brake, speed])
        
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["center", "left", "right", "steering", "throttle", "brake", "speed"])
        writer.writerows(records)
        
    print(f"Successfully created dataset at {LOG_PATH} with {len(records)} frames.")

if __name__ == "__main__":
    generate_synthetic_dataset()
