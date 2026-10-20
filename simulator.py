import os
import sys
import math
import time
import base64
import csv
import threading
from io import BytesIO
import numpy as np
import cv2
from PIL import Image

import pygame

# Try importing socketio client
try:
    import socketio
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMG_DIR = os.path.join(DATA_DIR, "IMG")
LOG_PATH = os.path.join(DATA_DIR, "driving_log.csv")

# Canvas Constants
WIN_W, WIN_H = 960, 680
CAM_W, CAM_H = 320, 160

class Car:
    def __init__(self, x, y, angle=0.0):
        self.x = x
        self.y = y
        self.angle = angle  # Radians
        self.speed = 0.0
        self.max_speed = 35.0
        self.steering_angle = 0.0
        self.throttle = 0.0
        self.length = 40
        self.width = 20

    def update(self, dt=0.05):
        # Physics update
        self.speed += self.throttle * 12.0 * dt
        self.speed *= (1.0 - 0.5 * dt)  # Friction
        self.speed = max(-5.0, min(self.max_speed, self.speed))
        
        # Angular rotation based on speed and steering
        turn_rate = (self.speed / 25.0) * self.steering_angle * 1.8
        self.angle += turn_rate * dt
        
        # Movement vector
        self.x += math.cos(self.angle) * self.speed * dt * 10.0
        self.y += math.sin(self.angle) * self.speed * dt * 10.0

class Track:
    def __init__(self):
        # Oval track with S-curve waypoints
        self.points = [
            (200, 180), (480, 140), (760, 200), (820, 360),
            (700, 520), (450, 540), (220, 480), (140, 320)
        ]
        self.track_width = 90

    def draw(self, surface):
        # Fill background grass
        surface.fill((45, 115, 55))
        
        # Draw track asphalt
        if len(self.points) >= 3:
            pygame.draw.polygon(surface, (70, 70, 70), self.points)
            
            # Thick outlines
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i + 1) % len(self.points)]
                pygame.draw.line(surface, (70, 70, 70), p1, p2, self.track_width)
                
            # Draw outer and inner white borders
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i + 1) % len(self.points)]
                pygame.draw.line(surface, (230, 230, 230), p1, p2, self.track_width + 6)
                
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i + 1) % len(self.points)]
                pygame.draw.line(surface, (70, 70, 70), p1, p2, self.track_width)
                
            # Draw yellow center dashed line
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i + 1) % len(self.points)]
                pygame.draw.line(surface, (240, 200, 0), p1, p2, 4)

    def get_track_center(self, x, y):
        """Find closest waypoint and track tangent angle"""
        min_dist = float('inf')
        closest_idx = 0
        for i, (px, py) in enumerate(self.points):
            dist = math.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        p1 = self.points[closest_idx]
        p2 = self.points[(closest_idx + 1) % len(self.points)]
        tangent = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        return min_dist, tangent

class SimulatorApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Self-Driving Car Simulator & Socket Controller")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18, bold=True)
        self.title_font = pygame.font.SysFont("Arial", 22, bold=True)
        
        self.track = Track()
        self.car = Car(200, 180, angle=0.0)
        
        self.mode = "AUTONOMOUS"  # MANUAL or AUTONOMOUS
        self.is_recording = False
        self.log_file = None
        self.csv_writer = None
        self.frame_count = 0
        
        # Socket client attributes
        self.sio = None
        self.connected_to_server = False
        self.auto_steering = 0.0
        self.auto_throttle = 0.0
        self.auto_brake = False
        
        if HAS_SOCKETIO:
            self.setup_socket_client()
            
    def setup_socket_client(self):
        self.sio = socketio.Client()
        
        @self.sio.on('connect')
        def on_connect():
            self.connected_to_server = True
            print("[Simulator Client] Connected to drive.py socket server!")
            
        @self.sio.on('disconnect')
        def on_disconnect():
            self.connected_to_server = False
            print("[Simulator Client] Disconnected from drive.py server.")
            
        @self.sio.on('steer')
        def on_steer(data):
            try:
                self.auto_steering = float(data.get('steering_angle', 0.0))
                self.auto_throttle = float(data.get('throttle', 0.0))
                # Support explicit braking parameter
                brake_val = data.get('brake', 'False')
                self.auto_brake = (brake_val == 'True' or brake_val is True)
            except Exception as e:
                print(f"Error parsing steer payload: {e}")
                
        def connect_thread():
            try:
                print("[Simulator Client] Attempting socket connection to http://localhost:4567...")
                self.sio.connect('http://localhost:4567', wait_timeout=5)
            except Exception as err:
                print(f"[Simulator Client] Socket connection notice: drive.py server not found yet ({err}).")
                
        threading.Thread(target=connect_thread, daemon=True).start()

    def generate_camera_frame(self):
        """
        Synthesizes a 320x160 front camera view based on car heading & track curvature
        """
        dist, track_tangent = self.track.get_track_center(self.car.x, self.car.y)
        angle_diff = math.atan2(math.sin(self.car.angle - track_tangent), math.cos(self.car.angle - track_tangent))
        
        # Road curvature in camera perspective
        curve = -angle_diff * 0.8
        road_offset = (dist / 40.0) * (-1.0 if dist > 0 else 1.0)
        road_offset = max(-1.0, min(1.0, road_offset))
        
        # Draw frame using same logic as synthetic generator
        img = np.ones((CAM_H, CAM_W, 3), dtype=np.uint8) * 60
        horizon_y = CAM_H // 2
        cv2.rectangle(img, (0, 0), (CAM_W, horizon_y), (180, 130, 90), -1)
        
        center_bottom = int(CAM_W // 2 + road_offset * 100)
        center_top = int(CAM_W // 2 + curve * 80)
        
        left_b = (center_bottom - 110, CAM_H)
        right_b = (center_bottom + 110, CAM_H)
        left_t = (center_top - 15, horizon_y)
        right_t = (center_top + 15, horizon_y)
        
        road_poly = np.array([left_b, left_t, right_t, right_b], dtype=np.int32)
        cv2.fillPoly(img, [road_poly], (80, 80, 80))
        cv2.line(img, (center_bottom, CAM_H), (center_top, horizon_y), (0, 215, 255), 3)
        cv2.line(img, left_b, left_t, (240, 240, 240), 2)
        cv2.line(img, right_b, right_t, (240, 240, 240), 2)
        
        return img

    def encode_frame_base64(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def send_telemetry(self, frame_bgr):
        if self.connected_to_server and self.sio:
            b64_image = self.encode_frame_base64(frame_bgr)
            telemetry_data = {
                'image': b64_image,
                'speed': str(self.car.speed),
                'steering_angle': str(self.car.steering_angle),
                'throttle': str(self.car.throttle)
            }
            try:
                self.sio.emit('telemetry', telemetry_data)
            except Exception as e:
                pass

    def record_manual_frame(self, frame_bgr):
        os.makedirs(IMG_DIR, exist_ok=True)
        self.frame_count += 1
        img_name = f"manual_{self.frame_count:05d}.png"
        img_path = os.path.join(IMG_DIR, img_name)
        cv2.imwrite(img_path, frame_bgr)
        
        rel_path = os.path.join("IMG", img_name).replace("\\", "/")
        
        file_exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["center", "left", "right", "steering", "throttle", "brake", "speed"])
            writer.writerow([rel_path, rel_path, rel_path, self.car.steering_angle, self.car.throttle, 0.0, self.car.speed])

    def draw_car(self):
        # Draw car rectangle rotated according to angle
        car_surface = pygame.Surface((self.car.length, self.car.width), pygame.SRCALPHA)
        car_surface.fill((220, 40, 40))  # Red car body
        pygame.draw.rect(car_surface, (30, 30, 30), (28, 2, 10, 16)) # Windshield
        
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.car.angle))
        rect = rotated_surface.get_rect(center=(int(self.car.x), int(self.car.y)))
        self.screen.blit(rotated_surface, rect)

    def draw_hud(self, cam_frame):
        # Draw Camera Preview Box in top right
        cam_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        cam_surf = pygame.surfarray.make_surface(cam_rgb.swapaxes(0, 1))
        self.screen.blit(cam_surf, (WIN_W - CAM_W - 20, 20))
        pygame.draw.rect(self.screen, (255, 255, 255), (WIN_W - CAM_W - 20, 20, CAM_W, CAM_H), 2)
        
        # Panel Overlay (Top Left)
        hud_surface = pygame.Surface((340, 220), pygame.SRCALPHA)
        hud_surface.fill((20, 20, 20, 210))
        self.screen.blit(hud_surface, (20, 20))
        
        t_title = self.title_font.render("SELF-DRIVING CAR SIMULATOR", True, (255, 215, 0))
        self.screen.blit(t_title, (30, 28))
        
        mode_color = (50, 205, 50) if self.mode == "AUTONOMOUS" else (255, 140, 0)
        t_mode = self.font.render(f"MODE: {self.mode} (Press 'M' to Toggle)", True, mode_color)
        self.screen.blit(t_mode, (30, 60))
        
        status_str = "CONNECTED" if self.connected_to_server else "SEARCHING (Port 4567)..."
        status_col = (50, 205, 50) if self.connected_to_server else (255, 60, 60)
        t_socket = self.font.render(f"Socket Server: {status_str}", True, status_col)
        self.screen.blit(t_socket, (30, 90))
        
        t_speed = self.font.render(f"Speed: {self.car.speed:.1f} mph", True, (240, 240, 240))
        self.screen.blit(t_speed, (30, 120))
        
        t_steer = self.font.render(f"Steering Angle: {self.car.steering_angle:+.4f}", True, (240, 240, 240))
        self.screen.blit(t_steer, (30, 150))
        
        rec_str = "ON" if self.is_recording else "OFF (Press 'R' to Toggle)"
        rec_col = (255, 50, 50) if self.is_recording else (180, 180, 180)
        t_rec = self.font.render(f"Recording Log: {rec_str}", True, rec_col)
        self.screen.blit(t_rec, (30, 180))

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(30) / 1000.0  # 30 FPS
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self.mode = "MANUAL" if self.mode == "AUTONOMOUS" else "AUTONOMOUS"
                        print(f"[Simulator] Switched to {self.mode} Mode.")
                    elif event.key == pygame.K_r:
                        self.is_recording = not self.is_recording
                        print(f"[Simulator] Recording log set to: {self.is_recording}")
                        
            # Camera View Frame Generation
            cam_frame = self.generate_camera_frame()
            
            # Handle Driving Input based on Mode
            if self.mode == "MANUAL":
                keys = pygame.key.get_pressed()
                steer_input = 0.0
                if keys[pygame.K_LEFT]:
                    steer_input -= 1.0
                if keys[pygame.K_RIGHT]:
                    steer_input += 1.0
                    
                throttle_input = 0.0
                if keys[pygame.K_UP]:
                    throttle_input = 1.0
                elif keys[pygame.K_DOWN]:
                    throttle_input = -0.5
                    
                self.car.steering_angle = steer_input
                self.car.throttle = throttle_input
                
                if self.is_recording:
                    self.record_manual_frame(cam_frame)
                    
            elif self.mode == "AUTONOMOUS":
                # Send frame over socket and apply autonomous steer response
                self.send_telemetry(cam_frame)
                self.car.steering_angle = self.auto_steering
                self.car.throttle = self.auto_throttle
                if self.auto_brake:
                    self.car.speed = 0.0
                    self.car.throttle = 0.0
                
            # Physics Update
            self.car.update(dt)
            
            # Rendering
            self.track.draw(self.screen)
            self.draw_car()
            self.draw_hud(cam_frame)
            
            pygame.display.flip()
            
        if self.sio and self.connected_to_server:
            self.sio.disconnect()
        pygame.quit()

if __name__ == "__main__":
    app = SimulatorApp()
    app.run()
