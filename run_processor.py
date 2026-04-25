import cv2
import numpy as np
import json
import os
import torch
import time
import pandas as pd
import math
from collections import deque

# Import custom classes
from core.detector import PersonDetector
from core.zone_manager import ZoneManager
from core.feature_extractor import FeatureExtractor
from ml.anomaly_detector import AnomalyDetector
from ml.risk_scorer import RiskScorer

def main():
    print("[INFO] Starting Intrusion Detection System...")

    # Ensure the results/ directory exists
    os.makedirs('results', exist_ok=True)

    # Determine PyTorch device: RTX 4050 GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using inference device: {device}")

    # Initialize Core & ML Components
    # Assuming the classes accept device/model_path parameters as standard
    detector = PersonDetector()
    zone_manager = ZoneManager(config_path='data/zone_config.json')
    feature_extractor = FeatureExtractor()
    
    anomaly_detector = AnomalyDetector()
    try:
        anomaly_detector.load_model() # Load the trained Random Forest model
    except AttributeError:
        pass # In case AnomalyDetector loads the model inside its __init__ automatically

    risk_scorer = RiskScorer()

    # Video I/O Setup
    input_video_path = 'sample.mp4'
    output_video_path = os.path.join('results', 'output_tracked.mp4')
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {input_video_path}")
        return

    # Extract video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0 # fallback
        
    total_expected_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Set up cv2.VideoWriter with MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Metric tracking
    frames_processed = 0
    start_time = time.time()

    print("[INFO] Beginning main processing loop...")
    
    # Storage for tracking history and reporting
    telemetry_data = []
    track_history = {}  # Format: track_id -> deque of dicts {'frame':, 'position':, 'distance':}

    # 3. Main Processing Loop
    while True:
        ret, frame = cap.read()
        
        # Adding HEVC fallback bypass (skip frame if broken)
        if not ret:
            print("[WARNING] OpenCV failed to read frame. Video stream may be corrupted or EOF reached. Skipping.")
            # We break instead of skip at EOF, but for corruption let's just abort
            break

        # A. Vision & Tracking
        # Run the YOLO Detector on the frame, strictly filtering for class=0 (persons)
        results = detector.track(frame, persist=True)

        # Draw Base Overlay
        annotated_frame = frame.copy()
        
        # 1. Restricted Zone: Semi-transparent red area and thick outline
        overlay = annotated_frame.copy()
        # Ensure polygon coordinates are formatted appropriately for cv2
        polygon_pts = np.array(zone_manager.polygon, np.int32).reshape((-1, 1, 2))
        
        cv2.fillPoly(overlay, [polygon_pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)
        cv2.polylines(annotated_frame, [polygon_pts], isClosed=True, color=(0, 0, 255), thickness=3)

        # B. Feature Extraction & Telemetry
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                # Using tracker.py format
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate the bottom-center point of their bounding box
                bottom_center_x = (x1 + x2) // 2
                bottom_center_y = y2
                bottom_center = (bottom_center_x, bottom_center_y)

                # Calculate distance_to_restricted_zone
                is_inside = zone_manager.is_inside(bottom_center)
                if is_inside:
                    distance_to_zone = 0.0
                else:
                    distance_to_zone = zone_manager.distance_to_polygon(bottom_center)

                # Update trajectory history
                if track_id not in track_history:
                    track_history[track_id] = deque(maxlen=15)
                
                track_history[track_id].append({
                    'frame': frames_processed,
                    'position': bottom_center,
                    'distance': distance_to_zone
                })

                history = track_history[track_id]
                
                # 1. Implement Trajectory & Vector Math
                speed_px_per_sec = 0.0
                zone_approach_delta = 0.0
                action_vector = "Neutral"
                
                if len(history) > 1:
                    past_record = history[0] # Earliest available up to 15 frames ago
                    current_record = history[-1]
                    
                    frames_passed = current_record['frame'] - past_record['frame']
                    if frames_passed > 0:
                        # Velocity Vector Math
                        dx = current_record['position'][0] - past_record['position'][0]
                        dy = current_record['position'][1] - past_record['position'][1]
                        dist_traveled = math.hypot(dx, dy)
                        # Speed in px per sec (approx)
                        speed_px_per_sec = (dist_traveled / frames_passed) * fps
                        
                        # Zone Approach Delta
                        zone_approach_delta = current_record['distance'] - past_record['distance']
                        
                        if zone_approach_delta < -5:
                            action_vector = "Closing In"
                        elif zone_approach_delta > 5:
                            action_vector = "Moving Away"
                        else:
                            action_vector = "Parallel"
                            
                        # Draw velocity arrow
                        end_point = (
                            current_record['position'][0] + int(dx * 2),
                            current_record['position'][1] + int(dy * 2)
                        )
                        cv2.arrowedLine(annotated_frame, current_record['position'], end_point, (255, 255, 255), 2, tipLength=0.3)

                # C. ML Inference (Intent & Risk)
                features = {
                    'speed_variance': speed_px_per_sec, 
                    'distance_to_restricted_zone': distance_to_zone,
                    'time_near_boundary': 5.0 if (distance_to_zone < 50 and not is_inside) else 0.0,
                    'time_inside_zone': 1.0 if is_inside else 0.0,
                    'sudden_stops': 1 if speed_px_per_sec < 5 else 0,
                    'carrying_baggage': 0,
                    'time_of_day_multiplier': 1.0
                }

                # Feed this dictionary to AnomalyDetector to get the intent_class string
                base_intent = anomaly_detector.predict_intent(features)
                
                # Feed the dictionary to RiskScorer to get the smoothed float risk score
                risk_score = risk_scorer.calculate_risk(track_id, features)
                risk_score = min(float(risk_score), 100.0)

                # 2. The Granular Intent Matrix
                intent_state = "Casual Transit"
                
                if is_inside:
                    intent_state = "CRITICAL INTRUSION"
                elif risk_score >= 80 or (distance_to_zone < 50 and speed_px_per_sec < 20 and action_vector == "Parallel"):
                    intent_state = "Active Scouting"
                elif speed_px_per_sec < 10 and action_vector == "Neutral" and distance_to_zone < 100:
                    intent_state = "Suspicious Loitering"
                elif action_vector == "Closing In" and distance_to_zone < 150:
                    intent_state = "Approaching Boundary"
                elif action_vector == "Moving Away" and speed_px_per_sec > 100:
                    intent_state = "Fleeing"
                elif base_intent == "passing_by":
                    intent_state = "Casual Transit"

                # 3. Log Data Per Frame to Telemetry
                timestamp_sec = frames_processed / fps
                telemetry_data.append({
                    'frame': frames_processed,
                    'timestamp_sec': round(timestamp_sec, 2),
                    'track_id': track_id,
                    'intent_state': intent_state,
                    'action_vector': action_vector,
                    'speed_px_per_sec': round(speed_px_per_sec, 2),
                    'distance_to_zone': round(distance_to_zone, 2),
                    'risk_score': round(risk_score, 2)
                })

                # D. Overlay & Annotation UI
                # 2. Bounding Boxes color-coding based on Granular Intent Matrix
                if intent_state == 'Casual Transit':
                    box_color = (0, 255, 0)       # Green
                elif intent_state == 'Approaching Boundary':
                    box_color = (0, 255, 255)     # Yellow
                elif intent_state in ['Suspicious Loitering', 'Active Scouting']:
                    box_color = (0, 165, 255)     # Orange
                elif intent_state == 'CRITICAL INTRUSION':
                    box_color = (0, 0, 255)       # Red
                elif intent_state == 'Fleeing':
                    box_color = (255, 0, 255)     # Purple
                else:
                    box_color = (255, 255, 255)   # Default White
                    
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

                # 3. Telemetry UI: Background rectangle + white text
                display_text = f"ID: {track_id} | {intent_state} | Vec: {action_vector}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                
                (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
                
                # Draw dark semi-transparent rectangle for text background
                text_overlay = annotated_frame.copy()
                cv2.rectangle(text_overlay, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 0, 0), -1)
                cv2.addWeighted(text_overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                
                # Put crisp white text over it
                cv2.putText(annotated_frame, display_text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)

        out.write(annotated_frame)
        frames_processed += 1
        
        # Optional: Print progress periodically
        if frames_processed % 30 == 0:
            print(f"Processed {frames_processed} frames...")

    # 4. Cleanup
    end_time = time.time()
    execution_time = end_time - start_time
    actual_fps = frames_processed / execution_time if execution_time > 0 else 0

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # 4. Export to Excel
    if telemetry_data:
        df = pd.DataFrame(telemetry_data)
        excel_path = os.path.join('results', 'intrusion_stats.xlsx')
        df.to_excel(excel_path, index=False)
    else:
        excel_path = "None (No tracking data generated)"

    print("\n" + "="*40)
    print("        EXECUTION SUMMARY        ")
    print("="*40)
    print(f"Total Frames Processed : {frames_processed}")
    print(f"Overall Processing FPS : {actual_fps:.2f}")
    print(f"Output Video Path      : {output_video_path}")
    print(f"Telemetry Excel Path   : {excel_path}")
    print("="*40)

if __name__ == "__main__":
    main()
