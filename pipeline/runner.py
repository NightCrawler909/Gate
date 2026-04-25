import os
import sys
import time
import logging
from typing import Optional, Dict

import cv2
import numpy as np
from sqlalchemy.orm import Session

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from database.db import SessionLocal, init_db
from database import crud

from core.detector import GateDetector
from core.tracker import GateTracker
from core.zone_manager import ZoneManager
from core.feature_extractor import FeatureExtractor
from core.anpr import ANPRSystem
from core.color_classifier import VehicleColorClassifier

from ml.risk_scorer import RiskScorer
from ml.anomaly_detector import AnomalyDetector


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Runner] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Central orchestrator for the AI CCTV Gate Monitoring System.
    Links Detection -> Tracking -> Zone -> Features -> ANPR/Color -> ML -> DB.
    """

    def __init__(self):
        logger.info("Initializing PipelineRunner modules...")
        
        # 1. Initialize DB
        init_db()
        self.db: Session = SessionLocal()

        # 2. Initialize Core CV Modules
        self.detector = GateDetector(model_size=settings.YOLO_MODEL_SIZE)
        self.tracker = GateTracker()
        self.zone_manager = ZoneManager()
        self.feature_extractor = FeatureExtractor(
            zone_center=self.zone_manager.get_zone_center()
        )
        self.anpr = ANPRSystem()
        self.color_classifier = VehicleColorClassifier()

        # 3. Initialize ML Modules
        # Ensure we don't crash if models aren't trained yet by setting auto_train=True
        self.risk_scorer = RiskScorer(auto_train=True)
        self.anomaly_detector = AnomalyDetector(auto_train=True)

        # State tracking
        self.frame_number = 0
        self.active_alerts = set() # Store visitor_ids currently alerting
        
        logger.info("PipelineRunner initialization complete.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Executes the full AI pipeline on a single frame.
        
        Returns the annotated frame.
        """
        self.frame_number += 1
        annotated_frame = frame.copy()

        # 1. Detection
        detections = self.detector.detect(frame, self.frame_number)

        # 2. Tracking
        tracked_objects = self.tracker.update(detections, self.frame_number)

        # 3. Zone Management
        for obj in tracked_objects:
            event = self.zone_manager.check_zone_event(
                track_id=obj.track_id,
                center_x=obj.center[0],
                center_y=obj.center[1],
                frame_number=self.frame_number
            )
            
            if event:
                if event.event_type == "ENTRY":
                    # Create DB Entry
                    db_entry = crud.create_entry(self.db, track_id=obj.track_id, object_type=obj.object_type)
                    obj.visitor_id = db_entry.visitor_id
                    obj.is_inside_zone = True
                    obj.visit_count = 1 # Would query DB for past visits in real system
                elif event.event_type == "EXIT" and obj.visitor_id:
                    # Update DB Entry
                    obj.is_inside_zone = False
                    from datetime import datetime
                    crud.update_entry_exit(
                        self.db, 
                        visitor_id=obj.visitor_id, 
                        exit_time=datetime.now(),
                        dwell_seconds=obj.get_dwell_seconds()
                    )

        # 4. Feature Extraction
        features_list = self.feature_extractor.extract_all(tracked_objects, self.frame_number)

        # 5. ANPR & Color Classification (Vehicles only)
        # We only run these periodically or if we don't have confident readings yet
        # to save GPU cycles. For simplicity in the runner, we batch process them.
        anpr_results = {}
        color_results = {}
        
        vehicles = [obj for obj in tracked_objects if obj.object_type == "vehicle"]
        if vehicles:
            plates = self.anpr.process_batch(frame, vehicles)
            for p in plates:
                anpr_results[p.track_id] = p
                # Update tracker object with plate info
                obj = self.tracker.get_track(p.track_id)
                if obj and not hasattr(obj, 'plate_text'):
                    obj.plate_text = p.plate_text
            
            colors = self.color_classifier.process_batch(frame, vehicles)
            for c in colors:
                color_results[c.track_id] = c
                obj = self.tracker.get_track(c.track_id)
                if obj and not hasattr(obj, 'vehicle_color'):
                    obj.vehicle_color = c.color

        # 6. ML Scoring (Risk & Anomaly)
        # Convert FrameFeatures to dicts for ML models
        ml_inputs = []
        for ff in features_list:
            ml_in = self.risk_scorer.get_feature_vector(ff)
            ml_in["track_id"] = ff.track_id
            ml_inputs.append(ml_in)

        risk_results = []
        anomaly_results = []
        if ml_inputs:
            risk_results = self.risk_scorer.batch_score(ml_inputs)
            anomaly_results = self.anomaly_detector.batch_detect(ml_inputs)

        # Combine ML results into a lookup dictionary
        ml_data_by_track = {}
        for r in risk_results:
            ml_data_by_track.setdefault(r.track_id, {})["risk"] = r
        for a in anomaly_results:
            ml_data_by_track.setdefault(a.track_id, {})["anomaly"] = a

        # 7. Database Updates & Alert Logic
        self.handle_database_updates(features_list, anpr_results, color_results, ml_data_by_track)

        # 8. Draw Overlays
        annotated_frame = self.draw_overlay(
            annotated_frame, 
            tracked_objects, 
            features_list, 
            ml_data_by_track, 
            anpr_results, 
            color_results
        )

        return annotated_frame

    def handle_database_updates(
        self, 
        features_list: list, 
        anpr_results: dict, 
        color_results: dict, 
        ml_data: dict
    ):
        """Handle persistence and alert generation."""
        # Only save features to DB every N frames to avoid overwhelming it
        if self.frame_number % settings.FRAME_SKIP != 0:
            return

        for ff in features_list:
            if not ff.visitor_id or ff.visitor_id == "UNKNOWN":
                continue

            track_id = ff.track_id
            
            # Extract ML stats
            risk_res = ml_data.get(track_id, {}).get("risk")
            anomaly_res = ml_data.get(track_id, {}).get("anomaly")
            
            risk_score = risk_res.risk_score if risk_res else 0.0
            is_anomaly = anomaly_res.is_anomaly if anomaly_res else False
            is_high_risk = risk_res.is_high_risk if risk_res else False

            # Update Entry risk
            crud.update_entry_risk(self.db, ff.visitor_id, risk_score, is_anomaly)

            # Check Alerts
            needs_alert = False
            alert_msg = ""
            
            if is_high_risk:
                needs_alert = True
                alert_msg = f"High risk score detected: {risk_score}"
            elif is_anomaly:
                needs_alert = True
                alert_msg = f"Anomalous behavior detected"
            elif ff.is_loitering:
                needs_alert = True
                alert_msg = f"Loitering detected ({ff.dwell_seconds:.1f}s)"

            if needs_alert and ff.visitor_id not in self.active_alerts:
                crud.create_alert(
                    self.db, 
                    visitor_id=ff.visitor_id, 
                    alert_type="BEHAVIOR", 
                    alert_message=alert_msg,
                    risk_score=risk_score
                )
                self.active_alerts.add(ff.visitor_id)
                logger.warning(f"ALERT Triggered for Track {track_id}: {alert_msg}")

            # Save specific feature tables
            if ff.object_type == "person":
                crud.save_person_feature(
                    db=self.db,
                    visitor_id=ff.visitor_id,
                    frame_number=self.frame_number,
                    bbox=(ff.bbox_x, ff.bbox_y, ff.bbox_w, ff.bbox_h),
                    center=(ff.center_x, ff.center_y),
                    speed_ppf=ff.speed_pixels_per_frame,
                    speed_mps=ff.speed_mps,
                    direction_angle=ff.direction_angle,
                    is_inside_zone=ff.is_inside_zone,
                    trajectory_json=ff.trajectory_json
                )
            elif ff.object_type == "vehicle":
                plate_data = anpr_results.get(track_id)
                color_data = color_results.get(track_id)
                
                crud.save_vehicle_feature(
                    db=self.db,
                    visitor_id=ff.visitor_id,
                    frame_number=self.frame_number,
                    bbox=(ff.bbox_x, ff.bbox_y, ff.bbox_w, ff.bbox_h),
                    center=(ff.center_x, ff.center_y),
                    speed_ppf=ff.speed_pixels_per_frame,
                    plate_number=plate_data.plate_text if plate_data else None,
                    plate_confidence=plate_data.confidence if plate_data else 0.0,
                    vehicle_color=color_data.color if color_data else None,
                    is_inside_zone=ff.is_inside_zone
                )

    def draw_overlay(
        self, 
        frame: np.ndarray, 
        tracked_objects: list, 
        features_list: list, 
        ml_data: dict,
        anpr_results: dict,
        color_results: dict
    ) -> np.ndarray:
        """Draw bounding boxes, zones, ML scores, and ANPR results."""
        
        # Draw base tracking boxes and trails
        frame = self.tracker.draw_tracks(frame)
        
        # Draw zone polygon
        frame = self.zone_manager.draw_zone(frame)

        # Create a lookup for features
        feat_lookup = {f.track_id: f for f in features_list}

        for obj in tracked_objects:
            tid = obj.track_id
            x1 = int(obj.bbox[0])
            y1 = int(obj.bbox[1])
            y2 = int(obj.bbox[3])
            
            ff = feat_lookup.get(tid)
            risk_res = ml_data.get(tid, {}).get("risk")
            anomaly_res = ml_data.get(tid, {}).get("anomaly")

            # Basic string parts
            labels = []
            
            if ff:
                labels.append(f"{ff.speed_mps:.1f}m/s | D:{ff.dwell_seconds:.0f}s")
            
            if risk_res:
                r_str = f"Risk:{risk_res.risk_score}"
                if risk_res.is_high_risk:
                    r_str += " [!]"
                labels.append(r_str)
                
            if anomaly_res and anomaly_res.is_anomaly:
                labels.append("ANOMALY")

            # Draw feature/ML labels below the bounding box
            for i, text in enumerate(labels):
                ty = y2 + 20 + (i * 20)
                # Warning color if high risk or anomaly, else cyan
                color = (0, 0, 255) if ("[!" in text or "ANOMALY" in text) else (255, 255, 0)
                
                # Shadow
                cv2.putText(frame, text, (x1 + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # Text
                cv2.putText(frame, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw ANPR / Color for vehicles above the box
            if obj.object_type == "vehicle":
                p_data = anpr_results.get(tid)
                c_data = color_results.get(tid)
                
                v_labels = []
                if p_data:
                    v_labels.append(f"[{p_data.plate_text}]")
                if c_data:
                    v_labels.append(f"{c_data.color}")
                
                if v_labels:
                    v_text = " ".join(v_labels)
                    ty = y1 - 10
                    cv2.putText(frame, v_text, (x1 + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(frame, v_text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def process_video(self, video_path: str, show: bool = True):
        """Processes a video file completely."""
        logger.info(f"Starting processing for video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open {video_path}")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames to process: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            try:
                out_frame = self.process_frame(frame)
                
                if self.frame_number % 100 == 0:
                    logger.info(f"Processed frame {self.frame_number}/{total_frames} | "
                              f"Active Tracks: {len(self.tracker.active_tracks)} | "
                              f"Alerts: {len(self.active_alerts)}")

                if show:
                    # Resize for display if it's 4K
                    display = cv2.resize(out_frame, (1280, 720)) if out_frame.shape[1] > 1920 else out_frame
                    cv2.imshow("Gate Monitor AI", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User interrupted video processing.")
                        break
            except Exception as e:
                logger.error(f"Error processing frame {self.frame_number}: {e}", exc_info=True)

        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Finished processing {video_path}")

    def cleanup(self):
        """Release DB and CV resources."""
        self.detector.release()
        self.db.close()
        cv2.destroyAllWindows()
        logger.info("PipelineRunner cleanup complete.")


if __name__ == "__main__":
    runner = None
    try:
        videos = settings.get_video_files()
        if not videos:
            logger.error("No videos found to process. Check CAMERA_SOURCE in .env")
            sys.exit(1)
            
        test_video = videos[0]
        runner = PipelineRunner()
        runner.process_video(test_video, show=True)
    except Exception as exc:
        logger.error(f"Fatal error in runner test: {exc}")
    finally:
        if runner:
            runner.cleanup()
