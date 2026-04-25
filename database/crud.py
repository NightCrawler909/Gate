import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from database.models import Entry, PersonFeature, VehicleFeature, Alert
from config.settings import settings

def create_entry(db: Session, track_id: int, object_type: str, camera_id: str = "CAM_01") -> Entry:
    """
    Generates a new UUID for visitor_id and creates a new Entry record.
    """
    visitor_id = str(uuid.uuid4())
    db_entry = Entry(
        visitor_id=visitor_id,
        track_id=track_id,
        object_type=object_type,
        camera_id=camera_id
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry

def get_entry_by_visitor_id(db: Session, visitor_id: str) -> Entry:
    """
    Returns the Entry where visitor_id matches. Returns None if not found.
    """
    return db.query(Entry).filter(Entry.visitor_id == visitor_id).first()

def get_entry_by_track_id(db: Session, track_id: int) -> Entry:
    """
    Returns the most recent Entry for this track_id.
    """
    return db.query(Entry).filter(Entry.track_id == track_id).order_by(Entry.created_at.desc()).first()

def update_entry_exit(db: Session, visitor_id: str, exit_time: datetime, dwell_seconds: float) -> Entry:
    """
    Finds entry by visitor_id, sets exit_time and dwell_seconds.
    """
    entry = get_entry_by_visitor_id(db, visitor_id)
    if entry:
        entry.exit_time = exit_time
        entry.dwell_seconds = dwell_seconds
        db.commit()
        db.refresh(entry)
    return entry

def update_entry_risk(db: Session, visitor_id: str, risk_score: float, is_anomaly: bool = False) -> Entry:
    """
    Updates risk_score and is_anomaly.
    Sets alert_triggered = True if risk_score >= settings.RISK_ALERT_THRESHOLD.
    """
    entry = get_entry_by_visitor_id(db, visitor_id)
    if entry:
        entry.risk_score = risk_score
        entry.is_anomaly = is_anomaly
        if risk_score >= settings.RISK_ALERT_THRESHOLD:
            entry.alert_triggered = True
        db.commit()
        db.refresh(entry)
    return entry

def increment_visit_count(db: Session, visitor_id: str) -> Entry:
    """
    Finds entry by visitor_id and adds 1 to visit_count.
    """
    entry = get_entry_by_visitor_id(db, visitor_id)
    if entry:
        entry.visit_count += 1
        db.commit()
        db.refresh(entry)
    return entry

def get_all_entries(db: Session, limit: int = 100, skip: int = 0) -> list:
    """
    Returns list of Entry records ordered by created_at desc.
    """
    return db.query(Entry).order_by(Entry.created_at.desc()).offset(skip).limit(limit).all()

def get_active_entries(db: Session) -> list:
    """
    Returns entries where exit_time is None (currently inside zone).
    """
    return db.query(Entry).filter(Entry.exit_time == None).all()

def save_person_feature(db: Session, visitor_id: str, frame_number: int, bbox: tuple, center: tuple,
                        speed_ppf: float, speed_mps: float, direction_angle: float, is_inside_zone: bool,
                        trajectory_json: str = None) -> PersonFeature:
    """
    Creates and commits PersonFeature record.
    """
    feature = PersonFeature(
        visitor_id=visitor_id,
        frame_number=frame_number,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_w=bbox[2],
        bbox_h=bbox[3],
        center_x=center[0],
        center_y=center[1],
        speed_pixels_per_frame=speed_ppf,
        speed_mps=speed_mps,
        direction_angle=direction_angle,
        is_inside_zone=is_inside_zone,
        trajectory_json=trajectory_json
    )
    db.add(feature)
    db.commit()
    db.refresh(feature)
    return feature

def save_vehicle_feature(db: Session, visitor_id: str, frame_number: int, bbox: tuple, center: tuple,
                         speed_ppf: float, plate_number: str = None, plate_confidence: float = 0.0,
                         vehicle_color: str = None, vehicle_type: str = None, is_inside_zone: bool = False) -> VehicleFeature:
    """
    Creates and commits VehicleFeature record.
    """
    feature = VehicleFeature(
        visitor_id=visitor_id,
        frame_number=frame_number,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_w=bbox[2],
        bbox_h=bbox[3],
        center_x=center[0],
        center_y=center[1],
        speed_pixels_per_frame=speed_ppf,
        plate_number=plate_number,
        plate_confidence=plate_confidence,
        vehicle_color=vehicle_color,
        vehicle_type=vehicle_type,
        is_inside_zone=is_inside_zone
    )
    db.add(feature)
    db.commit()
    db.refresh(feature)
    return feature

def get_vehicle_by_plate(db: Session, plate_number: str) -> list:
    """
    Returns all VehicleFeature records matching plate_number (case-insensitive).
    """
    return db.query(VehicleFeature).filter(VehicleFeature.plate_number.ilike(f"%{plate_number}%")).all()

def create_alert(db: Session, visitor_id: str, alert_type: str, alert_message: str,
                 risk_score: float = 0.0, camera_id: str = "CAM_01") -> Alert:
    """
    Creates and commits Alert record.
    """
    alert = Alert(
        visitor_id=visitor_id,
        alert_type=alert_type,
        alert_message=alert_message,
        risk_score=risk_score,
        camera_id=camera_id
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert

def get_recent_alerts(db: Session, limit: int = 50) -> list:
    """
    Returns alerts ordered by triggered_at desc.
    """
    return db.query(Alert).order_by(Alert.triggered_at.desc()).limit(limit).all()

def acknowledge_alert(db: Session, alert_id: int) -> Alert:
    """
    Sets acknowledged = True for alert with given id.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert:
        alert.acknowledged = True
        db.commit()
        db.refresh(alert)
    return alert

if __name__ == "__main__":
    from database.db import init_db, SessionLocal
    init_db()
    db = SessionLocal()
    
    test_entry = create_entry(db, track_id=999, object_type="person")
    print(f"{test_entry.visitor_id}")
    
    test_alert = create_alert(
        db,
        visitor_id=test_entry.visitor_id,
        alert_type="TEST",
        alert_message="Test alert"
    )
    print("CRUD test passed")
    db.close()
