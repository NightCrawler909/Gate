from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Entry(Base):
    __tablename__ = "entries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    visitor_id = Column(String(36), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)
    object_type = Column(String(20), nullable=False)
    camera_id = Column(String(50), default="CAM_01")
    zone_id = Column(String(50), default="GATE_ZONE_01")
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    dwell_seconds = Column(Float, default=0.0)
    visit_count = Column(Integer, default=1)
    risk_score = Column(Float, default=0.0)
    is_anomaly = Column(Boolean, default=False)
    alert_triggered = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class PersonFeature(Base):
    __tablename__ = "person_features"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    visitor_id = Column(String(36), ForeignKey("entries.visitor_id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_w = Column(Float)
    bbox_h = Column(Float)
    center_x = Column(Float)
    center_y = Column(Float)
    speed_pixels_per_frame = Column(Float, default=0.0)
    speed_mps = Column(Float, default=0.0)
    direction_angle = Column(Float, default=0.0)
    is_inside_zone = Column(Boolean, default=False)
    trajectory_json = Column(Text, nullable=True)

class VehicleFeature(Base):
    __tablename__ = "vehicle_features"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    visitor_id = Column(String(36), ForeignKey("entries.visitor_id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_w = Column(Float)
    bbox_h = Column(Float)
    center_x = Column(Float)
    center_y = Column(Float)
    speed_pixels_per_frame = Column(Float, default=0.0)
    plate_number = Column(String(20), nullable=True)
    plate_confidence = Column(Float, default=0.0)
    vehicle_color = Column(String(30), nullable=True)
    vehicle_type = Column(String(20), nullable=True)
    is_inside_zone = Column(Boolean, default=False)

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    visitor_id = Column(String(36), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    alert_message = Column(Text, nullable=False)
    risk_score = Column(Float, default=0.0)
    camera_id = Column(String(50), default="CAM_01")
    triggered_at = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False)

__all__ = ["Entry", "PersonFeature", "VehicleFeature", "Alert", "Base"]
