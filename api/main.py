import os
import sys
from datetime import datetime
from typing import List, Optional
import asyncio

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import SessionLocal, init_db
from database.models import Entry, PersonFeature, VehicleFeature, Alert

# Create the app
app = FastAPI(
    title="Gate Monitor API",
    description="Backend service for AI CCTV Gate Monitoring System"
)

# Initialize Database
init_db()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Schemas
class EntrySchema(BaseModel):
    id: int
    visitor_id: str
    track_id: int
    object_type: str
    camera_id: str
    zone_id: str
    entry_time: datetime
    exit_time: Optional[datetime]
    dwell_seconds: float
    visit_count: int
    risk_score: float
    is_anomaly: bool
    alert_triggered: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class AlertSchema(BaseModel):
    id: int
    visitor_id: str
    alert_type: str
    alert_message: str
    risk_score: float
    camera_id: str
    triggered_at: datetime
    acknowledged: bool

    model_config = ConfigDict(from_attributes=True)

class VehicleFeatureSchema(BaseModel):
    id: int
    visitor_id: str
    frame_number: int
    timestamp: datetime
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    center_x: float
    center_y: float
    speed_pixels_per_frame: float
    plate_number: Optional[str]
    plate_confidence: float
    vehicle_color: Optional[str]
    vehicle_type: Optional[str]
    is_inside_zone: bool

    model_config = ConfigDict(from_attributes=True)

class StatsSchema(BaseModel):
    active_tracks: int
    total_entries: int
    alerts_today: int

class HealthSchema(BaseModel):
    status: str

# Endpoints
@app.get("/health", response_model=HealthSchema)
def health_check():
    return {"status": "ok"}

@app.get("/entries", response_model=List[EntrySchema])
def get_entries(limit: int = 100, db: Session = Depends(get_db)):
    entries = db.query(Entry).order_by(Entry.created_at.desc()).limit(limit).all()
    return entries

@app.get("/entries/active", response_model=List[EntrySchema])
def get_active_entries(db: Session = Depends(get_db)):
    entries = db.query(Entry).filter(Entry.exit_time == None).order_by(Entry.created_at.desc()).all()
    return entries

@app.get("/alerts", response_model=List[AlertSchema])
def get_alerts(limit: int = 50, db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(Alert.triggered_at.desc()).limit(limit).all()
    return alerts

@app.post("/alerts/{alert_id}/acknowledge", response_model=AlertSchema)
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.acknowledged = True
    db.commit()
    db.refresh(alert)
    return alert

@app.get("/vehicles/{plate}", response_model=List[VehicleFeatureSchema])
def get_vehicle_by_plate(plate: str, db: Session = Depends(get_db)):
    vehicles = db.query(VehicleFeature).filter(VehicleFeature.plate_number.ilike(f"%{plate}%")).order_by(VehicleFeature.timestamp.desc()).all()
    return vehicles

@app.get("/stats", response_model=StatsSchema)
def get_stats(db: Session = Depends(get_db)):
    active_tracks = db.query(Entry).filter(Entry.exit_time == None).count()
    total_entries = db.query(Entry).count()
    
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    alerts_today = db.query(Alert).filter(Alert.triggered_at >= today).count()
    
    return {
        "active_tracks": active_tracks,
        "total_entries": total_entries,
        "alerts_today": alerts_today
    }

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await manager.connect(websocket)
    try:
        while True:
            # Poll DB for updates and broadcast to client
            # In a full-scale system, the pipeline would push to this WS or Redis PubSub directly
            active_tracks = db.query(Entry).filter(Entry.exit_time == None).count()
            recent_alerts = db.query(Alert).order_by(Alert.triggered_at.desc()).limit(5).all()
            
            alerts_data = [
                {
                    "id": a.id,
                    "type": a.alert_type,
                    "message": a.alert_message,
                    "risk": a.risk_score
                }
                for a in recent_alerts
            ]

            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "active_tracks": active_tracks,
                "recent_alerts": alerts_data
            }
            
            await manager.broadcast(payload)
            await asyncio.sleep(2.0)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
