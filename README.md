# Gate: Advanced Intrusion Detection & Intent Prediction System

Gate is a high-performance, ML-powered real-time tracking and perimeter security system. It leverages **Ultralytics YOLOv8** for precise person detection and **ByteTrack** for ultra-persistent ID tracking, paired with a custom Machine Learning backend to predict behavioral intent and assign dynamic risk scores.

## Key Features

- **Ultra-Fast Edge Inference:** Optimized YOLOv8n execution on CUDA (FP16) with explicit I/O blocking to guarantee maximum FPS on hardware like the RTX 4050.
- **Persistent Tracking:** Customized ByteTrack configuration with a 120-frame memory buffer to prevent ID switching during occlusion or frame drops.
- **Behavioral Intent Matrix:** Custom Random Forest ML models evaluate speed variance, zone approach trajectory vectors, distances, and loitering times to predict granular states:
  - 🟩 `Casual Transit`
  - 🟨 `Approaching Boundary`
  - 🟧 `Active Scouting` / `Suspicious Loitering`
  - 🟥 `CRITICAL INTRUSION`
  - 🟪 `Fleeing`
- **Smart Pipeline Throttling:** Physics-based trajectory vector math (speed, direction) runs every frame natively, while heavy ML inference is intelligently throttled (3:1 cadence) using a low-latency tracking cache buffer.
- **Dynamic Zone Mapping:** Includes an interactive tool (`map_zone.py`) to easily draw and serialize restricted polygonal zones, securely mapped into native 3D NumPy arrays for zero-friction OpenCV bounding math.

## Project Structure

```text
Gate/
├── core/                  # Computer Vision & Tracking (detector.py, tracker.py, zone_manager.py)
├── ml/                    # Behavioral Anomaly Models & Risk Scorer
├── api/                   # Backend FastAPI routes and schemas
├── dashboard/             # Web-based visual dashboard for monitoring
├── data/                  # Serialized configs (zone_config.json)
├── database/              # SQLite/PostgreSQL CRUD operations & Models
├── pipeline/              # Batch video processing runners
├── results/               # Compiled video outputs and Excel telemetry logs
├── map_zone.py            # Interactive script to define the restricted polygon
└── run_processor.py       # Main orchestrator script for the optimized pipeline
```

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/NightCrawler909/Gate.git
cd Gate
```

**2. Set up Python Virtual Environment:**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
# Core requirements including OpenCV, Pandas, Scikit-Learn, Ultralytics
pip install -r requirements.txt

# Install PyTorch explicitly with CUDA 11.8 (or your specific GPU architecture)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

**Step 1: Map the Restricted Zone**  
Draw a polygon over an initial frame of your target area to establish the security perimeter. Press `q` to save and quit.
```bash
python map_zone.py
```

**Step 2: Run the Processor**  
Execute the highly optimized tracking pipeline. The system will process `sample.mp4` (or live feed), generate telemetry memory caching, execute ML predictions safely, and render output overlays without bottlenecking your CPU.
```bash
python run_processor.py
```

Outputs, including the telemetry data (`intrusion_stats.xlsx`) and the fully rendered annotated video (`output_tracked.mp4`), will be saved inside the `results/` folder.

## Architecture Optimizations

This system was dramatically refactored to fix core bottlenecks associated with running deep learning models alongside UI rendering:
- Replaced deep alpha-blending logic (`cv2.addWeighted`) with raw pixel writing (`cv2.polylines`).
- Implemented `cv2` dimension rescaling explicitly prioritizing 720p to relieve CPU tensor strains.
- Created `cached_tracks` dict mapping for 0-latency dictionary hits when accessing historical trajectory intents between ML cycles.