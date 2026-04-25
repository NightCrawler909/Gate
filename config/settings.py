import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    def __init__(self):
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        
        camera_source = os.getenv("CAMERA_SOURCE")
        self.CAMERA_SOURCE = os.path.normpath(camera_source) if camera_source else None
        
        ext_str = os.getenv("VIDEO_EXTENSIONS", ".mp4,.avi,.mov,.mkv,.ts")
        self.VIDEO_EXTENSIONS = [ext.strip() for ext in ext_str.split(",") if ext.strip()]
        self.CAMERA_ID = os.getenv("CAMERA_ID", "CAM_01")
        
        self.CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
        self.ZONE_CONFIG_PATH = os.getenv("ZONE_CONFIG_PATH")
        self.RISK_ALERT_THRESHOLD = int(os.getenv("RISK_ALERT_THRESHOLD", 70))
        self.DWELL_ALERT_SECONDS = int(os.getenv("DWELL_ALERT_SECONDS", 30))
        self.SECRET_KEY = os.getenv("SECRET_KEY")

    def get_video_files(self):
        if self.CAMERA_SOURCE and os.path.isdir(self.CAMERA_SOURCE):
            files = os.listdir(self.CAMERA_SOURCE)
            video_files = []
            for f in files:
                if any(f.lower().endswith(ext.lower()) for ext in self.VIDEO_EXTENSIONS):
                    video_files.append(os.path.normpath(os.path.join(self.CAMERA_SOURCE, f)))
            video_files.sort()
            print(f"Found {len(video_files)} video files in {self.CAMERA_SOURCE}")
            if not video_files:
                raise FileNotFoundError(f"No video files found in: {self.CAMERA_SOURCE}")
            return video_files
        elif self.CAMERA_SOURCE:
            return [self.CAMERA_SOURCE]
        else:
            raise ValueError("CAMERA_SOURCE is not set in environment variables")

    def get_first_video(self):
        files = self.get_video_files()
        return files[0] if files else None

# Single global instance
settings = Settings()

# Hardcoded constants
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
TRACK_HISTORY_LENGTH = 60
FRAME_SKIP = 2
COLORS = {
    "person": (0, 255, 0),
    "car": (255, 128, 0),
    "motorcycle": (0, 128, 255),
    "truck": (0, 0, 255),
    "bus": (128, 0, 255),
    "zone": (0, 255, 255),
    "alert": (0, 0, 255)
}

if __name__ == "__main__":
    print(f"CAMERA_SOURCE: {settings.CAMERA_SOURCE}")
    try:
        found_files = settings.get_video_files()
        print("Found files:", found_files)
        first_video = settings.get_first_video()
        print(f"First video: {first_video}")
    except Exception as e:
        print(f"Error finding videos: {e}")
    print("Settings loaded successfully")
