import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Allow imports from project root regardless of working directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from config.settings import settings
from database.db import init_db, SessionLocal

# ──────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

log_filename = logs_dir / f"processing_{datetime.now().strftime('%Y-%m-%d')}.log"

logger = logging.getLogger("GateMonitor")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ──────────────────────────────────────────────
# VideoMetadata CLASS
# ──────────────────────────────────────────────
class VideoMetadata:
    """Holds all metadata and processing stats for a single video file."""

    def __init__(self, file_path: str):
        self.file_path: str = file_path
        self.file_name: str = os.path.basename(file_path)
        self.file_size_mb: float = 0.0
        self.duration_seconds: float = 0.0
        self.total_frames: int = 0
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.processed_frames: int = 0
        self.skipped_frames: int = 0
        self.start_time: datetime = None
        self.end_time: datetime = None
        self.status: str = "pending"  # pending | processing | completed | failed
        self.error_message: str = None

    def to_dict(self) -> dict:
        """Serialize metadata to a JSON-safe dictionary."""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size_mb": round(self.file_size_mb, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2),
            "width": self.width,
            "height": self.height,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_seconds": (
                round((self.end_time - self.start_time).total_seconds(), 2)
                if self.start_time and self.end_time else None
            ),
            "status": self.status,
            "error_message": self.error_message,
        }


# ──────────────────────────────────────────────
# VideoProcessor CLASS
# ──────────────────────────────────────────────
class VideoProcessor:
    """Main processor that scans a folder of CCTV videos and processes them one by one."""

    def __init__(self):
        logger.info("Initialising VideoProcessor...")

        # Load settings
        self.settings = settings

        # Initialise database tables
        init_db()

        # Discover video files
        self.video_files = self.settings.get_video_files()
        logger.info(f"Discovered {len(self.video_files)} video(s) to process.")

        # State tracking
        self.processed_videos: dict[str, VideoMetadata] = {}
        self.is_running: bool = True

        # Ensure results folder exists
        Path("results").mkdir(exist_ok=True)

    # ──────────────────────────────────────────────
    def get_video_metadata(self, video_path: str) -> "VideoMetadata":
        """
        Opens the video and reads its core properties without decoding frames.
        Returns a populated VideoMetadata object, or None on failure.
        """
        meta = VideoMetadata(video_path)
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"cv2 could not open: {video_path}")

            meta.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            meta.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta.duration_seconds = meta.total_frames / meta.fps if meta.fps else 0.0
            meta.file_size_mb = os.path.getsize(video_path) / 1024 / 1024
        except Exception as exc:
            logger.error(f"Failed to read metadata for {video_path}: {exc}")
            return None
        finally:
            if cap:
                cap.release()
        return meta

    # ──────────────────────────────────────────────
    def process_single_video(self, video_path: str) -> VideoMetadata:
        """
        Reads every frame of a single video, logs progress, and returns
        the completed VideoMetadata object. Never raises — errors are caught
        and the status is set to 'failed' so the caller can move on.
        """
        meta = self.get_video_metadata(video_path)
        if meta is None:
            # Create a minimal object so the caller still gets something back
            meta = VideoMetadata(video_path)
            meta.status = "failed"
            meta.error_message = "Could not read video metadata"
            return meta

        meta.status = "processing"
        meta.start_time = datetime.now()

        logger.info(
            f"Starting: {meta.file_name} | "
            f"{meta.duration_seconds:.1f}s | "
            f"{meta.total_frames} frames | "
            f"{meta.fps:.2f}fps | "
            f"{meta.file_size_mb:.2f}MB"
        )

        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Reduce memory buffer

            if not cap.isOpened():
                raise IOError(f"Failed to open video for frame reading: {video_path}")

            last_heartbeat = time.time()

            while self.is_running:
                ret, frame = cap.read()

                if not ret:
                    break  # End of video or unreadable frame

                if frame is None:
                    meta.skipped_frames += 1
                    continue

                meta.processed_frames += 1

                # Immediately discard frame — do NOT accumulate in memory
                del frame

                # ── Log progress every 500 frames ──
                if meta.processed_frames % 500 == 0 and meta.total_frames > 0:
                    percent = (meta.processed_frames / meta.total_frames) * 100
                    elapsed = (datetime.now() - meta.start_time).total_seconds()
                    logger.info(
                        f"Progress: {meta.file_name} | "
                        f"Frame {meta.processed_frames}/{meta.total_frames} | "
                        f"{percent:.1f}% | "
                        f"Elapsed: {elapsed:.0f}s"
                    )

                # ── Heartbeat every 30 seconds ──
                now = time.time()
                if now - last_heartbeat >= 30:
                    percent = (
                        (meta.processed_frames / meta.total_frames) * 100
                        if meta.total_frames > 0 else 0
                    )
                    logger.info(
                        f"HEARTBEAT: {meta.file_name} still processing... "
                        f"{percent:.1f}% complete"
                    )
                    last_heartbeat = now

            # ── Mark as completed ──
            meta.status = "completed"
            meta.end_time = datetime.now()
            elapsed = (meta.end_time - meta.start_time).total_seconds()
            logger.info(
                f"COMPLETED: {meta.file_name} | "
                f"Processed {meta.processed_frames} frames in {elapsed:.1f}s"
            )

        except Exception as exc:
            meta.status = "failed"
            meta.error_message = str(exc)
            meta.end_time = datetime.now()
            logger.error(f"ERROR processing {meta.file_name}: {exc}")
            logger.error(traceback.format_exc())

        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()

        return meta

    # ──────────────────────────────────────────────
    def save_progress_report(self):
        """
        Serialises current processing state to JSON in the results/ folder.
        Writes both a latest snapshot and a timestamped backup.
        """
        completed = [m for m in self.processed_videos.values() if m.status == "completed"]
        failed    = [m for m in self.processed_videos.values() if m.status == "failed"]
        pending   = [m for m in self.processed_videos.values() if m.status == "pending"]
        total_frames = sum(m.processed_frames for m in self.processed_videos.values())

        report = {
            "report_generated_at": datetime.now().isoformat(),
            "total_videos": len(self.video_files),
            "completed": len(completed),
            "failed": len(failed),
            "pending": len(pending),
            "total_frames_processed": total_frames,
            "details": [m.to_dict() for m in self.processed_videos.values()],
        }

        # Latest report (always overwritten)
        latest_path = Path("results") / "progress_report.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Timestamped backup snapshot
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path("results") / f"progress_report_{ts}.json"
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Progress report saved to results/")

    # ──────────────────────────────────────────────
    def run(self):
        """
        Main entry point. Iterates through all discovered videos,
        processes each one, and saves progress after every video.
        """
        total = len(self.video_files)
        run_start = datetime.now()

        logger.info("═══════════════════════════════════════════════")
        logger.info("   GATE MONITOR VIDEO PROCESSOR STARTED")
        logger.info(f"  Found {total} videos to process")
        logger.info(f"  Logs folder:    logs/")
        logger.info(f"  Results folder: results/")
        logger.info("  You can safely leave — processing will continue")
        logger.info("═══════════════════════════════════════════════")

        for i, video_path in enumerate(self.video_files):
            filename = os.path.basename(video_path)
            logger.info("───────────────────────────────────")
            logger.info(f"VIDEO {i + 1} of {total}: {filename}")
            logger.info("───────────────────────────────────")

            meta = self.process_single_video(video_path)
            self.processed_videos[video_path] = meta

            # Save a snapshot after every video
            self.save_progress_report()

            status_icon = "✓" if meta.status == "completed" else "✗"
            elapsed = (
                round((meta.end_time - meta.start_time).total_seconds(), 1)
                if meta.start_time and meta.end_time else "N/A"
            )
            logger.info(
                f"{status_icon} Summary: {filename} | "
                f"Status={meta.status} | "
                f"Frames={meta.processed_frames} | "
                f"Time={elapsed}s"
            )

        # ── Final report ──
        self.save_progress_report()
        run_end = datetime.now()
        total_elapsed_min = round((run_end - run_start).total_seconds() / 60, 1)

        completed_count = sum(1 for m in self.processed_videos.values() if m.status == "completed")
        failed_count    = sum(1 for m in self.processed_videos.values() if m.status == "failed")
        total_frames    = sum(m.processed_frames for m in self.processed_videos.values())

        logger.info("═══════════════════════════════════════════════")
        logger.info("            ALL VIDEOS PROCESSED")
        logger.info(f"  Total videos : {total}")
        logger.info(f"  Completed    : {completed_count}")
        logger.info(f"  Failed       : {failed_count}")
        logger.info(f"  Total frames : {total_frames:,}")
        logger.info(f"  Total time   : {total_elapsed_min} minutes")
        logger.info("  Final report saved to results/progress_report.json")
        logger.info("═══════════════════════════════════════════════")
