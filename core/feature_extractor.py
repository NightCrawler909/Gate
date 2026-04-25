import os
import sys
import math
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import deque

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config.settings import settings
from core.tracker import TrackedObject


# ──────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────

PIXELS_PER_METER = 40.0
# Approximate conversion for overhead 1080p CCTV.
# 1 metre ≈ 40 pixels at a typical gate-camera height.
# Calibrate later by measuring a known distance (e.g. door width).

FPS = 30.0
# Assumed source frame rate for time calculations.

LOITERING_THRESHOLD_SECONDS = settings.DWELL_ALERT_SECONDS
# Pulled from .env — default 30 s.

SLOW_SPEED_THRESHOLD = 0.5
# pixels/frame — below this the person is considered stationary.

DIRECTION_NAMES = {
    "N":  (337.5, 360.0),
    "N2": (0.0,   22.5),
    "NE": (22.5,  67.5),
    "E":  (67.5,  112.5),
    "SE": (112.5, 157.5),
    "S":  (157.5, 202.5),
    "SW": (202.5, 247.5),
    "W":  (247.5, 292.5),
    "NW": (292.5, 337.5),
}


# ──────────────────────────────────────────────────────────────────
# FrameFeatures
# ──────────────────────────────────────────────────────────────────

class FrameFeatures:
    """
    All computed features for ONE tracked object at ONE frame.

    Returned by FeatureExtractor.extract() and optionally persisted
    to the database via FeatureExtractor.save_to_database().
    """

    def __init__(
        self,
        track_id: int,
        visitor_id: str,
        frame_number: int,
        object_type: str,
    ):
        self.track_id    = track_id
        self.visitor_id  = visitor_id
        self.frame_number = frame_number
        self.object_type = object_type
        self.timestamp   = datetime.now()

        # ── Position ──────────────────────────────────────────
        self.center_x = 0.0
        self.center_y = 0.0
        self.bbox_x   = 0.0
        self.bbox_y   = 0.0
        self.bbox_w   = 0.0
        self.bbox_h   = 0.0

        # ── Motion ────────────────────────────────────────────
        self.speed_pixels_per_frame = 0.0
        self.speed_mps              = 0.0
        self.direction_angle        = 0.0
        self.direction_name         = "UNKNOWN"
        self.is_stationary          = False

        # ── Zone ──────────────────────────────────────────────
        self.is_inside_zone = False
        self.dwell_seconds  = 0.0
        self.is_loitering   = False

        # ── Behavioral ────────────────────────────────────────
        self.trajectory_smoothness   = 1.0  # 1.0 = smooth, 0.0 = erratic
        self.approach_angle_to_zone  = 0.0
        self.visit_count             = 0

        # ── Trajectory JSON ───────────────────────────────────
        self.trajectory_json = "[]"         # JSON array of last 30 positions

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise all fields to a plain dictionary."""
        return {
            "track_id":                 self.track_id,
            "visitor_id":               self.visitor_id,
            "frame_number":             self.frame_number,
            "object_type":              self.object_type,
            "timestamp":                self.timestamp.isoformat(),
            "center_x":                 self.center_x,
            "center_y":                 self.center_y,
            "bbox_x":                   self.bbox_x,
            "bbox_y":                   self.bbox_y,
            "bbox_w":                   self.bbox_w,
            "bbox_h":                   self.bbox_h,
            "speed_pixels_per_frame":   self.speed_pixels_per_frame,
            "speed_mps":                self.speed_mps,
            "direction_angle":          self.direction_angle,
            "direction_name":           self.direction_name,
            "is_stationary":            self.is_stationary,
            "is_inside_zone":           self.is_inside_zone,
            "dwell_seconds":            self.dwell_seconds,
            "is_loitering":             self.is_loitering,
            "trajectory_smoothness":    self.trajectory_smoothness,
            "approach_angle_to_zone":   self.approach_angle_to_zone,
            "visit_count":              self.visit_count,
            "trajectory_json":          self.trajectory_json,
        }

    def __repr__(self) -> str:
        return (
            f"FrameFeatures(track={self.track_id}, "
            f"type={self.object_type}, "
            f"speed={self.speed_mps:.2f}m/s, "
            f"dir={self.direction_name}, "
            f"dwell={self.dwell_seconds:.1f}s, "
            f"loiter={self.is_loitering})"
        )


# ──────────────────────────────────────────────────────────────────
# FeatureExtractor
# ──────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Computes behavioural and spatial features for every tracked object
    every frame.

    All calculation methods handle edge cases (empty history, division
    by zero, NaN) and return safe defaults rather than raising.

    Usage
    -----
        extractor = FeatureExtractor()
        features = extractor.extract_all(tracked_objects, frame_number)
        summary  = extractor.get_summary(features)
    """

    def __init__(self, zone_center: Tuple[float, float] = None):
        """
        Args:
            zone_center: (cx, cy) centroid of the restricted zone.
                         Defaults to approximate centre of the default
                         gate zone for 1080p footage (660, 450).
        """
        self.zone_center = zone_center or (660.0, 450.0)
        self.logger = logging.getLogger("GateMonitor")
        self.logger.info(
            f"[FeatureExtractor] Initialized. "
            f"Zone center: {self.zone_center}"
        )

    # ──────────────────────────────────────────────────────────────
    # Core calculation methods
    # ──────────────────────────────────────────────────────────────

    def calculate_speed(
        self, position_history: deque, window: int = 5
    ) -> Tuple[float, float]:
        """
        Calculate speed as the average distance over the last N frame
        pairs.

        Args:
            position_history: deque of (cx, cy) tuples.
            window:           Number of consecutive pairs to average.

        Returns:
            (speed_pixels_per_frame, speed_mps)
        """
        positions = list(position_history)
        if len(positions) < 2:
            return (0.0, 0.0)

        # Take last (window + 1) positions; use all available if shorter
        recent = positions[-min(window + 1, len(positions)):]
        distances = []
        for i in range(len(recent) - 1):
            p1, p2 = recent[i], recent[i + 1]
            # Guard against NaN
            try:
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if math.isnan(dx) or math.isnan(dy):
                    continue
                distances.append(math.sqrt(dx * dx + dy * dy))
            except (TypeError, ValueError):
                continue

        if not distances:
            return (0.0, 0.0)

        avg_ppf   = sum(distances) / len(distances)
        speed_mps = (avg_ppf * FPS) / PIXELS_PER_METER

        return (round(avg_ppf, 3), round(speed_mps, 3))

    # ------------------------------------------------------------------
    def calculate_direction(
        self, position_history: deque
    ) -> Tuple[float, str]:
        """
        Calculate direction of movement as an angle in degrees and a
        compass name.

        Coordinate system (image space):
            0°   = moving right  (East)
            90°  = moving down   (South in image coords)
            180° = moving left   (West)
            270° = moving up     (North in image coords)

        Returns:
            (angle_degrees, direction_name)
        """
        positions = list(position_history)
        if len(positions) < 2:
            return (0.0, "STATIONARY")

        recent = positions[-1]
        lookback = min(5, len(positions) - 1)
        past = positions[-lookback - 1]

        dx = recent[0] - past[0]
        dy = recent[1] - past[1]

        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return (0.0, "STATIONARY")

        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360.0

        # Match angle to compass name
        name = "UNKNOWN"
        for label, (lo, hi) in DIRECTION_NAMES.items():
            if label == "N":
                if angle >= lo:            # 337.5–360
                    name = "N"
                    break
            elif label == "N2":
                if angle < hi:             # 0–22.5
                    name = "N"
                    break
            else:
                if lo <= angle < hi:
                    name = label
                    break

        return (round(angle, 1), name)

    # ------------------------------------------------------------------
    def calculate_trajectory_smoothness(
        self, position_history: deque
    ) -> float:
        """
        Measure how straight/smooth the path is.

        Returns:
            float in [0.0, 1.0] — 1.0 = perfectly straight,
                                   0.0 = very erratic.
        """
        positions = list(position_history)
        if len(positions) < 3:
            return 1.0

        pts = positions[-min(15, len(positions)):]

        angle_changes = []
        for i in range(1, len(pts) - 1):
            try:
                v1 = (pts[i][0] - pts[i - 1][0],
                      pts[i][1] - pts[i - 1][1])
                v2 = (pts[i + 1][0] - pts[i][0],
                      pts[i + 1][1] - pts[i][1])

                dot  = v1[0] * v2[0] + v1[1] * v2[1]
                mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot / (mag1 * mag2)
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    angle_change = math.degrees(math.acos(cos_angle))
                    angle_changes.append(angle_change)
            except (TypeError, ValueError, ZeroDivisionError):
                continue

        if not angle_changes:
            return 1.0

        avg_change = sum(angle_changes) / len(angle_changes)
        smoothness = max(0.0, 1.0 - (avg_change / 180.0))
        return round(smoothness, 3)

    # ------------------------------------------------------------------
    def calculate_approach_angle(
        self, position_history: deque
    ) -> float:
        """
        Angle between the object's movement vector and the direction
        toward the zone centre.

        Returns:
            0°   = moving directly toward zone centre
            180° = moving directly away from zone centre
        """
        positions = list(position_history)
        if len(positions) < 2:
            return 0.0

        current  = positions[-1]
        previous = positions[-2]

        mv = (current[0] - previous[0], current[1] - previous[1])
        zv = (self.zone_center[0] - current[0],
              self.zone_center[1] - current[1])

        mag_mv = math.sqrt(mv[0] ** 2 + mv[1] ** 2)
        mag_zv = math.sqrt(zv[0] ** 2 + zv[1] ** 2)

        if mag_mv < 1e-6 or mag_zv < 1e-6:
            return 0.0

        try:
            dot   = mv[0] * zv[0] + mv[1] * zv[1]
            cos_a = dot / (mag_mv * mag_zv)
            cos_a = max(-1.0, min(1.0, cos_a))
            angle = math.degrees(math.acos(cos_a))
        except (ValueError, ZeroDivisionError):
            return 0.0

        return round(angle, 1)

    # ------------------------------------------------------------------
    def build_trajectory_json(
        self, position_history: deque, last_n: int = 30
    ) -> str:
        """
        Serialise the last N positions to a compact JSON string.

        Format: [[cx1, cy1], [cx2, cy2], ...]
        """
        positions = list(position_history)
        recent    = positions[-last_n:]
        rounded   = [[round(p[0], 1), round(p[1], 1)] for p in recent]
        return json.dumps(rounded)

    # ──────────────────────────────────────────────────────────────
    # Main extraction method
    # ──────────────────────────────────────────────────────────────

    def extract(
        self, obj: TrackedObject, frame_number: int
    ) -> FrameFeatures:
        """
        Extract all features for one TrackedObject at the current frame.
        Updates obj.speed_history in-place.

        Returns:
            FrameFeatures instance (never raises).
        """
        ff = FrameFeatures(
            track_id=obj.track_id,
            visitor_id=obj.visitor_id or "UNKNOWN",
            frame_number=frame_number,
            object_type=obj.object_type,
        )

        try:
            # ── Position ──────────────────────────────────────
            ff.center_x = float(obj.center[0])
            ff.center_y = float(obj.center[1])
            ff.bbox_x   = float(obj.bbox_xywh[0])
            ff.bbox_y   = float(obj.bbox_xywh[1])
            ff.bbox_w   = float(obj.bbox_xywh[2])
            ff.bbox_h   = float(obj.bbox_xywh[3])

            # ── Motion ────────────────────────────────────────
            speed_ppf, speed_mps = self.calculate_speed(
                obj.position_history
            )
            ff.speed_pixels_per_frame = speed_ppf
            ff.speed_mps              = speed_mps
            ff.is_stationary          = speed_ppf < SLOW_SPEED_THRESHOLD

            angle, direction = self.calculate_direction(
                obj.position_history
            )
            ff.direction_angle = angle
            ff.direction_name  = direction

            obj.speed_history.append(speed_ppf)

            # ── Zone ──────────────────────────────────────────
            ff.is_inside_zone = obj.is_inside_zone
            ff.dwell_seconds  = obj.get_dwell_seconds()
            ff.is_loitering   = (
                obj.is_inside_zone
                and ff.dwell_seconds >= LOITERING_THRESHOLD_SECONDS
            )
            ff.visit_count = obj.visit_count

            # ── Behavioral ────────────────────────────────────
            ff.trajectory_smoothness  = self.calculate_trajectory_smoothness(
                obj.position_history
            )
            ff.approach_angle_to_zone = self.calculate_approach_angle(
                obj.position_history
            )

            # ── Trajectory JSON ───────────────────────────────
            ff.trajectory_json = self.build_trajectory_json(
                obj.position_history
            )

        except Exception as exc:
            self.logger.warning(
                f"[FeatureExtractor] extract() error "
                f"track {obj.track_id}: {exc}"
            )

        return ff

    # ──────────────────────────────────────────────────────────────
    # Batch extraction
    # ──────────────────────────────────────────────────────────────

    def extract_all(
        self,
        tracked_objects: List[TrackedObject],
        frame_number: int,
    ) -> List[FrameFeatures]:
        """
        Extract features for ALL tracked objects in one call.

        Returns:
            List of FrameFeatures (never raises, skips bad objects).
        """
        results = []
        for obj in tracked_objects:
            try:
                ff = self.extract(obj, frame_number)
                results.append(ff)
            except Exception as exc:
                self.logger.warning(
                    f"[FeatureExtractor] extract_all() skipped "
                    f"track {obj.track_id}: {exc}"
                )
        return results

    # ──────────────────────────────────────────────────────────────
    # Database persistence
    # ──────────────────────────────────────────────────────────────

    def save_to_database(
        self, features: List[FrameFeatures], db_session
    ) -> int:
        """
        Persist extracted features for zone-entered objects.

        Rules:
        - Only objects with a real visitor_id (zone entry happened)
        - Only every 10th frame to avoid flooding the database

        Returns:
            Number of records saved.
        """
        from database import crud

        saved_count = 0
        for ff in features:
            try:
                if ff.visitor_id == "UNKNOWN":
                    continue
                if ff.frame_number % 10 != 0:
                    continue

                if ff.object_type == "person":
                    crud.save_person_feature(
                        db=db_session,
                        visitor_id=ff.visitor_id,
                        frame_number=ff.frame_number,
                        bbox=(ff.bbox_x, ff.bbox_y,
                              ff.bbox_w, ff.bbox_h),
                        center=(ff.center_x, ff.center_y),
                        speed_ppf=ff.speed_pixels_per_frame,
                        speed_mps=ff.speed_mps,
                        direction_angle=ff.direction_angle,
                        is_inside_zone=ff.is_inside_zone,
                        trajectory_json=ff.trajectory_json,
                    )
                    saved_count += 1

                elif ff.object_type == "vehicle":
                    crud.save_vehicle_feature(
                        db=db_session,
                        visitor_id=ff.visitor_id,
                        frame_number=ff.frame_number,
                        bbox=(ff.bbox_x, ff.bbox_y,
                              ff.bbox_w, ff.bbox_h),
                        center=(ff.center_x, ff.center_y),
                        speed_ppf=ff.speed_pixels_per_frame,
                        is_inside_zone=ff.is_inside_zone,
                    )
                    saved_count += 1

            except Exception as exc:
                self.logger.warning(
                    f"[FeatureExtractor] DB save failed "
                    f"track {ff.track_id}: {exc}"
                )

        return saved_count

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────

    def get_summary(self, features: List[FrameFeatures]) -> dict:
        """
        Aggregate statistics across a list of FrameFeatures.
        Useful for logging and dashboard display.
        """
        if not features:
            return {}

        speeds    = [f.speed_mps for f in features]
        loiterers = [f for f in features if f.is_loitering]
        stationary = [f for f in features if f.is_stationary]
        persons   = [f for f in features if f.object_type == "person"]
        vehicles  = [f for f in features if f.object_type == "vehicle"]

        return {
            "total_objects":       len(features),
            "persons":             len(persons),
            "vehicles":            len(vehicles),
            "avg_speed_mps":       round(sum(speeds) / len(speeds), 3)
                                   if speeds else 0.0,
            "max_speed_mps":       round(max(speeds), 3)
                                   if speeds else 0.0,
            "loitering_count":     len(loiterers),
            "stationary_count":    len(stationary),
            "loitering_track_ids": [f.track_id for f in loiterers],
        }


# ──────────────────────────────────────────────────────────────────
# STANDALONE TEST BLOCK
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from core.detector import GateDetector
    from core.tracker import GateTracker
    from core.zone_manager import ZoneManager

    print("=" * 55)
    print("  FeatureExtractor -- self test")
    print("=" * 55)

    # ── Initialise ──────────────────────────────────────────────
    detector  = GateDetector(model_size="n")
    tracker   = GateTracker()
    zone      = ZoneManager()
    extractor = FeatureExtractor()

    files = settings.get_video_files()
    if not files:
        print("ERROR: No video files found — check CAMERA_SOURCE in .env")
        sys.exit(1)

    cap = cv2.VideoCapture(files[0])
    print(f"Video: {os.path.basename(files[0])}")
    print()

    # ── Process 150 frames (skip-on-fail for HEVC 4K) ───────────
    frame_number      = 0
    frames_read       = 0
    all_features: List[FrameFeatures] = []
    last_features:    List[FrameFeatures] = []
    consecutive_fails = 0
    max_fails         = 30

    while frames_read < 150:
        ret, frame = cap.read()
        frame_number += 1

        if not ret or frame is None:
            consecutive_fails += 1
            if consecutive_fails >= max_fails:
                print(f"  [warn] {consecutive_fails} consecutive decode "
                      f"failures at frame {frame_number} — stopping.")
                break
            continue

        consecutive_fails = 0
        frames_read += 1

        if frames_read % 3 != 0:
            continue

        detections   = detector.detect(frame, frame_number)
        tracked_objs = tracker.update(detections, frame_number)

        for obj in tracked_objs:
            event = zone.check_zone_event(
                obj.track_id,
                obj.center[0],
                obj.center[1],
                frame_number,
            )
            if event and event.event_type == "ENTRY":
                obj.visit_count += 1
                obj.is_inside_zone = True

        features = extractor.extract_all(tracked_objs, frame_number)
        all_features.extend(features)
        last_features = features   # keep for annotation

        if frames_read % 30 == 0 and features:
            print(f"Frame {frames_read}:")
            for ff in features:
                print(
                    f"  Track {ff.track_id:3d} "
                    f"| {ff.object_type:7s} "
                    f"| speed: {ff.speed_mps:.2f} m/s "
                    f"| dir: {ff.direction_name:12s} "
                    f"| dwell: {ff.dwell_seconds:.1f}s "
                    f"| smooth: {ff.trajectory_smoothness:.2f} "
                    f"| loiter: {ff.is_loitering}"
                )
            print()

    cap.release()

    # ── Summary ──────────────────────────────────────────────────
    print("=" * 55)
    print("  FEATURE SUMMARY")
    print("=" * 55)
    if all_features:
        summary = extractor.get_summary(all_features)
        for k, v in summary.items():
            print(f"  {k:25s}: {v}")
    print()

    # ── Math unit tests ──────────────────────────────────────────
    print("--- Math unit tests ---")
    test_history = deque(maxlen=60)
    for p in [
        (100, 100), (110, 105), (120, 110),
        (130, 115), (140, 120), (150, 125),
        (160, 130), (170, 135), (180, 140),
    ]:
        test_history.append(p)

    speed_ppf, speed_mps = extractor.calculate_speed(test_history)
    angle, name          = extractor.calculate_direction(test_history)
    smooth               = extractor.calculate_trajectory_smoothness(test_history)
    approach             = extractor.calculate_approach_angle(test_history)
    traj_json            = extractor.build_trajectory_json(test_history)

    print(f"  Speed     : {speed_ppf:.2f} px/frame, {speed_mps:.2f} m/s")
    print(f"  Direction : {angle:.1f}° ({name})")
    print(f"  Smoothness: {smooth:.3f}")
    print(f"  Approach angle: {approach:.1f}°")
    print(f"  Trajectory JSON (first 50 chars): {traj_json[:50]}")
    print()

    # ── Annotated frame ──────────────────────────────────────────
    cap2 = cv2.VideoCapture(files[0])
    ann_frame = None
    ann_fails = 0
    count = 0
    while count < 100:
        ret, frm = cap2.read()
        if not ret or frm is None:
            ann_fails += 1
            if ann_fails > 30:
                break
            continue
        ann_fails = 0
        count += 1
        ann_frame = frm
    cap2.release()

    if ann_frame is not None:
        out = ann_frame.copy()
        out = zone.draw_zone(out)
        out = tracker.draw_tracks(out)
        out = tracker.draw_tracker_stats(out)

        # Draw speed / direction labels below each bbox
        for ff in last_features:
            try:
                # Find the matching tracked object for its bbox
                obj = tracker.get_track(ff.track_id)
                if obj is None:
                    continue
                x1 = int(obj.bbox[0])
                y2 = int(obj.bbox[3])

                texts = [
                    f"{ff.speed_mps:.1f}m/s",
                    f"{ff.direction_name}",
                    f"D:{ff.dwell_seconds:.0f}s",
                ]
                for j, text in enumerate(texts):
                    ty = y2 + 18 + j * 18
                    # Shadow
                    cv2.putText(out, text, (x1 + 1, ty + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 2, cv2.LINE_AA)
                    # Foreground in yellow
                    cv2.putText(out, text, (x1, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                continue

        os.makedirs("results", exist_ok=True)
        cv2.imwrite("results/features_test.jpg", out)
        print("Saved results/features_test.jpg")

    # ── Cleanup ──────────────────────────────────────────────────
    detector.release()
    print()
    print("FeatureExtractor test complete")
    print("   Check results/features_test.jpg")
