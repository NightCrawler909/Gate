import os
import sys
import logging
import time
from datetime import datetime
from collections import deque
from typing import List, Dict, Tuple, Optional, Any
from math import sqrt

import numpy as np
import cv2
import supervision as sv

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config.settings import settings, COLORS, TRACK_HISTORY_LENGTH
# Removed Detection import

# ──────────────────────────────────────────────────────────────────
# TrackedObject
# ──────────────────────────────────────────────────────────────────

class TrackedObject:
    """
    Represents one tracked entity (person or vehicle) across frames.

    Uses a regular __init__ — NOT a dataclass — so that mutable
    deque defaults are initialised safely per instance.
    """

    def __init__(
        self,
        track_id: int,
        object_type: str,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: tuple,
        bbox_xywh: tuple,
        center: tuple,
        first_seen_frame: int,
    ):
        self.track_id = track_id
        self.visitor_id = None             # UUID assigned on first zone entry

        self.object_type = object_type     # "person" or "vehicle"
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence

        self.bbox = bbox                   # (x1, y1, x2, y2) — current frame
        self.bbox_xywh = bbox_xywh        # (x, y, w, h)     — current frame
        self.center = center               # (cx, cy)          — current frame

        self.is_inside_zone = False
        self.first_seen_frame = first_seen_frame
        self.last_seen_frame = first_seen_frame
        self.first_seen_time = datetime.now()
        self.last_seen_time = datetime.now()

        # Bounded history deques — safe for long videos
        self.position_history = deque(maxlen=TRACK_HISTORY_LENGTH)
        self.frame_history    = deque(maxlen=TRACK_HISTORY_LENGTH)
        self.speed_history    = deque(maxlen=30)

        self.risk_score = 0.0
        self.is_anomaly = False
        self.alert_triggered = False

        self.plate_number  = None
        self.vehicle_color = None

        self.frames_in_zone = 0
        self.visit_count    = 0

        # Seed the history with the first observation
        self.position_history.append(center)
        self.frame_history.append(first_seen_frame)

    # ------------------------------------------------------------------
    def update(
        self,
        bbox: tuple,
        bbox_xywh: tuple,
        center: tuple,
        frame_number: int,
        confidence: float,
    ):
        """Apply a new detection observation from the current frame."""
        self.bbox        = bbox
        self.bbox_xywh   = bbox_xywh
        self.center      = center
        self.confidence  = confidence
        self.last_seen_frame = frame_number
        self.last_seen_time  = datetime.now()
        self.position_history.append(center)
        self.frame_history.append(frame_number)
        if self.is_inside_zone:
            self.frames_in_zone += 1

    # ------------------------------------------------------------------
    def get_dwell_seconds(self) -> float:
        """Return time spent inside zone in seconds (assumes 30 FPS)."""
        return self.frames_in_zone / 30.0

    def get_trajectory(self) -> list:
        """Return position history as a plain list of (cx, cy) tuples."""
        return list(self.position_history)

    def get_age_frames(self) -> int:
        """Return track age in frames."""
        return self.last_seen_frame - self.first_seen_frame

    def get_speed_pixels(self) -> float:
        """
        Return current speed in pixels per frame using the last 2
        recorded positions.  Returns 0.0 if fewer than 2 positions exist.
        """
        pos = list(self.position_history)
        if len(pos) < 2:
            return 0.0
        dx = pos[-1][0] - pos[-2][0]
        dy = pos[-1][1] - pos[-2][1]
        return sqrt(dx * dx + dy * dy)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise all fields to a JSON-safe dictionary."""
        return {
            "track_id":            self.track_id,
            "visitor_id":          self.visitor_id,
            "object_type":         self.object_type,
            "class_id":            self.class_id,
            "class_name":          self.class_name,
            "confidence":          round(self.confidence, 3),
            "bbox":                self.bbox,
            "bbox_xywh":           self.bbox_xywh,
            "center":              self.center,
            "is_inside_zone":      self.is_inside_zone,
            "first_seen_frame":    self.first_seen_frame,
            "last_seen_frame":     self.last_seen_frame,
            "first_seen_time":     self.first_seen_time.isoformat(),
            "last_seen_time":      self.last_seen_time.isoformat(),
            "risk_score":          self.risk_score,
            "is_anomaly":          self.is_anomaly,
            "alert_triggered":     self.alert_triggered,
            "plate_number":        self.plate_number,
            "vehicle_color":       self.vehicle_color,
            "frames_in_zone":      self.frames_in_zone,
            "visit_count":         self.visit_count,
            "dwell_seconds":       round(self.get_dwell_seconds(), 2),
            "age_frames":          self.get_age_frames(),
            "current_speed_pixels": round(self.get_speed_pixels(), 2),
            "trajectory":          list(self.position_history),
        }

    def __repr__(self) -> str:
        return (
            f"TrackedObject(id={self.track_id}, type={self.object_type}, "
            f"center=({self.center[0]:.0f},{self.center[1]:.0f}), "
            f"zone={self.is_inside_zone}, age={self.get_age_frames()}f)"
        )


# ──────────────────────────────────────────────────────────────────
# GateTracker
# ──────────────────────────────────────────────────────────────────

class GateTracker:
    """
    ByteTrack wrapper for the gate monitoring pipeline.

    Converts raw Detection objects from GateDetector into persistent
    TrackedObject instances with full position history, trail drawing,
    and zone-aware rendering.

    Usage
    -----
        tracker = GateTracker()
        tracked_list = tracker.update(detections, frame_number)
        frame = tracker.draw_tracks(frame)
        frame = tracker.draw_tracker_stats(frame)
    """

    def __init__(self):
        # supervision ≥ 0.20 uses sv.ByteTrack; older versions use sv.ByteTracker
        _ByteTrack = getattr(sv, "ByteTrack", None) or getattr(sv, "ByteTracker", None)
        if _ByteTrack is None:
            raise ImportError(
                "supervision has neither ByteTrack nor ByteTracker. "
                f"Installed version: {sv.__version__}"
            )
        try:
            self.tracker = _ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
        except TypeError:
            self.tracker = _ByteTrack()

        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.lost_tracks:     Dict[int, TrackedObject] = {}

        self.frame_count         = 0
        self.total_tracks_created = 0
        self.max_lost_age        = 150   # frames before a lost track is purged

        self.logger = logging.getLogger("GateMonitor")
        self.logger.info("[Tracker] GateTracker initialized with ByteTrack.")

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _get_class_name(self, class_id: int) -> str:
        """Map a COCO class index to a human-readable name."""
        mapping = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }
        return mapping.get(class_id, "unknown")

    def _detections_to_sv(
        self, detections: List[Any]
    ) -> sv.Detections:
        """Convert a list of Detection objects to supervision format."""
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array(
            [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]]
             for d in detections],
            dtype=np.float32,
        )
        confidence = np.array(
            [d.confidence for d in detections], dtype=np.float32
        )
        class_id = np.array(
            [d.class_id for d in detections], dtype=int
        )
        return sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id
        )

    def _cleanup_lost_tracks(self, frame_number: int):
        """Discard lost tracks that have been gone too long."""
        to_remove = [
            tid
            for tid, obj in self.lost_tracks.items()
            if frame_number - obj.last_seen_frame > self.max_lost_age
        ]
        for tid in to_remove:
            self.lost_tracks.pop(tid)
            self.logger.debug(f"[Tracker] Track {tid} discarded (too old).")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def update(
        self, detections: List[Any], frame_number: int
    ) -> List[TrackedObject]:
        """
        Main update method. Call every frame with detections from
        GateDetector.  Returns a list of all currently active
        TrackedObject instances.
        """
        self.frame_count += 1

        if not detections:
            self._cleanup_lost_tracks(frame_number)
            return list(self.tracked_objects.values())

        sv_dets = self._detections_to_sv(detections)

        try:
            tracked = self.tracker.update_with_detections(sv_dets)
        except Exception as exc:
            self.logger.error(f"[Tracker] ByteTrack update error: {exc}")
            return list(self.tracked_objects.values())

        if tracked is None or len(tracked) == 0:
            self._cleanup_lost_tracks(frame_number)
            return list(self.tracked_objects.values())

        if tracked.tracker_id is None:
            self._cleanup_lost_tracks(frame_number)
            return list(self.tracked_objects.values())

        active_ids: set = set()

        for i in range(len(tracked)):
            track_id     = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = tracked.xyxy[i]
            w  = x2 - x1
            h  = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            conf   = float(tracked.confidence[i]) \
                     if tracked.confidence is not None else 0.5
            cls_id = int(tracked.class_id[i]) \
                     if tracked.class_id is not None else 0

            bbox      = (float(x1), float(y1), float(x2), float(y2))
            bbox_xywh = (float(x1), float(y1), float(w),  float(h))
            center    = (float(cx), float(cy))
            cls_name  = self._get_class_name(cls_id)
            obj_type  = "person" if cls_id == 0 else "vehicle"

            active_ids.add(track_id)

            if track_id not in self.tracked_objects:
                if track_id in self.lost_tracks:
                    # Restore from lost buffer (re-identification)
                    self.tracked_objects[track_id] = \
                        self.lost_tracks.pop(track_id)
                    self.logger.info(
                        f"[Tracker] Track {track_id} restored from lost."
                    )
                else:
                    # Brand new track
                    obj = TrackedObject(
                        track_id=track_id,
                        object_type=obj_type,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=bbox,
                        bbox_xywh=bbox_xywh,
                        center=center,
                        first_seen_frame=frame_number,
                    )
                    self.tracked_objects[track_id] = obj
                    self.total_tracks_created += 1
                    self.logger.info(
                        f"[Tracker] New track {track_id}: "
                        f"{cls_name} at {center}"
                    )

            # Apply the new observation
            obj = self.tracked_objects[track_id]
            obj.update(
                bbox=bbox,
                bbox_xywh=bbox_xywh,
                center=center,
                frame_number=frame_number,
                confidence=conf,
            )
            obj.class_name  = cls_name
            obj.class_id    = cls_id
            obj.object_type = obj_type

        # Move unseen tracks to the lost buffer
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in active_ids:
                self.lost_tracks[track_id] = \
                    self.tracked_objects.pop(track_id)
                self.lost_tracks[track_id].last_seen_frame = frame_number

        self._cleanup_lost_tracks(frame_number)
        return list(self.tracked_objects.values())

    # ──────────────────────────────────────────────────────────────
    # Track accessors
    # ──────────────────────────────────────────────────────────────

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Return the active TrackedObject for track_id, or None."""
        return self.tracked_objects.get(track_id, None)

    def get_all_tracks(self) -> List[TrackedObject]:
        """Return all currently active TrackedObject instances."""
        return list(self.tracked_objects.values())

    def get_person_tracks(self) -> List[TrackedObject]:
        """Return active tracks whose object_type is 'person'."""
        return [o for o in self.tracked_objects.values()
                if o.object_type == "person"]

    def get_vehicle_tracks(self) -> List[TrackedObject]:
        """Return active tracks whose object_type is 'vehicle'."""
        return [o for o in self.tracked_objects.values()
                if o.object_type == "vehicle"]

    # ──────────────────────────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────────────────────────

    def draw_tracks(
        self,
        frame: np.ndarray,
        show_trail: bool = True,
        show_id: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, ID labels, movement trails, and zone /
        risk indicators for all active tracked objects.

        Each object is wrapped in try/except so a single bad bbox
        never crashes the entire draw pass.
        """
        for obj in self.tracked_objects.values():
            try:
                color = COLORS.get(obj.class_name, (128, 128, 128))
                x1 = int(obj.bbox[0])
                y1 = int(obj.bbox[1])
                x2 = int(obj.bbox[2])
                y2 = int(obj.bbox[3])

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"ID:{obj.track_id} {obj.class_name}"
                if obj.risk_score > 0:
                    label += f" R:{obj.risk_score:.0f}"

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - th - 8),
                    (x1 + tw + 4, y1),
                    color, -1,
                )
                cv2.putText(
                    frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1,
                )

                # Centre dot
                cv2.circle(
                    frame,
                    (int(obj.center[0]), int(obj.center[1])),
                    4, color, -1,
                )

                # Movement trail (last 20 positions)
                if show_trail:
                    positions = list(obj.position_history)
                    trail = positions[-20:]
                    if len(trail) > 1:
                        trail_color = (
                            int(color[0] * 0.6),
                            int(color[1] * 0.6),
                            int(color[2] * 0.6),
                        )
                        for j in range(len(trail) - 1):
                            pt1 = (int(trail[j][0]),     int(trail[j][1]))
                            pt2 = (int(trail[j + 1][0]), int(trail[j + 1][1]))
                            cv2.line(frame, pt1, pt2, trail_color, 1)

                # Cyan zone-membership ring
                if obj.is_inside_zone:
                    cv2.rectangle(
                        frame,
                        (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                        (0, 255, 255), 1,
                    )

                # Red alert border for high-risk objects
                if obj.risk_score >= settings.RISK_ALERT_THRESHOLD:
                    cv2.rectangle(
                        frame,
                        (x1 - 5, y1 - 5), (x2 + 5, y2 + 5),
                        (0, 0, 255), 2,
                    )

            except Exception as exc:
                self.logger.warning(
                    f"[Tracker] Draw error track {obj.track_id}: {exc}"
                )
                continue

        return frame

    def draw_tracker_stats(self, frame: np.ndarray) -> np.ndarray:
        """Draw a compact stats panel in the bottom-left corner."""
        lines = [
            f"Active : {len(self.tracked_objects)}",
            f"Lost   : {len(self.lost_tracks)}",
            f"Total  : {self.total_tracks_created}",
            f"Frame  : {self.frame_count}",
        ]
        x = 10
        y = frame.shape[0] - 110
        for i, line in enumerate(lines):
            yy = y + i * 25
            cv2.rectangle(frame, (x - 2, yy - 18), (x + 180, yy + 6),
                          (0, 0, 0), -1)
            cv2.putText(frame, line, (x, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1)
        return frame

    # ──────────────────────────────────────────────────────────────
    # Stats / reset
    # ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a snapshot of current tracker metrics."""
        return {
            "active_tracks":    len(self.tracked_objects),
            "lost_tracks":      len(self.lost_tracks),
            "total_created":    self.total_tracks_created,
            "frames_processed": self.frame_count,
            "persons":          len(self.get_person_tracks()),
            "vehicles":         len(self.get_vehicle_tracks()),
        }

    def reset(self):
        """Clear all track state — call between video files."""
        self.tracked_objects.clear()
        self.lost_tracks.clear()
        self.frame_count = 0
        self.total_tracks_created = 0
        self.logger.info("[Tracker] Tracker reset.")


# ──────────────────────────────────────────────────────────────────
# STANDALONE TEST BLOCK
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from core.detector import GateDetector
    from core.zone_manager import ZoneManager

    print("=" * 55)
    print("  GateTracker -- self test")
    print("=" * 55)
    print(f"supervision version: {sv.__version__}")

    # ── Initialise components ────────────────────────────────────
    detector = GateDetector(model_size="n")
    tracker  = GateTracker()
    zone     = ZoneManager()

    # ── Open first CCTV video ────────────────────────────────────
    files = settings.get_video_files()
    if not files:
        print("ERROR: No video files found — check CAMERA_SOURCE in .env")
        sys.exit(1)

    cap = cv2.VideoCapture(files[0])
    print(f"Processing: {os.path.basename(files[0])}")
    print(f"Zone defined: {zone.is_defined}")
    print()

    # ── Process 150 frames ──────────────────────────────────────
    # NOTE: HEVC 4K video intermittently returns ret=False for
    # individual frames even mid-stream — we skip failed reads
    # and continue rather than breaking the loop early.
    frame_number    = 0
    frames_read     = 0
    last_annotated  = None
    process_every   = 3
    consecutive_fails = 0
    max_consecutive_fails = 30   # only abort if 30 frames fail in a row

    while frames_read < 150:
        ret, frame = cap.read()
        frame_number += 1

        if not ret or frame is None:
            consecutive_fails += 1
            if consecutive_fails >= max_consecutive_fails:
                print(f"  [warn] {consecutive_fails} consecutive decode "
                      f"failures at frame {frame_number} — stopping.")
                break
            continue  # skip this frame, keep going

        consecutive_fails = 0
        frames_read += 1

        if frames_read % process_every != 0:
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
            if event:
                print(
                    f"  >> ZONE {event.event_type}: "
                    f"Track {event.track_id} "
                    f"({obj.class_name}) "
                    f"frame {event.frame_number}"
                )
            # Sync zone membership onto the object
            if zone.track_states.get(obj.track_id, False):
                obj.is_inside_zone = True
                obj.frames_in_zone += 1

        if frames_read % 30 == 0:
            stats = tracker.get_stats()
            print(
                f"Frame {frames_read:4d} | "
                f"Active:{stats['active_tracks']} "
                f"Lost:{stats['lost_tracks']} "
                f"Total:{stats['total_created']} "
                f"Persons:{stats['persons']} "
                f"Vehicles:{stats['vehicles']}"
            )

        last_annotated = frame.copy()

    cap.release()

    # ── Annotate and save final frame ────────────────────────────
    if last_annotated is not None:
        out = last_annotated.copy()
        out = zone.draw_zone(out)
        out = tracker.draw_tracks(out)
        out = zone.draw_zone_stats(out)
        out = tracker.draw_tracker_stats(out)

        os.makedirs("results", exist_ok=True)
        cv2.imwrite("results/tracker_test.jpg", out)
        print()
        print("Annotated frame saved to results/tracker_test.jpg")

    # ── Final summary ────────────────────────────────────────────
    stats = tracker.get_stats()
    print()
    print("=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    print(f"  Frames processed : {frame_number}")
    print(f"  Total tracks     : {stats['total_created']}")
    print(f"  Active tracks    : {stats['active_tracks']}")
    print(f"  Persons          : {stats['persons']}")
    print(f"  Vehicles         : {stats['vehicles']}")
    print(f"  Zone events      : {len(zone.event_history)}")
    print()

    detector.release()
    print("Tracker test complete")
    print("   Check results/tracker_test.jpg")
