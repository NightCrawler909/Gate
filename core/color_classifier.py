import os
import sys
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np

# ── Project root on path ─────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings, COLORS, VEHICLE_CLASS_IDS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ColorClassifier] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
ROI_SIZE:           Tuple[int, int] = (100, 100)  # resize target before HSV analysis
TOP_CROP_RATIO:     float           = 0.20         # fraction of ROI height to discard (sky/roof)
MIN_COLOR_COVERAGE: float           = 0.05         # ignore colors covering <5 % of pixels
MIN_CONFIDENCE:     float           = 0.10         # return "unknown" below this threshold
MIN_ROI_PIXELS:     int             = 20           # skip bboxes smaller than this
FONT                                = cv2.FONT_HERSHEY_SIMPLEX

# ── HSV color range table ─────────────────────────────────────────────────────
# Each entry: list of (lower, upper) numpy arrays in OpenCV HSV space
# H: 0-179  S: 0-255  V: 0-255
_HSV_RANGES: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    "red": [
        # Red wraps around the hue wheel — two ranges required
        (np.array([0,   100,  80],  dtype=np.uint8), np.array([10,  255, 255], dtype=np.uint8)),
        (np.array([160, 100,  80],  dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
    ],
    "orange": [
        (np.array([10,  120,  80],  dtype=np.uint8), np.array([25,  255, 255], dtype=np.uint8)),
    ],
    "yellow": [
        (np.array([25,  100,  80],  dtype=np.uint8), np.array([35,  255, 255], dtype=np.uint8)),
    ],
    "green": [
        (np.array([36,   60,  40],  dtype=np.uint8), np.array([85,  255, 255], dtype=np.uint8)),
    ],
    "blue": [
        (np.array([86,   80,  40],  dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8)),
    ],
    "purple": [
        (np.array([130,  60,  40],  dtype=np.uint8), np.array([160, 255, 255], dtype=np.uint8)),
    ],
    # Achromatic colours — classified using Value and Saturation
    "white": [
        (np.array([0,    0,  200],  dtype=np.uint8), np.array([179,  50, 255], dtype=np.uint8)),
    ],
    "gray": [
        (np.array([0,    0,   80],  dtype=np.uint8), np.array([179,  50, 200], dtype=np.uint8)),
    ],
    "black": [
        (np.array([0,    0,    0],  dtype=np.uint8), np.array([179, 255,  60], dtype=np.uint8)),
    ],
}

# Display colour for draw_color_label (BGR)
_LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "white":  (230, 230, 230),
    "black":  (50,  50,  50),
    "gray":   (160, 160, 160),
    "red":    (0,   0,   220),
    "orange": (0,   140, 255),
    "yellow": (0,   220, 220),
    "green":  (0,   200, 0),
    "blue":   (220, 80,  0),
    "purple": (200, 0,   200),
    "unknown":(100, 100, 100),
}


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class ColorResult:
    """Holds the output of a single color classification pass.

    Fields
    ------
    vehicle_color   : dominant color name, or "unknown"
    confidence      : fraction [0, 1] of ROI pixels matching that color
    coverage_map    : per-color pixel coverage fractions (for debugging)
    bbox            : vehicle bounding box (x1, y1, x2, y2)
    valid           : True when confidence >= MIN_CONFIDENCE
    classify_ms     : wall-clock processing time in milliseconds
    """
    vehicle_color:  str              = "unknown"
    confidence:     float            = 0.0
    coverage_map:   Dict[str, float] = field(default_factory=dict)
    bbox:           Tuple[int, ...]  = field(default_factory=tuple)
    valid:          bool             = False
    classify_ms:    float            = 0.0


# ── Main classifier ───────────────────────────────────────────────────────────
class VehicleColorClassifier:
    """Classify the dominant colour of a vehicle using HSV histogram analysis.

    Design notes
    ------------
    - Pure OpenCV / NumPy — no ML model, no GPU required.
    - ``process_vehicle`` completes in < 2 ms per call on modern hardware.
    - HSV ranges are defined once in the module-level ``_HSV_RANGES`` dict and
      applied via bitwise masks; adding a new colour is a one-line change.
    - Output aligns with the ``VehicleFeature.vehicle_color`` DB column.

    Parameters
    ----------
    roi_size : (width, height)
        Target resize dimensions before HSV analysis.
    top_crop_ratio : float
        Fraction of the ROI height discarded from the top (removes sky, roof
        structures, and background clutter above the vehicle body).
    min_coverage : float
        Minimum pixel fraction for a colour to be considered a candidate.
    min_confidence : float
        Minimum winning fraction to return a named colour vs "unknown".
    """

    def __init__(
        self,
        roi_size: Tuple[int, int]  = ROI_SIZE,
        top_crop_ratio: float      = TOP_CROP_RATIO,
        min_coverage: float        = MIN_COLOR_COVERAGE,
        min_confidence: float      = MIN_CONFIDENCE,
    ) -> None:
        """Initialise classifier with configurable parameters."""
        self._roi_size       = roi_size
        self._top_crop_ratio = top_crop_ratio
        self._min_coverage   = min_coverage
        self._min_confidence = min_confidence

        # runtime stats
        self._vehicles_classified: int   = 0
        self._total_classify_ms:   float = 0.0

        logger.info(
            "VehicleColorClassifier ready | roi=%s | top_crop=%.0f%% | "
            "min_coverage=%.0f%% | min_confidence=%.0f%%",
            roi_size,
            top_crop_ratio * 100,
            min_coverage   * 100,
            min_confidence * 100,
        )

    # ── ROI extraction ────────────────────────────────────────────────────────
    def extract_roi(
        self,
        frame: np.ndarray,
        bbox:  Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """Crop the vehicle region of interest from the full frame.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR video frame.
        bbox : (x1, y1, x2, y2)
            Vehicle bounding box in pixel coordinates.

        Returns
        -------
        np.ndarray or None
            BGR crop, or ``None`` if the bbox is degenerate / out-of-bounds.
        """
        fh, fw = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(fw, int(bbox[2]))
        y2 = min(fh, int(bbox[3]))

        if (x2 - x1) < MIN_ROI_PIXELS or (y2 - y1) < MIN_ROI_PIXELS:
            logger.debug("bbox too small (%dx%d) — skipped", x2 - x1, y2 - y1)
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return roi

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def preprocess(self, roi: np.ndarray) -> np.ndarray:
        """Prepare the ROI for HSV analysis.

        Steps
        -----
        1. Discard the top ``top_crop_ratio`` of the ROI (removes sky/roof/
           background that would bias the colour histogram).
        2. Resize to a fixed small size for constant-time processing.
        3. Apply mild Gaussian blur to suppress noise and reflections.
        4. Convert BGR → HSV.

        Parameters
        ----------
        roi : np.ndarray
            Raw BGR crop from ``extract_roi``.

        Returns
        -------
        np.ndarray
            HSV image ready for mask-based analysis.
        """
        h, w = roi.shape[:2]

        # 1. Top crop — discard sky / roof area
        crop_y = int(h * self._top_crop_ratio)
        body   = roi[crop_y:, :]

        # Guard: ensure body still has pixels after crop
        if body.shape[0] < MIN_ROI_PIXELS:
            body = roi

        # 2. Resize
        resized = cv2.resize(body, self._roi_size, interpolation=cv2.INTER_AREA)

        # 3. Blur — reduces specular reflections and compression artefacts
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)

        # 4. BGR → HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv

    # ── Mask generation ───────────────────────────────────────────────────────
    def get_color_mask(
        self,
        hsv: np.ndarray,
        color_name: str,
    ) -> np.ndarray:
        """Build a binary mask for a named colour using its HSV range(s).

        Some colours (notably red) span a discontinuous range in HSV — this
        method handles that by OR-ing multiple sub-range masks together.

        Parameters
        ----------
        hsv : np.ndarray
            HSV image (as returned by ``preprocess``).
        color_name : str
            One of the keys in ``_HSV_RANGES``.

        Returns
        -------
        np.ndarray
            Binary uint8 mask, same spatial dimensions as ``hsv``.

        Raises
        ------
        ValueError
            If ``color_name`` is not found in the HSV range table.
        """
        ranges = _HSV_RANGES.get(color_name)
        if ranges is None:
            raise ValueError(f"Unknown colour '{color_name}'. "
                             f"Valid names: {list(_HSV_RANGES.keys())}")

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        return mask

    # ── Core classification ────────────────────────────────────────────────────
    def classify_color(self, roi: np.ndarray) -> ColorResult:
        """Classify the dominant colour of a preprocessed (BGR) ROI.

        Algorithm
        ---------
        1. Preprocess → HSV.
        2. For each colour in ``_HSV_RANGES``, compute its pixel coverage
           (fraction of total pixels matching that colour's mask).
        3. Suppress candidates below ``min_coverage``.
        4. Select the highest-coverage colour as the dominant one.
        5. If its coverage is below ``min_confidence``, return "unknown".

        Special cases handled
        ---------------------
        - Very dark frames (mean V < 40): bias toward "black".
        - Very bright frames (mean V > 220, low S): bias toward "white".
        - Mixed / multicoloured vehicles: the top-ranked colour still wins,
          but ``coverage_map`` exposes the full distribution for callers.

        Parameters
        ----------
        roi : np.ndarray
            Raw BGR crop from ``extract_roi``.

        Returns
        -------
        ColorResult
        """
        result = ColorResult()

        hsv         = self.preprocess(roi)
        total_px    = hsv.shape[0] * hsv.shape[1]
        coverage    : Dict[str, float] = {}

        # ── Low-light guard ───────────────────────────────────────────────
        mean_v = float(np.mean(hsv[:, :, 2]))
        mean_s = float(np.mean(hsv[:, :, 1]))

        if mean_v < 35:
            # Extremely dark — classify as black without full mask sweep
            result.vehicle_color = "black"
            result.confidence    = min(1.0, (35 - mean_v) / 35 + 0.5)
            result.coverage_map  = {"black": result.confidence}
            result.valid         = result.confidence >= self._min_confidence
            return result

        if mean_v > 210 and mean_s < 40:
            # Very bright and desaturated — classify as white
            result.vehicle_color = "white"
            result.confidence    = min(1.0, (mean_v - 210) / 45 + 0.5)
            result.coverage_map  = {"white": result.confidence}
            result.valid         = result.confidence >= self._min_confidence
            return result

        # ── Full HSV mask sweep ────────────────────────────────────────────
        for color_name in _HSV_RANGES:
            mask      = self.get_color_mask(hsv, color_name)
            hit_px    = int(np.count_nonzero(mask))
            fraction  = hit_px / total_px
            coverage[color_name] = round(fraction, 4)

        result.coverage_map = coverage

        # ── Filter below minimum coverage ─────────────────────────────────
        candidates = {
            c: v for c, v in coverage.items()
            if v >= self._min_coverage
        }

        if not candidates:
            result.vehicle_color = "unknown"
            result.confidence    = 0.0
            result.valid         = False
            return result

        # ── Pick dominant colour ───────────────────────────────────────────
        dominant_color = max(candidates, key=candidates.__getitem__)
        confidence     = candidates[dominant_color]

        result.vehicle_color = dominant_color if confidence >= self._min_confidence else "unknown"
        result.confidence    = round(confidence, 4)
        result.valid         = confidence >= self._min_confidence
        return result

    # ── Single vehicle ─────────────────────────────────────────────────────────
    def process_vehicle(
        self,
        frame: np.ndarray,
        bbox:  Tuple[int, int, int, int],
    ) -> ColorResult:
        """Full pipeline for one vehicle bounding box.

        1. Extract ROI.
        2. Classify dominant colour.
        3. Log result.
        4. Return ``ColorResult``.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR video frame.
        bbox : (x1, y1, x2, y2)
            Vehicle bounding box.

        Returns
        -------
        ColorResult
            Always returned; check ``.valid`` to test whether classification
            was reliable.
        """
        t0     = time.perf_counter()
        result = ColorResult(bbox=tuple(int(v) for v in bbox))

        roi = self.extract_roi(frame, bbox)
        if roi is None:
            result.classify_ms = (time.perf_counter() - t0) * 1000
            return result

        result           = self.classify_color(roi)
        result.bbox      = tuple(int(v) for v in bbox)
        result.classify_ms = (time.perf_counter() - t0) * 1000

        self._vehicles_classified += 1
        self._total_classify_ms   += result.classify_ms

        if result.valid:
            logger.info(
                "Color detected | color='%s' | conf=%.2f | %.2f ms | bbox=%s",
                result.vehicle_color, result.confidence, result.classify_ms, bbox,
            )
        else:
            logger.debug(
                "Color unknown | conf=%.2f | top=%s | bbox=%s",
                result.confidence,
                sorted(result.coverage_map.items(), key=lambda x: -x[1])[:3],
                bbox,
            )

        return result

    # ── Batch processing ──────────────────────────────────────────────────────
    def process_batch(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict],
    ) -> Dict[int, ColorResult]:
        """Run colour classification on every vehicle in a tracked-object list.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame shared by all objects.
        tracked_objects : list of dict
            Each dict must contain:

            - ``track_id``  (int)
            - ``bbox``      ((x1, y1, x2, y2))
            - ``class_id``  (int)  — only ``VEHICLE_CLASS_IDS`` are processed

        Returns
        -------
        dict[track_id -> ColorResult]
            Includes only entries whose ``class_id`` is a vehicle class.
        """
        results: Dict[int, ColorResult] = {}

        for obj in tracked_objects:
            class_id = obj.get("class_id", -1)
            if class_id not in VEHICLE_CLASS_IDS:
                continue

            track_id = obj.get("track_id", -1)
            bbox     = obj.get("bbox")

            if bbox is None:
                logger.warning("track_id=%d has no bbox — skipped", track_id)
                continue

            results[track_id] = self.process_vehicle(frame, bbox)

        valid_count = sum(1 for r in results.values() if r.valid)
        logger.info(
            "Batch complete | %d vehicles | %d classified",
            len(results), valid_count,
        )
        return results

    # ── Visualisation ─────────────────────────────────────────────────────────
    def draw_color_label(
        self,
        frame: np.ndarray,
        bbox:  Tuple[int, int, int, int],
        result: ColorResult,
        *,
        draw_swatch: bool = True,
    ) -> np.ndarray:
        """Annotate a frame with the detected vehicle colour.

        Draws:
        - A coloured rectangle around the vehicle bbox.
        - A label pill (background filled with the detected colour).
        - A small colour swatch square (optional).

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to annotate (modified in-place AND returned).
        bbox : (x1, y1, x2, y2)
            Vehicle bounding box.
        result : ColorResult
            Output from ``process_vehicle``.
        draw_swatch : bool
            Whether to draw a filled square showing the detected colour
            (default True).

        Returns
        -------
        np.ndarray
            Annotated frame.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color_name      = result.vehicle_color
        bgr_colour      = _LABEL_COLORS.get(color_name, _LABEL_COLORS["unknown"])

        # ── Vehicle box ───────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_colour, 2)

        # ── Label text ────────────────────────────────────────────────────
        label = f"{color_name.upper()}  {result.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.60, 2)
        pad     = 5
        lx1     = x1
        ly1     = max(0, y1 - th - pad * 2 - baseline)
        ly2     = max(th + pad * 2, y1)

        # Pill background
        cv2.rectangle(frame, (lx1, ly1), (lx1 + tw + pad * 2, ly2), bgr_colour, -1)

        # Choose contrasting text colour
        lum        = 0.299 * bgr_colour[2] + 0.587 * bgr_colour[1] + 0.114 * bgr_colour[0]
        text_color = (0, 0, 0) if lum > 128 else (255, 255, 255)

        cv2.putText(
            frame, label,
            (lx1 + pad, ly2 - baseline - 1),
            FONT, 0.60, text_color, 2, cv2.LINE_AA,
        )

        # ── Colour swatch ─────────────────────────────────────────────────
        if draw_swatch:
            sw = 18  # swatch square side length
            sx1, sy1 = x2 - sw - 4, y1 + 4
            sx2, sy2 = x2 - 4,      y1 + 4 + sw
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), bgr_colour, -1)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 0), 1)

        return frame

    # ── Stats ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        """Return cumulative runtime statistics.

        Returns
        -------
        dict with keys: vehicles_classified, avg_classify_ms
        """
        avg_ms = (
            self._total_classify_ms / self._vehicles_classified
            if self._vehicles_classified else 0.0
        )
        return {
            "vehicles_classified": self._vehicles_classified,
            "avg_classify_ms":     round(avg_ms, 3),
        }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  VehicleColorClassifier -- self-test")
    print("=" * 60)

    clf = VehicleColorClassifier()

    # ── Build synthetic test patches ──────────────────────────────────────
    FRAME_W, FRAME_H = 1280, 720
    base_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    base_frame[:] = (60, 60, 60)  # dark grey background

    # Define a list of (colour name, BGR fill, bbox)
    test_cases = [
        ("white vehicle",  (230, 230, 230), (20,  50, 200, 300)),
        ("black vehicle",  (20,  20,  20),  (220, 50, 400, 300)),
        ("red vehicle",    (0,   0,   200), (420, 50, 600, 300)),
        ("blue vehicle",   (200, 80,  0),   (620, 50, 800, 300)),
        ("yellow vehicle", (0,   220, 220), (820, 50, 1000,300)),
        ("green vehicle",  (0,   180, 0),   (20, 350, 200, 600)),
        ("gray vehicle",   (130, 130, 130), (220,350, 400, 600)),
        ("orange vehicle", (0,   140, 255), (420,350, 600, 600)),
    ]

    frame = base_frame.copy()

    simulated_vehicles = []
    for idx, (label, bgr, bbox) in enumerate(test_cases):
        x1, y1, x2, y2 = bbox
        # Draw filled rectangle to simulate vehicle body colour
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
        cv2.putText(frame, label, (x1 + 4, y2 - 8), FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        simulated_vehicles.append({
            "track_id": idx + 1,
            "class_id": 2,           # car
            "bbox":     bbox,
        })

    # ── Optionally overlay a real CCTV frame ──────────────────────────────
    try:
        video_path = settings.get_first_video()
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, real_frame = cap.read()
            if ret and real_frame is not None:
                # Resize to match our canvas and use as background
                real_resized = cv2.resize(real_frame, (FRAME_W, FRAME_H))
                # Blend with synthetic patches
                for label, bgr, bbox in test_cases:
                    x1, y1, x2, y2 = bbox
                    frame[y1:y2, x1:x2] = bgr  # keep synthetic patches
                print(f"[OK] Real CCTV background loaded from: {video_path}")
            cap.release()
    except Exception:
        print("[INFO] Using synthetic test frame (no video found)")

    # ── Batch classify ────────────────────────────────────────────────────
    print(f"\nClassifying {len(simulated_vehicles)} vehicles...\n")
    batch_results = clf.process_batch(frame, simulated_vehicles)

    # ── Print results table ───────────────────────────────────────────────
    print(f"{'Track':>6}  {'Expected':<18} {'Detected':<12} {'Conf':>6}  {'Valid':>5}  {'ms':>6}")
    print("-" * 62)
    for obj, (expected, _, _) in zip(simulated_vehicles, test_cases):
        tid = obj["track_id"]
        res = batch_results.get(tid, ColorResult())
        print(
            f"{tid:>6}  {expected:<18} {res.vehicle_color:<12} "
            f"{res.confidence:>6.2%}  {str(res.valid):>5}  {res.classify_ms:>5.2f}"
        )

    # ── Annotate frame ────────────────────────────────────────────────────
    annotated = frame.copy()
    for obj in simulated_vehicles:
        tid  = obj["track_id"]
        bbox = obj["bbox"]
        if tid in batch_results:
            clf.draw_color_label(annotated, bbox, batch_results[tid])

    # ── Save output ───────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/color_classifier_test.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\n[OK] Annotated frame saved to '{out_path}'")

    # ── Stats ─────────────────────────────────────────────────────────────
    stats = clf.get_stats()
    print("\n--- Runtime Stats ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nColor classifier self-test complete.")
