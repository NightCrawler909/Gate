import os
import sys
import re
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np

# ── Project root on path ────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings, COLORS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ANPR] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_PLATE_CHARS: int   = 5          # minimum valid plate length
MIN_CONFIDENCE:  float = 0.25       # discard EasyOCR results below this
PLATE_ROI_RATIO: float = 0.35       # bottom-N fraction of bbox used as plate ROI
MIN_ROI_PIXELS:  int   = 20         # skip ROI smaller than this in any dimension
FONT                   = cv2.FONT_HERSHEY_SIMPLEX


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class PlateResult:
    """Holds the result of a single ANPR pass on one vehicle bbox.

    Fields
    ------
    plate_text      : cleaned, uppercase plate string (or empty string)
    confidence      : float in [0, 1]; 0.0 means no plate found
    raw_texts       : all raw OCR strings returned (before filtering)
    bbox            : the vehicle bounding box (x1, y1, x2, y2)
    roi_bbox        : the cropped plate ROI bbox in frame coords
    valid           : True when plate_text meets minimum length requirement
    inference_ms    : wall-clock time for the OCR call in milliseconds
    """
    plate_text:    str              = ""
    confidence:    float            = 0.0
    raw_texts:     List[str]        = field(default_factory=list)
    bbox:          Tuple[int, ...]  = field(default_factory=tuple)
    roi_bbox:      Tuple[int, ...]  = field(default_factory=tuple)
    valid:         bool             = False
    inference_ms:  float            = 0.0

    # ── Aliases that match VehicleFeature column names ────────────────────
    @property
    def plate_number(self) -> Optional[str]:
        """Returns plate_text if valid, else None — maps to VehicleFeature.plate_number."""
        return self.plate_text if self.valid else None

    @property
    def plate_confidence(self) -> float:
        """Returns confidence — maps to VehicleFeature.plate_confidence."""
        return self.confidence


# ── Main ANPR class ───────────────────────────────────────────────────────────
class ANPRSystem:
    """Automatic Number Plate Recognition using EasyOCR on CUDA.

    Design principles
    -----------------
    - EasyOCR ``Reader`` is instantiated **once** in ``__init__`` and reused for
      every subsequent call — avoids the ~3 s GPU model-load penalty per frame.
    - All preprocessing is done on CPU (cheap NumPy/OpenCV ops) so the GPU
      stays free for the OCR LSTM pass.
    - Output ``PlateResult`` fields are named to align directly with the
      ``VehicleFeature`` SQLAlchemy model (``plate_number``, ``plate_confidence``).

    Parameters
    ----------
    gpu : bool
        Pass ``False`` to force CPU-only mode (useful for unit tests).
    languages : list[str]
        EasyOCR language list.  Default ``['en']``.
    """

    def __init__(
        self,
        gpu: bool = True,
        languages: Optional[List[str]] = None,
    ) -> None:
        """Initialise EasyOCR reader (loads model weights once)."""
        if languages is None:
            languages = ["en"]

        self._gpu = gpu
        self._languages = languages
        self._reader = None          # lazy-init on first use

        # stats
        self._plates_read: int    = 0
        self._plates_valid: int   = 0
        self._total_ocr_ms: float = 0.0

        logger.info(
            "ANPRSystem initialised | gpu=%s | languages=%s",
            gpu, languages,
        )

    # ── Lazy reader init ──────────────────────────────────────────────────────
    def _get_reader(self):
        """Return the EasyOCR Reader, initialising it on first call.

        Deferred so the expensive GPU load only happens when ANPR is actually
        needed (not at import time or during unit tests that mock the class).
        """
        if self._reader is None:
            try:
                import easyocr  # noqa: PLC0415
                logger.info("Loading EasyOCR (gpu=%s) — first call only…", self._gpu)
                t0 = time.perf_counter()
                self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info("EasyOCR ready in %.0f ms", elapsed)
            except ImportError as exc:
                raise ImportError(
                    "easyocr is not installed.  Run: pip install easyocr"
                ) from exc
        return self._reader

    # ── Plate-ROI extraction ─────────────────────────────────────────────────
    def extract_plate(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """Crop the likely plate region from a vehicle bounding box.

        Number plates appear in the lower portion of a vehicle bbox (front/rear).
        We take the bottom ``PLATE_ROI_RATIO`` fraction of the bbox height, and
        a horizontal centre crop (middle 80 %) to discard irrelevant background.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR video frame.
        bbox : (x1, y1, x2, y2)
            Vehicle bounding box in pixel coordinates.

        Returns
        -------
        roi : np.ndarray or None
            Cropped ROI, or ``None`` if the bbox is too small.
        roi_bbox : (rx1, ry1, rx2, ry2)
            Absolute frame coordinates of the returned ROI.
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = (
            max(0, int(bbox[0])),
            max(0, int(bbox[1])),
            min(fw, int(bbox[2])),
            min(fh, int(bbox[3])),
        )

        bw = x2 - x1
        bh = y2 - y1

        if bw < MIN_ROI_PIXELS or bh < MIN_ROI_PIXELS:
            logger.debug("bbox too small (%dx%d) — skipping", bw, bh)
            return None, (x1, y1, x2, y2)

        # Bottom fraction for plate region
        plate_h  = max(MIN_ROI_PIXELS, int(bh * PLATE_ROI_RATIO))
        ry1      = y2 - plate_h
        ry2      = y2

        # Horizontal centre crop (trim 10 % each side)
        h_margin = int(bw * 0.10)
        rx1      = x1 + h_margin
        rx2      = x2 - h_margin

        if (rx2 - rx1) < MIN_ROI_PIXELS or (ry2 - ry1) < MIN_ROI_PIXELS:
            return None, (rx1, ry1, rx2, ry2)

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None, (rx1, ry1, rx2, ry2)

        return roi, (rx1, ry1, rx2, ry2)

    # ── Image preprocessing ──────────────────────────────────────────────────
    def preprocess_plate(self, roi: np.ndarray) -> np.ndarray:
        """Apply lightweight preprocessing to maximise OCR accuracy.

        Pipeline
        --------
        1. Upscale small ROIs to at least 100 px tall (bilinear).
        2. Convert to grayscale.
        3. CLAHE contrast enhancement.
        4. Bilateral denoise (preserves edges better than Gaussian).
        5. Adaptive thresholding — handles uneven lighting common in CCTV.

        Parameters
        ----------
        roi : np.ndarray
            BGR crop of the plate region.

        Returns
        -------
        np.ndarray
            Binary (or near-binary) uint8 image ready for EasyOCR.
        """
        # 1. Upscale if necessary so OCR has enough pixels to work with
        h, w = roi.shape[:2]
        if h < 100:
            scale = 100 / h
            roi = cv2.resize(roi, (int(w * scale), 100), interpolation=cv2.INTER_LINEAR)

        # 2. Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE — local contrast normalisation
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 4. Bilateral filter — smooth noise, keep character edges sharp
        denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # 5. Adaptive threshold — robust against shadows / partial occlusion
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        return binary

    # ── OCR call ─────────────────────────────────────────────────────────────
    def read_plate(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """Run EasyOCR on a preprocessed plate image.

        Parameters
        ----------
        image : np.ndarray
            Preprocessed (typically binary) image.

        Returns
        -------
        list of (text, confidence) tuples, sorted by descending confidence.
        Empty list if no text detected.
        """
        reader = self._get_reader()
        try:
            results = reader.readtext(
                image,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
                detail=1,
                paragraph=False,
                batch_size=1,
                # width_ths / height_ths tune bbox merging
                width_ths=0.7,
                height_ths=0.7,
            )
            # results: list of ( [[x,y],...], text, conf )
            parsed = [(str(r[1]), float(r[2])) for r in results if float(r[2]) >= MIN_CONFIDENCE]
            parsed.sort(key=lambda x: x[1], reverse=True)
            return parsed
        except Exception as exc:
            logger.warning("EasyOCR readtext raised: %s", exc)
            return []

    # ── Text cleaning ─────────────────────────────────────────────────────────
    def clean_text(self, text: str) -> str:
        """Sanitise raw OCR output into a normalised plate string.

        Steps
        -----
        1. Uppercase.
        2. Strip non-alphanumeric characters (spaces, dashes, special chars).
        3. Collapse multiple spaces.

        Parameters
        ----------
        text : str

        Returns
        -------
        str — cleaned uppercase alphanumeric string.
        """
        if not text:
            return ""
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text.strip()

    # ── Single vehicle ────────────────────────────────────────────────────────
    def process_vehicle(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> PlateResult:
        """Full ANPR pipeline for one vehicle bounding box.

        1. Extract plate ROI from ``frame`` using ``bbox``.
        2. Preprocess the ROI.
        3. Run OCR.
        4. Clean and validate results.
        5. Return ``PlateResult``.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame.
        bbox : (x1, y1, x2, y2)
            Vehicle bounding box.

        Returns
        -------
        PlateResult
            Always returned; check ``.valid`` to test whether a plate was found.
        """
        result = PlateResult(bbox=tuple(bbox))

        # 1. Extract ROI
        roi, roi_bbox = self.extract_plate(frame, bbox)
        result.roi_bbox = roi_bbox

        if roi is None:
            logger.debug("No usable ROI for bbox %s", bbox)
            return result

        # 2. Preprocess
        processed = self.preprocess_plate(roi)

        # 3. OCR
        t0 = time.perf_counter()
        ocr_results = self.read_plate(processed)
        result.inference_ms = (time.perf_counter() - t0) * 1000

        self._total_ocr_ms += result.inference_ms
        self._plates_read += 1

        # 4. Collect raw texts
        result.raw_texts = [t for t, _ in ocr_results]

        if not ocr_results:
            logger.debug("No OCR output for bbox %s", bbox)
            return result

        # 5. Pick best candidate
        best_text, best_conf = ocr_results[0]

        # If top result is short, try concatenating all results
        if len(self.clean_text(best_text)) < MIN_PLATE_CHARS and len(ocr_results) > 1:
            combined = "".join(t for t, _ in ocr_results)
            combined_conf = sum(c for _, c in ocr_results) / len(ocr_results)
            if len(self.clean_text(combined)) >= len(self.clean_text(best_text)):
                best_text, best_conf = combined, combined_conf

        cleaned = self.clean_text(best_text)
        result.plate_text  = cleaned
        result.confidence  = round(best_conf, 4)
        result.valid       = len(cleaned) >= MIN_PLATE_CHARS

        if result.valid:
            self._plates_valid += 1
            logger.info(
                "Plate detected | text='%s' | conf=%.2f | %.0f ms | bbox=%s",
                cleaned, best_conf, result.inference_ms, bbox,
            )
        else:
            logger.debug(
                "Plate rejected (len=%d) | raw='%s' | conf=%.2f",
                len(cleaned), cleaned, best_conf,
            )

        return result

    # ── Batch processing ─────────────────────────────────────────────────────
    def process_batch(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict],
    ) -> Dict[int, PlateResult]:
        """Run ANPR on every vehicle in a list of tracked objects.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame shared by all tracked objects.
        tracked_objects : list of dict
            Each dict must contain:
            - ``track_id`` (int)
            - ``bbox``     ((x1, y1, x2, y2))
            - ``class_id`` (int)  — only VEHICLE_CLASS_IDS are processed

        Returns
        -------
        dict[track_id -> PlateResult]
            Only entries whose class_id is a vehicle class are included.
        """
        from config.settings import VEHICLE_CLASS_IDS  # avoid module-level circular risk

        results: Dict[int, PlateResult] = {}

        for obj in tracked_objects:
            class_id = obj.get("class_id", -1)
            if class_id not in VEHICLE_CLASS_IDS:
                continue

            track_id = obj.get("track_id", -1)
            bbox     = obj.get("bbox")

            if bbox is None:
                logger.warning("track_id=%d has no bbox — skipped", track_id)
                continue

            result = self.process_vehicle(frame, bbox)
            results[track_id] = result

        logger.info(
            "Batch complete | %d vehicles | %d valid plates",
            len(results),
            sum(1 for r in results.values() if r.valid),
        )
        return results

    # ── Visualisation ────────────────────────────────────────────────────────
    def draw_plate_text(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        plate_result: PlateResult,
        *,
        draw_roi: bool = True,
    ) -> np.ndarray:
        """Annotate a frame with the detected plate text and ROI rectangle.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to annotate (modified in-place AND returned).
        bbox : (x1, y1, x2, y2)
            Vehicle bounding box — used to position the label.
        plate_result : PlateResult
            Result from ``process_vehicle`` or ``process_batch``.
        draw_roi : bool
            Whether to also draw the plate ROI rectangle (default True).

        Returns
        -------
        np.ndarray
            Annotated frame (same object as input).
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Vehicle box colour: green = valid plate, yellow = no plate
        box_colour  = (0, 220, 0) if plate_result.valid else (0, 200, 220)
        label_colour = (0, 220, 0) if plate_result.valid else (80, 80, 80)

        # Draw vehicle bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)

        # Compose label
        if plate_result.valid:
            label = f"{plate_result.plate_text}  {plate_result.confidence:.0%}"
        else:
            label = "NO PLATE"

        # Label background pill
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.65, 2)
        pad      = 6
        label_y  = max(y1 - 10, th + pad * 2)

        cv2.rectangle(
            frame,
            (x1, label_y - th - pad),
            (x1 + tw + pad * 2, label_y + baseline),
            label_colour,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + pad, label_y),
            FONT,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Draw plate ROI rectangle
        if draw_roi and plate_result.roi_bbox:
            rx1, ry1, rx2, ry2 = [int(v) for v in plate_result.roi_bbox]
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 120, 255), 2)
            cv2.putText(
                frame,
                "PLATE ROI",
                (rx1, ry1 - 5),
                FONT,
                0.45,
                (0, 120, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    # ── Stats ────────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        """Return cumulative runtime statistics.

        Returns
        -------
        dict with keys: plates_read, plates_valid, hit_rate, avg_ocr_ms
        """
        hit_rate   = self._plates_valid / self._plates_read if self._plates_read else 0.0
        avg_ocr_ms = self._total_ocr_ms / self._plates_read if self._plates_read else 0.0
        return {
            "plates_read":  self._plates_read,
            "plates_valid": self._plates_valid,
            "hit_rate":     round(hit_rate, 3),
            "avg_ocr_ms":   round(avg_ocr_ms, 1),
        }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  ANPRSystem -- self-test")
    print("=" * 60)

    # ── 1. Check EasyOCR import ───────────────────────────────────────────
    try:
        import easyocr
        print("[OK] easyocr imported")
    except ImportError:
        print("[FAIL] easyocr not found.  Run: pip install easyocr")
        sys.exit(1)

    # ── 2. Instantiate (loads model weights) ─────────────────────────────
    anpr = ANPRSystem(gpu=True)

    # ── 3. Build a synthetic test frame with a fake plate ─────────────────
    #       We render white text on a dark rectangle so the pipeline has
    #       something to read even without real CCTV footage.
    FRAME_W, FRAME_H = 1280, 720
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)

    # Simulated vehicle bbox (large grey rectangle)
    vx1, vy1, vx2, vy2 = 300, 150, 900, 580
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (80, 80, 80), -1)   # vehicle body
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (200, 200, 200), 3) # outline

    # Simulated plate inside the bbox (bottom-centre area)
    px1, py1, px2, py2 = 430, 490, 780, 550
    cv2.rectangle(frame, (px1, py1), (px2, py2), (230, 230, 230), -1)   # plate bg
    cv2.rectangle(frame, (px1, py1), (px2, py2), (20, 20, 20), 2)       # border
    cv2.putText(
        frame, "MH12AB1234", (px1 + 15, py2 - 12),
        FONT, 1.2, (10, 10, 10), 3, cv2.LINE_AA,
    )

    # Optional: try loading a real CCTV frame
    try:
        video_path = settings.get_first_video()
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, real_frame = cap.read()
            if ret and real_frame is not None:
                frame = real_frame
                print(f"[OK] Loaded real frame from: {video_path}")
            cap.release()
    except Exception:
        print("[INFO] Using synthetic test frame (no video found)")

    # ── 4. Simulate vehicle bboxes ────────────────────────────────────────
    simulated_vehicles = [
        {"track_id": 1, "class_id": 2, "bbox": (vx1, vy1, vx2, vy2)},   # car
        {"track_id": 2, "class_id": 2, "bbox": (50, 50, 250, 250)},      # small/occluded
    ]

    print(f"\nProcessing {len(simulated_vehicles)} vehicle(s)...")

    # ── 5. Batch ANPR ─────────────────────────────────────────────────────
    batch_results = anpr.process_batch(frame, simulated_vehicles)

    # ── 6. Print results ──────────────────────────────────────────────────
    print("\n--- ANPR Results ---")
    for track_id, res in batch_results.items():
        print(
            f"  Track {track_id:>3} | plate_number={res.plate_number!r:>14} "
            f"| plate_confidence={res.plate_confidence:.3f} "
            f"| valid={res.valid} "
            f"| ocr_ms={res.inference_ms:.1f} "
            f"| raw={res.raw_texts}"
        )

    # ── 7. Draw annotations ───────────────────────────────────────────────
    annotated = frame.copy()
    for obj in simulated_vehicles:
        tid  = obj["track_id"]
        bbox = obj["bbox"]
        if tid in batch_results:
            annotated = anpr.draw_plate_text(annotated, bbox, batch_results[tid])

    # ── 8. Save output ────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/anpr_test.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\n[OK] Annotated frame saved to '{out_path}'")

    # ── 9. Stats ──────────────────────────────────────────────────────────
    stats = anpr.get_stats()
    print("\n--- Runtime Stats ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nANPR self-test complete.")
