"""
Advanced Object Detector — YOLOv8 + SAHI inference pipeline - standalone script.
Detects Dog, Cat, and Person with maximized accuracy using two complementary pipelines:

  Pipeline 1 — Full-image, high-resolution YOLOv8 inference with Test Time Augmentation.
               Best for normal-to-large objects visible in the full frame.

  Pipeline 2 — SAHI sliced inference (640×640 patches, 20% overlap).
               Best for small or partially-cut-off objects that are missed in Pipeline 1.

Both pipelines run with a low confidence floor so no real detections are discarded early.
After merging, class-specific thresholds are applied:
  • Person (0)  >= 0.55  — strict, reduces false positives (e.g. sofa texture → person)
  • Cat   (15)  >= 0.35  — lenient, catches partially visible or smaller cats
  • Dog   (16)  >= 0.35  — lenient, catches partially visible or smaller dogs

Usage:
    python detect.py --input <local_path>
    python detect.py --input <https://...url...>
    python detect.py --input camera
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# ─── Target Classes ───────────────────────────────────────────────────────────────
# COCO class IDs for the three objects we care about
TARGETS = {0: "Person", 15: "Cat", 16: "Dog"}

# Bounding box colors per label — OpenCV uses BGR order (not RGB)
BOX_COLORS = {
    "Person": (0, 255, 0),   # green
    "Dog":    (0, 0, 255),   # red
    "Cat":    (255, 0, 0),   # blue
}


# ─── Model Configuration ─────────────────────────────────────────────────────────
# YOLOv8 medium — best balance of accuracy and speed for this use case.
# Downloads automatically (~50 MB) on first run.
MODEL_PATH = "yolov8m.pt"


# ─── Pipeline 1 Parameters (High-Res YOLOv8) ─────────────────────────────────────
# imgsz=1280 — process at high resolution to preserve detail in crowded scenes
IMGSZ = 1280

# augment=True — Test Time Augmentation: runs multiple flipped/scaled versions
# of the image and merges the results, reducing false negatives
AUGMENT = True

# agnostic_nms=True — class-agnostic NMS prevents the same physical object from
# being kept as two different classes (e.g. both "Person" and "Cat" for one body)
AGNOSTIC_NMS = True

# iou=0.45 — NMS IoU threshold: suppress overlapping boxes above this overlap ratio
IOU_THRESHOLD = 0.45


# ─── Pipeline 2 Parameters (SAHI Sliced Inference) ───────────────────────────────
# Slice the image into 640×640 patches to catch objects that appear small at full scale
SLICE_HEIGHT = 640
SLICE_WIDTH = 640

# 20% overlap between adjacent slices ensures objects near slice borders are not cut off
SLICE_OVERLAP_RATIO = 0.2


# ─── Class-Specific Confidence Thresholds ────────────────────────────────────────
# These are the FINAL gates applied after both pipelines are merged.
# Using separate values per class lets us control precision vs. recall per target.
CLASS_CONF_THRESHOLDS = {
    0:  0.55,   # Person — higher bar to suppress false positives
    15: 0.35,   # Cat    — lower bar to catch subtle or small instances
    16: 0.35,   # Dog    — lower bar to catch subtle or small instances
}

# The floor is passed to YOLO / SAHI so candidates below every possible threshold
# are still surfaced for the post-merge filter to evaluate.
MODEL_CONF_FLOOR = min(CLASS_CONF_THRESHOLDS.values())  # 0.35

# IoU threshold used in our custom merge NMS (deduplicates across the two pipelines)
MERGE_IOU_THRESHOLD = 0.45


# ─── Image Loading ────────────────────────────────────────────────────────────────

def load_image(source: str) -> np.ndarray:
    """
    Load an image from a local file path, an HTTP/S URL, or the webcam.

    Returns a BGR numpy array (as returned by OpenCV).
    Raises SystemExit with a descriptive message on failure.
    """
    source = source.strip()

    # --- Webcam ---
    if source.lower() == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("Error: could not open webcam.")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            sys.exit("Error: could not capture a frame from the webcam.")
        return frame

    # --- URL ---
    if source.lower().startswith("http://") or source.lower().startswith("https://"):
        try:
            with urllib.request.urlopen(source, timeout=10) as response:
                image_bytes = response.read()
        except Exception as exc:
            sys.exit(f"Error: could not download image from URL.\n{exc}")

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            sys.exit("Error: downloaded data could not be decoded as an image.")
        return frame

    # --- Local file ---
    frame = cv2.imread(source)
    if frame is None:
        sys.exit(
            f"Error: could not read image file '{source}'. "
            "Check that the path is correct and the file is a valid image."
        )
    return frame


# ─── Pipeline 1: High-Res YOLOv8 Inference ───────────────────────────────────────

def run_standard_inference(image: np.ndarray, model: YOLO) -> list[dict]:
    """
    Full-image, high-resolution YOLOv8 inference with Test Time Augmentation.

    Parameters used:
        imgsz=1280        — high resolution for detail preservation
        augment=True      — TTA to reduce false negatives
        agnostic_nms=True — prevents one object being double-labelled
        iou=0.45          — NMS overlap threshold
        conf=floor        — low floor so post-merge filter makes the final call

    Returns a list of raw candidate detections (not yet threshold-filtered).
    """
    results = model.predict(
        image,
        conf=MODEL_CONF_FLOOR,
        imgsz=IMGSZ,
        augment=AUGMENT,
        agnostic_nms=AGNOSTIC_NMS,
        iou=IOU_THRESHOLD,
        verbose=False,
    )

    candidates = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id in TARGETS:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            candidates.append({
                "label":      TARGETS[class_id],
                "class_id":   class_id,
                "confidence": float(box.conf[0]),
                "bbox":       (x1, y1, x2, y2),
                "source":     "standard",
            })

    return candidates


# ─── Pipeline 2: SAHI Sliced Inference ───────────────────────────────────────────

def run_sahi_inference(image: np.ndarray) -> list[dict]:
    """
    SAHI sliced inference — splits the image into 640×640 patches with 20% overlap,
    runs YOLOv8 on each patch, then merges patch-level detections using SAHI's NMS.

    This pipeline is specifically designed to catch objects that appear small
    or near the edges of the full-resolution image.

    Returns a list of raw candidate detections (not yet threshold-filtered).
    """
    # Resolve absolute path so SAHI can find the model regardless of cwd
    model_path = str(Path(MODEL_PATH).resolve())

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        # Use the floor so SAHI doesn't suppress candidates we still want to evaluate
        confidence_threshold=MODEL_CONF_FLOOR,
        device="cpu",
    )

    # SAHI expects a PIL image in RGB
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    sahi_result = get_sliced_prediction(
        image_pil,
        sahi_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=SLICE_OVERLAP_RATIO,
        overlap_width_ratio=SLICE_OVERLAP_RATIO,
        verbose=0,
    )

    candidates = []
    for pred in sahi_result.object_prediction_list:
        class_id = pred.category.id
        if class_id in TARGETS:
            b = pred.bbox
            candidates.append({
                "label":      TARGETS[class_id],
                "class_id":   class_id,
                "confidence": float(pred.score.value),
                "bbox":       (int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)),
                "source":     "sahi",
            })

    return candidates


# ─── Merging & Filtering ──────────────────────────────────────────────────────────

def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Both boxes are in (x1, y1, x2, y2) format.
    Returns a float in [0, 1].
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    if intersection == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def merge_detections(standard: list[dict], sahi: list[dict]) -> list[dict]:
    """
    Merge detections from both pipelines and deduplicate using greedy NMS.

    Strategy:
      1. Pool all candidates from both pipelines together.
      2. Sort by confidence, highest first.
      3. Greedy NMS: keep a candidate only if it does not overlap
         any already-kept detection by more than MERGE_IOU_THRESHOLD.
         This handles cases where both pipelines find the same object.
    """
    all_candidates = standard + sahi
    all_candidates.sort(key=lambda d: d["confidence"], reverse=True)

    kept = []
    for candidate in all_candidates:
        overlaps_existing = any(
            compute_iou(candidate["bbox"], kept_det["bbox"]) > MERGE_IOU_THRESHOLD
            for kept_det in kept
        )
        if not overlaps_existing:
            kept.append(candidate)

    return kept


def apply_class_thresholds(detections: list[dict]) -> list[dict]:
    """
    Apply class-specific confidence thresholds as the final filter.

      Person (0)  >= 0.55  — eliminates false positives
      Cat    (15) >= 0.35  — keeps low-confidence but real cat detections
      Dog    (16) >= 0.35  — keeps low-confidence but real dog detections

    Returns only the detections that pass their class threshold.
    """
    return [
        det for det in detections
        if det["confidence"] >= CLASS_CONF_THRESHOLDS[det["class_id"]]
    ]


# ─── Visualization ────────────────────────────────────────────────────────────────

def draw_boxes(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw color-coded bounding boxes and confidence labels on a copy of the image.

    Color scheme:
        Person → green   (0, 255, 0)
        Dog    → red     (0, 0, 255)
        Cat    → blue    (255, 0, 0)

    Each box has a filled label background above it for readability.
    Returns the annotated image; the original is not modified.
    """
    annotated  = image.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color      = BOX_COLORS[det["label"]]
        label_text = f"{det['label']} {det['confidence'] * 100:.1f}%"

        # Bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Filled label background — sits just above the top edge of the box
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        bg_y1 = max(y1 - text_h - baseline - 4, 0)
        bg_y2 = y1
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w, bg_y2), color, cv2.FILLED)

        # White label text for contrast against the filled background
        cv2.putText(
            annotated, label_text,
            (x1, bg_y2 - baseline // 2 - 1),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return annotated


# ─── Console Output ───────────────────────────────────────────────────────────────

def print_results(detections: list[dict]) -> None:
    """Print the final detection results to stdout in a readable format."""
    if not detections:
        print("\nNo target detected.\n")
        return

    print(f"\nDetected {len(detections)} target(s):\n")
    for det in detections:
        source_tag = f"[{det['source']}]"
        print(f"  {det['label']:<10}  {det['confidence'] * 100:.1f}%  {source_tag}")
    print()


# ─── Entry Point ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect Dogs, Cats, and Persons using a dual YOLOv8 + SAHI pipeline."
        )
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help=(
            "Image source. Can be:\n"
            "  • A local file path  (e.g. dog.jpg)\n"
            "  • An HTTP/S URL      (e.g. https://example.com/photo.jpg)\n"
            "  • The word 'camera'  to capture from the webcam"
        ),
    )
    args = parser.parse_args()

    # Step 1 — Load image
    print(f"\nLoading image from: {args.input}")
    image = load_image(args.input)

    # Step 2 — Load YOLO model once (ensures yolov8m.pt is downloaded before SAHI needs it)
    model = YOLO(MODEL_PATH)

    # Step 3 — Pipeline 1: high-res full-image inference
    print("Pipeline 1: High-res YOLOv8 inference (imgsz=1280, TTA, agnostic NMS)...")
    standard_detections = run_standard_inference(image, model)
    print(f"           → {len(standard_detections)} raw candidate(s)")

    # Step 4 — Pipeline 2: SAHI sliced inference
    print("Pipeline 2: SAHI sliced inference (640×640 slices, 20% overlap)...")
    sahi_detections = run_sahi_inference(image)
    print(f"           → {len(sahi_detections)} raw candidate(s)")

    # Step 5 — Merge both pipelines and apply class-specific thresholds
    print("Merging pipelines and applying class-specific confidence thresholds...")
    merged = merge_detections(standard_detections, sahi_detections)
    final  = apply_class_thresholds(merged)
    final.sort(key=lambda d: d["confidence"], reverse=True)

    # Step 6 — Print results
    print_results(final)

    # Step 7 — Save annotated image
    if final:
        annotated = draw_boxes(image, final)
        cv2.imwrite("output.jpg", annotated)
        print("Annotated image saved to: output.jpg\n")


if __name__ == "__main__":
    main()
