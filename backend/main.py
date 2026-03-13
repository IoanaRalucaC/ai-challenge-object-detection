"""
Advanced Object Detector — FastAPI Application
Wraps the YOLOv8 + SAHI dual-pipeline inference in a production-ready REST API.

Endpoint:
    POST /detect   — Accepts a multipart image upload; returns JSON detections.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


# ─── Target Classes ───────────────────────────────────────────────────────────────
TARGETS = {0: "Person", 15: "Cat", 16: "Dog"}

BOX_COLORS = {
    "Person": (0, 255, 0),
    "Dog":    (0, 0, 255),
    "Cat":    (255, 0, 0),
}


# ─── Model Configuration ─────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8m.pt")


# ─── Pipeline 1 Parameters ───────────────────────────────────────────────────────
IMGSZ = int(os.getenv("IMGSZ", 1280))
AUGMENT = True
AGNOSTIC_NMS = True
IOU_THRESHOLD = 0.45


# ─── Pipeline 2 Parameters ───────────────────────────────────────────────────────
SLICE_HEIGHT = int(os.getenv("SLICE_HEIGHT", 640))
SLICE_WIDTH  = int(os.getenv("SLICE_WIDTH",  640))
SLICE_OVERLAP_RATIO = float(os.getenv("SLICE_OVERLAP_RATIO", 0.2))


# ─── Class-Specific Confidence Thresholds ────────────────────────────────────────
CLASS_CONF_THRESHOLDS = {
    0:  0.55,   # Person
    15: 0.35,   # Cat
    16: 0.35,   # Dog
}

MODEL_CONF_FLOOR = min(CLASS_CONF_THRESHOLDS.values())  # 0.35
MERGE_IOU_THRESHOLD = 0.45


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────────

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[int]  # [x_min, y_min, x_max, y_max]


class DetectionResponse(BaseModel):
    count: int
    detections: list[Detection]


# ─── Application Lifespan (model loaded once at startup) ─────────────────────────

# Holds the YOLO model instance after startup so it is never reloaded per-request.
_yolo_model: YOLO | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the YOLO model on startup and release it on shutdown."""
    global _yolo_model
    _yolo_model = YOLO(MODEL_PATH)
    yield
    _yolo_model = None


# ─── FastAPI App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Object Detector API",
    description="Detects Dog, Cat, and Person using a YOLOv8 + SAHI dual pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Detection Helpers ────────────────────────────────────────────────────────────

def _run_standard_inference(image: np.ndarray) -> list[dict]:
    """Pipeline 1 — full-image, high-resolution YOLOv8 inference with TTA."""
    results = _yolo_model.predict(
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


def _run_sahi_inference(image: np.ndarray) -> list[dict]:
    """Pipeline 2 — SAHI sliced inference over 640×640 patches."""
    model_path = str(Path(MODEL_PATH).resolve())
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=MODEL_CONF_FLOOR,
        device="cpu",
    )
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


def _compute_iou(box_a: tuple, box_b: tuple) -> float:
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


def _merge_detections(standard: list[dict], sahi: list[dict]) -> list[dict]:
    """Greedy NMS merge across both pipelines, highest confidence first."""
    all_candidates = sorted(
        standard + sahi, key=lambda d: d["confidence"], reverse=True
    )
    kept = []
    for candidate in all_candidates:
        if not any(
            _compute_iou(candidate["bbox"], k["bbox"]) > MERGE_IOU_THRESHOLD
            for k in kept
        ):
            kept.append(candidate)
    return kept


def _apply_class_thresholds(detections: list[dict]) -> list[dict]:
    return [
        det for det in detections
        if det["confidence"] >= CLASS_CONF_THRESHOLDS[det["class_id"]]
    ]


# ─── Endpoint ─────────────────────────────────────────────────────────────────────

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)) -> DetectionResponse:
    """
    Detect Dogs, Cats, and Persons in an uploaded image.

    - **file**: Any common image format (JPEG, PNG, BMP, …).

    Returns a JSON object with `count` and a `detections` array where each entry
    contains `label`, `confidence` (0–1), and `bbox` as `[x_min, y_min, x_max, y_max]`.
    """
    # ── Decode the uploaded bytes into an OpenCV image ──────────────────────────
    raw = await file.read()
    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be decoded as a valid image. "
                   "Supported formats: JPEG, PNG, BMP, TIFF, WebP.",
        )

    # ── Run dual-pipeline inference ─────────────────────────────────────────────
    standard = _run_standard_inference(image)
    sahi     = _run_sahi_inference(image)
    merged   = _merge_detections(standard, sahi)
    final    = sorted(
        _apply_class_thresholds(merged),
        key=lambda d: d["confidence"],
        reverse=True,
    )

    # ── Build response ───────────────────────────────────────────────────────────
    detections = [
        Detection(
            label=det["label"],
            confidence=round(det["confidence"], 4),
            bbox=list(det["bbox"]),
        )
        for det in final
    ]
    return DetectionResponse(count=len(detections), detections=detections)
