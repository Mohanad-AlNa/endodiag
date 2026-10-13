from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    max_detections: int


def get_settings() -> Settings:
    model_path = Path(os.getenv("ENDODIAG_MODEL_PATH", "../models/endodiag-seg.pt")).resolve()
    image_size = int(os.getenv("ENDODIAG_IMAGE_SIZE", "1024"))
    confidence_threshold = float(os.getenv("ENDODIAG_CONF", "0.25"))
    iou_threshold = float(os.getenv("ENDODIAG_IOU", "0.5"))
    max_detections = int(os.getenv("ENDODIAG_MAX_DET", "30"))

    return Settings(
        model_path=model_path,
        image_size=image_size,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
