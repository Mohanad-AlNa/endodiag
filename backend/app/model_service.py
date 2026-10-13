from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .config import Settings
from .schemas import Finding, Point


CLASS_METADATA = {
    "class_ii_caries": {
        "severity": "warn",
        "recommendation": "مطابقة شعاعية مع Bitewing وفحص سريري دقيق للسطوح القريبة.",
    },
    "apical_lesion": {
        "severity": "alert",
        "recommendation": "ربط مع فحص القرع/الجس/الحيوية وتقييم شعاعي متابع.",
    },
    "pdl_widening": {
        "severity": "warn",
        "recommendation": "تقييم حالة الرباط والاطباق ومتابعة سريرية شعاعية.",
    },
    "resorption": {
        "severity": "alert",
        "recommendation": "تقييم نوع الامتصاص وتخطيط علاجي مختص مع تصوير إضافي.",
    },
}


@dataclass
class LoadedModel:
    model: YOLO | None
    model_name: str
    loaded: bool


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loaded_model = self._load_model()

    def _load_model(self) -> LoadedModel:
        if not self.settings.model_path.exists():
            return LoadedModel(model=None, model_name="unavailable", loaded=False)

        model = YOLO(str(self.settings.model_path))
        model_name = self.settings.model_path.name
        return LoadedModel(model=model, model_name=model_name, loaded=True)

    def predict(self, file_bytes: bytes) -> tuple[int, int, list[Finding]]:
        if not self.loaded_model.loaded or self.loaded_model.model is None:
            raise RuntimeError("Model not loaded. Provide ENDODIAG_MODEL_PATH with trained weights.")

        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        image_np = np.array(image)
        image_h, image_w = image_np.shape[:2]

        results = self.loaded_model.model.predict(
            source=image_np,
            imgsz=self.settings.image_size,
            conf=self.settings.confidence_threshold,
            iou=self.settings.iou_threshold,
            max_det=self.settings.max_detections,
            verbose=False,
        )
        findings = self._to_findings(results[0], image_w, image_h)
        return image_w, image_h, findings

    def _to_findings(self, result: Any, image_w: int, image_h: int) -> list[Finding]:
        names = result.names if hasattr(result, "names") else {}
        boxes = result.boxes
        masks = getattr(result, "masks", None)

        findings: list[Finding] = []
        if boxes is None:
            return findings

        for idx in range(len(boxes)):
            cls_id = int(boxes.cls[idx].item())
            conf = float(boxes.conf[idx].item())
            label = names.get(cls_id, str(cls_id))
            xyxy = boxes.xyxy[idx].tolist()

            polygon_points: list[Point] = []
            area_px = float(max(1.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])))

            if masks is not None and masks.xy is not None and idx < len(masks.xy):
                pts = masks.xy[idx]
                polygon_points = [Point(x=float(p[0]), y=float(p[1])) for p in pts]
                area_px = polygon_area(pts)
            else:
                polygon_points = bbox_polygon(xyxy)

            centroid_x, centroid_y = centroid_of_points(polygon_points)
            meta = CLASS_METADATA.get(label, {"severity": "warn", "recommendation": "يحتاج تقييم سريري مكمل."})

            findings.append(
                Finding(
                    label=label,
                    confidence=conf,
                    severity=meta["severity"],
                    polygon=polygon_points,
                    bbox_xyxy=[float(v) for v in xyxy],
                    area_px=float(area_px),
                    centroid=Point(x=centroid_x, y=centroid_y),
                    recommendation=meta["recommendation"],
                )
            )

        findings.sort(key=lambda x: x.confidence, reverse=True)
        return findings


def bbox_polygon(xyxy: list[float]) -> list[Point]:
    x1, y1, x2, y2 = xyxy
    return [
        Point(x=x1, y=y1),
        Point(x=x2, y=y1),
        Point(x=x2, y=y2),
        Point(x=x1, y=y2),
    ]


def centroid_of_points(points: list[Point]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    x = sum(p.x for p in points) / len(points)
    y = sum(p.y for p in points) / len(points)
    return x, y


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 1.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)
