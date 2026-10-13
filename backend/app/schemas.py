from __future__ import annotations

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class Finding(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    severity: str
    polygon: list[Point]
    bbox_xyxy: list[float]
    area_px: float
    centroid: Point
    recommendation: str


class PredictionResponse(BaseModel):
    model_loaded: bool
    model_name: str
    image_width: int
    image_height: int
    findings: list[Finding]
    summary: str
    uncertainty_note: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_name: str
