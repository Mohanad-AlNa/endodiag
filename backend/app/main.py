from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .model_service import ModelService
from .schemas import HealthResponse, PredictionResponse

settings = get_settings()
model_service = ModelService(settings)

app = FastAPI(title="EndoDiag API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.loaded_model.loaded,
        model_path=str(settings.model_path),
        model_name=model_service.loaded_model.model_name,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    if not model_service.loaded_model.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model weights are not loaded. Train a model and set ENDODIAG_MODEL_PATH.",
        )

    try:
        image_w, image_h, findings = model_service.predict(content)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    if findings:
        summary = f"تم رصد {len(findings)} منطقة مشتبهة بواسطة النموذج."
    else:
        summary = "لم يرصد النموذج آفات ضمن إعدادات الثقة الحالية."

    return PredictionResponse(
        model_loaded=True,
        model_name=model_service.loaded_model.model_name,
        image_width=image_w,
        image_height=image_h,
        findings=findings,
        summary=summary,
        uncertainty_note="النتائج آلية وتحتاج تأكيدًا سريريًا وشعاعيًا قبل القرار العلاجي.",
    )
