"""DeepSentinel REST API."""

import base64
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .inference import DeepSentinel

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
detector: DeepSentinel | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector
    detector = DeepSentinel()
    yield


app = FastAPI(
    title="DeepSentinel API",
    description="Deepfake face detection with EfficientNet-B4 and Grad-CAM.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    label: str
    p_fake: float
    face_detected: bool
    face_box: list[int] | None
    heatmap_png_base64: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image containing a face (JPEG or PNG)"),
    heatmap: bool = Query(False, description="Include a Grad-CAM overlay as base64 PNG"),
):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    result = detector.predict(image, with_heatmap=heatmap)

    encoded = None
    if result.heatmap is not None:
        buf = io.BytesIO()
        result.heatmap.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()

    return PredictionResponse(
        label=result.label,
        p_fake=result.p_fake,
        face_detected=result.face_detected,
        face_box=result.face_box,
        heatmap_png_base64=encoded,
    )
