from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
import tempfile
import os

from app.schemas.predict import PredictionAPIResponse, AuthenticityAPIResponse
from app.services.ml_service import PredictionService, get_prediction_service
from app.utils.file_upload import save_uploaded_image

router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=PredictionAPIResponse)
async def predict_endpoint(
    text_report: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """
    Model-Agnostic Prediction Endpoint.
    Supports: Image only, Text only, Image + Text (Multimodal).
    Returns disaster classification, priority score, level, confidence, class probabilities,
    and optional AI image authenticity verification.
    """
    if not image and (not text_report or not text_report.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one input (image file or text report) must be provided for prediction.",
        )

    temp_image_path = None
    try:
        if image:
            # Save uploaded image to temp file for inference
            suffix = os.path.splitext(image.filename)[1] if image.filename else ".jpg"
            contents = await image.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(contents)
                temp_image_path = tmp.name

        # Execute prediction via abstract PredictionService
        result = prediction_service.run_prediction(
            image_input=temp_image_path,
            text_report=text_report,
        )

        return PredictionAPIResponse(
            incident_id=None,
            disaster_type=result["disaster_type"],
            priority_score=result["priority_score"],
            priority_level=result["priority_level"],
            confidence=result["confidence"],
            language_detected=result["language_detected"],
            class_probabilities=result["class_probabilities"],
            authenticity=result.get("authenticity"),
            created_at=datetime.utcnow(),
        )

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)


@router.post("/predict/authenticity", response_model=AuthenticityAPIResponse)
async def predict_authenticity_endpoint(
    image: UploadFile = File(...),
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """
    Standalone AI Image Authenticity Classifier Endpoint.
    Accepts an image upload and returns:
    - AI Generated / Real Photograph classification
    - Confidence score
    - Detailed model information
    """
    temp_image_path = None
    try:
        suffix = os.path.splitext(image.filename)[1] if image.filename else ".jpg"
        contents = await image.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            temp_image_path = tmp.name

        auth_res = prediction_service.run_authenticity_prediction(temp_image_path)

        return AuthenticityAPIResponse(
            classification=auth_res["classification"],
            confidence=auth_res["confidence"],
            ai_probability=auth_res["ai_probability"],
            assessment=auth_res["assessment"],
            is_ai_generated=auth_res["is_ai_generated"],
            model_info=auth_res["model_info"],
            created_at=datetime.utcnow(),
        )
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
