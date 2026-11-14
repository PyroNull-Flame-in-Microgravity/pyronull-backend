from fastapi import APIRouter, HTTPException

from app.models.burn_time_prediction import BurnTimePredictionRequest, BurnTimePredictionResponse
from app.services.burn_time_prediction import (
    BurnTimeModelUnavailable,
    BurnTimePredictionError,
    build_prediction_response,
)


router = APIRouter()


@router.post("/predictions/burn-time", response_model=BurnTimePredictionResponse)
def predict_burn_time(payload: BurnTimePredictionRequest) -> BurnTimePredictionResponse:
    try:
        return build_prediction_response(payload)
    except BurnTimeModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BurnTimePredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
