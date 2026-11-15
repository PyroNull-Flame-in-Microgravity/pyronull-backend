from fastapi import APIRouter, HTTPException

from app.models.additional_predictions import (
    AdditionalClassificationResponse,
    Psi107FuelClassificationRequest,
    Psi69FlameOutcomeRequest,
    Psi10InverseFlameRequest,
    Psi10RegressionPredictionResponse,
    Psi159IgnitionPredictionResponse,
    Psi159IgnitionRequest,
)
from app.models.burn_time_prediction import BurnTimePredictionRequest, BurnTimePredictionResponse
from app.services.additional_predictions import (
    PredictionInferenceError,
    PredictionModelUnavailable,
    predict_psi107_fuel,
    predict_psi69_flame_outcome,
    predict_psi10_pressure_rise,
    predict_psi10_tad,
    predict_psi10_zst,
    predict_psi159_ignition,
)
from app.services.burn_time_prediction import (
    BurnTimeModelUnavailable,
    BurnTimePredictionError,
    build_prediction_response,
)


router = APIRouter()


def _handle_additional_prediction(function, payload):
    try:
        return function(payload)
    except PredictionModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PredictionInferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/predictions/psi10/adiabatic-flame-temperature",
    response_model=Psi10RegressionPredictionResponse,
)
def predict_psi10_adiabatic_flame_temperature(payload: Psi10InverseFlameRequest):
    return _handle_additional_prediction(predict_psi10_tad, payload)


@router.post(
    "/predictions/psi10/stoichiometric-mixture-fraction",
    response_model=Psi10RegressionPredictionResponse,
)
def predict_psi10_stoichiometric_mixture_fraction(payload: Psi10InverseFlameRequest):
    return _handle_additional_prediction(predict_psi10_zst, payload)


@router.post(
    "/predictions/psi10/pressure-rise",
    response_model=Psi10RegressionPredictionResponse,
)
def predict_psi10_pressure_rise_endpoint(payload: Psi10InverseFlameRequest):
    return _handle_additional_prediction(predict_psi10_pressure_rise, payload)


@router.post(
    "/predictions/psi159/ignition",
    response_model=Psi159IgnitionPredictionResponse,
)
def predict_psi159_ignition_endpoint(payload: Psi159IgnitionRequest):
    return _handle_additional_prediction(predict_psi159_ignition, payload)


@router.post(
    "/predictions/psi107/fuel-classification",
    response_model=AdditionalClassificationResponse,
)
def predict_psi107_fuel_endpoint(payload: Psi107FuelClassificationRequest):
    return _handle_additional_prediction(predict_psi107_fuel, payload)


@router.post(
    "/predictions/psi69/flame-outcome",
    response_model=AdditionalClassificationResponse,
)
def predict_psi69_flame_outcome_endpoint(payload: Psi69FlameOutcomeRequest):
    return _handle_additional_prediction(predict_psi69_flame_outcome, payload)


@router.post("/predictions/burn-time", response_model=BurnTimePredictionResponse)
def predict_burn_time(payload: BurnTimePredictionRequest) -> BurnTimePredictionResponse:
    try:
        return build_prediction_response(payload)
    except BurnTimeModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BurnTimePredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
