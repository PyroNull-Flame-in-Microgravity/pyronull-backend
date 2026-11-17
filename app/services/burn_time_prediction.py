"""Inference service for the PSI-159 experimental burn-time model."""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd

from app.models.burn_time_prediction import BurnTimePredictionRequest


MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "PSI_159_burntime_regressor_Predictive_Model.joblib"
)
MODEL_SHA256 = "fa63f122f984f221df53e65c778e3d2d7a3a90b2a50a28ecf066e776967a3309"
MODEL_NAME = "PSI-159 Burn Time Regressor"
MODEL_TYPE = "Scikit-learn Random Forest regression pipeline"
MODEL_FEATURES: Tuple[str, ...] = (
    "Flame Type",
    "Fuel",
    "approx p (bar)",
    "Tad (K)",
    "HC burnrate (mg/s)",
    "O2 flow rate (mg/s)",
    "XO2",
    "Xf",
)
EVALUATION_METRICS = {
    "mae_seconds": 23.08,
    "rmse_seconds": 36.14,
    "r_squared": 0.1575,
}
MODEL_LIMITATIONS = [
    "Documented test-set MAE is approximately 23.08 seconds and RMSE is approximately 36.14 seconds; individual prediction errors may be larger.",
    "Documented R-squared is approximately 0.1575, indicating limited explanatory performance on the reported evaluation set.",
    "Evaluation metrics are reported from the accompanying model documentation and have not been independently reproduced by this API.",
    "Predictions depend on the serialized model artifact and its training data; no uncertainty interval is produced.",
]
EXPERIMENTAL_DOMAIN_WARNING = (
    "This is an experimental burn-time estimate for inputs resembling the PSI-159 model's training domain. "
    "It is not a fire-risk prediction, fire probability, suppression-time prediction, or habitat safety assessment. "
    "It is not validated for lunar, Martian, spacecraft, or other environments outside the training domain."
)


class BurnTimeModelUnavailable(RuntimeError):
    """The configured estimator artifact cannot be loaded."""


class BurnTimePredictionError(RuntimeError):
    """The estimator failed or returned an unusable prediction."""


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the local artifact once per process and reuse the estimator."""
    try:
        if not MODEL_PATH.is_file():
            raise BurnTimeModelUnavailable("The PSI-159 burn-time model artifact is not installed.")
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            raise BurnTimeModelUnavailable("The PSI-159 burn-time model artifact failed its integrity check.")
        model = joblib.load(MODEL_PATH)
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None and tuple(model_features) != MODEL_FEATURES:
            raise BurnTimeModelUnavailable("The PSI-159 model artifact has an incompatible feature schema.")
        return model
    except BurnTimeModelUnavailable:
        raise
    except Exception as exc:
        raise BurnTimeModelUnavailable("The PSI-159 burn-time model could not be loaded.") from exc


def _feature_frame(request: BurnTimePredictionRequest) -> pd.DataFrame:
    values = [
        request.flame_type,
        request.fuel,
        request.pressure_bar,
        request.adiabatic_flame_temperature_k,
        request.hydrocarbon_burn_rate_mg_s,
        request.oxygen_flow_rate_mg_s,
        request.oxygen_mole_fraction,
        request.fuel_fraction,
    ]
    return pd.DataFrame([values], columns=MODEL_FEATURES)


def predict_burn_time(request: BurnTimePredictionRequest) -> float:
    try:
        prediction = load_model().predict(_feature_frame(request))
        value = float(prediction[0])
    except BurnTimeModelUnavailable:
        raise
    except Exception as exc:
        raise BurnTimePredictionError("The PSI-159 model could not produce a prediction for these inputs.") from exc
    if not np.isfinite(value):
        raise BurnTimePredictionError("The PSI-159 model returned a non-finite prediction.")
    return value


def build_prediction_response(request: BurnTimePredictionRequest) -> dict:
    return {
        "predicted_burn_time_seconds": predict_burn_time(request),
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "evaluation_metrics": EVALUATION_METRICS,
        "model_limitations": MODEL_LIMITATIONS,
        "experimental_domain_warning": EXPERIMENTAL_DOMAIN_WARNING,
    }
