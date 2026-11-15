"""Inference for selected, documented PSI-10 and PSI-159 models.

The local joblib artifacts are UNVERIFIED research artifacts. Only artifacts
whose documented schema and installed serialization dependencies were checked
are registered here.
"""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from app.models.additional_predictions import (
    PredictionEvaluationMetrics,
    PredictionInputValidation,
    PredictionProvenance,
    Psi10InverseFlameRequest,
    Psi10RegressionPredictionResponse,
    Psi159IgnitionPredictionResponse,
    Psi159IgnitionRequest,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_WARNING = (
    "Experimental research-model output only. This is not a certified spacecraft fire-safety assessment, "
    "operational hazard score, or proof of NASA verification. It has not been validated for lunar, Martian, "
    "spacecraft, or other conditions beyond the documented training domain."
)

PSI10_FEATURES: Tuple[str, ...] = (
    "XO2",
    "mO2 (mg/s)",
    "mN2 (mg/s)",
    "mtot (mg/s)",
    "pinit (atm)",
    "pmean (atm)",
)
PSI159_IGNITION_FEATURES: Tuple[str, ...] = (
    "Flame Type",
    "Fuel",
    "approx p (bar)",
    "Tad (K)",
    "HC burnrate (mg/s)",
    "O2 flow rate (mg/s)",
    "XO2",
    "Xf",
)

# Hashes were calculated from the source artifacts before copying them into this
# repository. Refuse to deserialize a missing or changed binary.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "psi10_tad": {
        "path": BACKEND_ROOT / "models/psi10/inverse_flame_tad_predictor.joblib",
        "sha256": "4dc536e250d41afefd2e134519e154e6542a7f4554cd7becb8bece20af473b1a",
        "features": PSI10_FEATURES,
        "name": "PSI-10 Inverse Flame Adiabatic Temperature Predictor",
        "target": "Adiabatic flame temperature (Tad)",
        "unit": "K",
        "metrics": PredictionEvaluationMetrics(
            mae=53.50,
            mae_unit="K",
            r_squared=0.6218,
            documented_sample_count=87,
            documented_test_split_ratio=0.20,
            sample_count_note="Model card reports 87 samples and a 20% test split but does not state exact partition counts; the associated inverse-flame CSV has 120 rows and the difference is unexplained.",
        ),
        "source_dataset": "PSI-10_Experimental table_Flame Design_Inverse Flame.csv",
        "limitations": [
            "Model card documents inverse spherical diffusion flames in microgravity; it is not documented for normal coflow flames, terrestrial high-gravity flames, or complex multiphase fuels.",
            "Model-card sample count (87) does not reconcile with the 120-row supplied CSV; the training-row selection is undocumented.",
            "MAE and R-squared are reported in the supplied model card and have not been independently reproduced.",
            "NASA provenance and the source dataset URL have not been independently verified.",
        ],
    },
    "psi10_zst": {
        "path": BACKEND_ROOT / "models/psi10/inverse_flame_zst_predictor.joblib",
        "sha256": "a0ada663d7aefccd6350a4d04582765542578c1414ae0caf0f2ef8ae7d42b031",
        "features": PSI10_FEATURES,
        "name": "PSI-10 Inverse Flame Stoichiometric Mixture Fraction Predictor",
        "target": "Mean stoichiometric mixture fraction (mean Zst)",
        "unit": "dimensionless fraction (unit implied by target name)",
        "metrics": PredictionEvaluationMetrics(
            mae=0.024039,
            mae_unit="dimensionless fraction",
            r_squared=0.9305,
            documented_sample_count=87,
            documented_test_split_ratio=0.20,
            sample_count_note="Model card reports 87 samples and a 20% test split but does not state exact partition counts; the associated inverse-flame CSV has 120 rows and the difference is unexplained.",
        ),
        "source_dataset": "PSI-10_Experimental table_Flame Design_Inverse Flame.csv",
        "limitations": [
            "The unit is not explicitly specified in the model card; dimensionless is inferred from the mean Zst target name.",
            "Model card reports 87 samples but the associated inverse-flame CSV has 120 rows; training-row selection is undocumented.",
            "MAE and R-squared are reported in the supplied model card and have not been independently reproduced.",
            "NASA provenance and the source dataset URL have not been independently verified.",
        ],
    },
    "psi10_pressure_rise": {
        "path": BACKEND_ROOT / "models/psi10/inverse_flame_pressure_rise_predictor.joblib",
        "sha256": "5e9f667beebf56ef43eb2b7fdf83950a6671f66b4745536ce4d4a5b2ba238c7a",
        "features": PSI10_FEATURES,
        "name": "PSI-10 Inverse Flame Pressure Rise Predictor",
        "target": "Pressure rise (pfinal - pinit)",
        "unit": "atm",
        "metrics": PredictionEvaluationMetrics(
            mae=0.001406,
            mae_unit="atm",
            r_squared=0.6030,
            documented_sample_count=87,
            documented_test_split_ratio=0.20,
            sample_count_note="Model card reports 87 samples and a 20% test split but does not state exact partition counts; the associated inverse-flame CSV has 120 rows and the difference is unexplained.",
        ),
        "source_dataset": "PSI-10_Experimental table_Flame Design_Inverse Flame.csv",
        "limitations": [
            "Model card documents inverse spherical diffusion flames under microgravity; high-pressure explosive, coflow, and non-gaseous fuel systems are outside its stated scope.",
            "Model card reports 87 samples but the associated inverse-flame CSV has 120 rows; training-row selection is undocumented.",
            "MAE and R-squared are reported in the supplied model card and have not been independently reproduced.",
            "This output is not a containment or safety limit. NASA provenance and the source dataset URL have not been independently verified.",
        ],
    },
    "psi159_ignition": {
        "path": BACKEND_ROOT / "models/psi159/PSI_159_ignition_classifier_Predictive_Model.joblib",
        "sha256": "0fcd9acf46d2b8da51882feb9c58df01135455b8dfa2730ec42efc5233e1a224",
        "features": PSI159_IGNITION_FEATURES,
        "classes": (0, 1),
        "name": "PSI-159 Ignition / Flammability Classifier",
        "target": "Hot ignition / flash classification",
        "metrics": PredictionEvaluationMetrics(
            accuracy=0.9048,
            cross_validation_accuracy=0.8888,
            cross_validation_standard_deviation=0.0283,
            sample_count_note="The accompanying README reports test accuracy and five-fold cross-validation but gives no dataset, training, or test sample counts.",
        ),
        "source_dataset": "PSI-159_Experimental table_CFI-G_Inverse Flames.csv and PSI-159_Experimental table_CFI-G_Normal Flames.csv",
        "limitations": [
            "README reports 90.48% test accuracy and 88.88% ± 2.83% five-fold cross-validation; split details and sample counts are not reported and metrics are not independently reproduced.",
            "The two named source tables have different columns; the model README does not document how their rows/features were harmonized.",
            "The pipeline can emit a class code only; no class probabilities are exposed by this API.",
            "NASA provenance and the source dataset URLs have not been independently verified.",
        ],
    },
}


class PredictionModelUnavailable(RuntimeError):
    """The selected model artifact is missing, altered, or incompatible."""


class PredictionInferenceError(RuntimeError):
    """A selected model failed to produce a finite, valid prediction."""


@lru_cache(maxsize=4)
def load_additional_model(model_key: str) -> Any:
    """Integrity-check and load a registered model once per process."""
    config = MODEL_REGISTRY.get(model_key)
    if config is None:
        raise PredictionModelUnavailable("Requested prediction model is not registered.")
    try:
        path = config["path"]
        if not path.is_file():
            raise PredictionModelUnavailable("Prediction model artifact is unavailable.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config["sha256"]:
            raise PredictionModelUnavailable("Prediction model artifact failed its integrity check.")
        model = joblib.load(path)
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None or tuple(model_features) != config["features"]:
            raise PredictionModelUnavailable("Prediction model feature metadata is missing or incompatible.")
        if getattr(model, "n_features_in_", len(config["features"])) != len(config["features"]):
            raise PredictionModelUnavailable("Prediction model feature count is incompatible.")
        expected_classes = config.get("classes")
        if expected_classes is not None and tuple(getattr(model, "classes_", ())) != expected_classes:
            raise PredictionModelUnavailable("Prediction model class metadata is missing or incompatible.")
        return model
    except PredictionModelUnavailable:
        raise
    except Exception as exc:
        raise PredictionModelUnavailable("Prediction model could not be loaded.") from exc


def _psi10_input_frame(request: Psi10InverseFlameRequest) -> pd.DataFrame:
    values = [
        request.oxygen_mole_fraction,
        request.oxygen_mass_flow_mg_s,
        request.nitrogen_mass_flow_mg_s,
        request.total_mass_flow_mg_s,
        request.initial_pressure_atm,
        request.mean_pressure_atm,
    ]
    return pd.DataFrame([values], columns=PSI10_FEATURES)


def _psi159_ignition_input_frame(request: Psi159IgnitionRequest) -> pd.DataFrame:
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
    return pd.DataFrame([values], columns=PSI159_IGNITION_FEATURES)


def _input_validation(feature_order: Tuple[str, ...], units: Dict[str, str]) -> PredictionInputValidation:
    return PredictionInputValidation(
        status="VALIDATED",
        feature_order=list(feature_order),
        units=units,
        note="Input types and basic nonnegative/fraction constraints were validated; membership in the model training distribution is not established.",
    )


def _provenance(config: Dict[str, Any]) -> PredictionProvenance:
    return PredictionProvenance(
        dataset_filename=config["source_dataset"],
        source_url=None,
        verification_status="UNVERIFIED",
        note="The local model-card/repository description was inspected, but the original dataset and NASA source have not been independently confirmed.",
    )


def _finite_prediction(model_key: str, frame: pd.DataFrame) -> float:
    try:
        prediction = load_additional_model(model_key).predict(frame)
        value = float(prediction[0])
    except PredictionModelUnavailable:
        raise
    except Exception as exc:
        raise PredictionInferenceError("Prediction model failed to process the supplied input.") from exc
    if not np.isfinite(value):
        raise PredictionInferenceError("Prediction model returned a non-finite value.")
    return value


def _psi10_response(model_key: str, request: Psi10InverseFlameRequest) -> Psi10RegressionPredictionResponse:
    config = MODEL_REGISTRY[model_key]
    result = _finite_prediction(model_key, _psi10_input_frame(request))
    return Psi10RegressionPredictionResponse(
        model_name=config["name"],
        prediction_target=config["target"],
        predicted_value=result,
        output_unit=config["unit"],
        evaluation_metrics=config["metrics"],
        input_validation=_input_validation(
            PSI10_FEATURES,
            {
                "XO2": "mole fraction (0 to 1)",
                "mO2 (mg/s)": "mg/s",
                "mN2 (mg/s)": "mg/s",
                "mtot (mg/s)": "mg/s",
                "pinit (atm)": "atm",
                "pmean (atm)": "atm",
            },
        ),
        provenance=_provenance(config),
        limitations=config["limitations"],
        scientific_warning=SCIENTIFIC_WARNING,
    )


def predict_psi10_tad(request: Psi10InverseFlameRequest) -> Psi10RegressionPredictionResponse:
    return _psi10_response("psi10_tad", request)


def predict_psi10_zst(request: Psi10InverseFlameRequest) -> Psi10RegressionPredictionResponse:
    return _psi10_response("psi10_zst", request)


def predict_psi10_pressure_rise(request: Psi10InverseFlameRequest) -> Psi10RegressionPredictionResponse:
    return _psi10_response("psi10_pressure_rise", request)


def predict_psi159_ignition(request: Psi159IgnitionRequest) -> Psi159IgnitionPredictionResponse:
    config = MODEL_REGISTRY["psi159_ignition"]
    frame = _psi159_ignition_input_frame(request)
    try:
        model = load_additional_model("psi159_ignition")
        class_code = int(model.predict(frame)[0])
    except PredictionModelUnavailable:
        raise
    except Exception as exc:
        raise PredictionInferenceError("Ignition classifier failed to process the supplied input.") from exc
    labels = {0: "No Ignition", 1: "Hot Ignition/Flash"}
    if class_code not in labels:
        raise PredictionInferenceError("Ignition classifier returned an undocumented class code.")
    return Psi159IgnitionPredictionResponse(
        model_name=config["name"],
        predicted_class=labels[class_code],
        predicted_class_code=class_code,
        evaluation_metrics=config["metrics"],
        input_validation=_input_validation(
            PSI159_IGNITION_FEATURES,
            {
                "Flame Type": "categorical label (e.g. Inverse)",
                "Fuel": "categorical label (e.g. ethane)",
                "approx p (bar)": "bar",
                "Tad (K)": "K",
                "HC burnrate (mg/s)": "mg/s",
                "O2 flow rate (mg/s)": "mg/s",
                "XO2": "mole fraction (0 to 1)",
                "Xf": "fuel fraction (0 to 1)",
            },
        ),
        provenance=_provenance(config),
        limitations=config["limitations"],
        scientific_warning=(
            SCIENTIFIC_WARNING
            + " The class label is an experimental classification and must not be interpreted as a real-world fire probability."
        ),
    )
