"""Inference for selected, documented PSI-10 and PSI-159 models.

The local joblib artifacts are UNVERIFIED research artifacts. Only artifacts
whose documented schema and installed serialization dependencies were checked
are registered here.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import tempfile
from typing import Any, Dict, Tuple
import zipfile

import joblib
import numpy as np
import pandas as pd

from app.models.additional_predictions import (
    PredictionEvaluationMetrics,
    PredictionInputValidation,
    PredictionProvenance,
    AdditionalClassificationResponse,
    Psi107FuelClassificationRequest,
    Psi10InverseFlameRequest,
    Psi10RegressionPredictionResponse,
    Psi69FlameOutcomeRequest,
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
PSI107_FEATURES: Tuple[str, ...] = (
    "Nozzle (mm)",
    "Fuel Rate (mg/s)",
    "HC Fuel Rate (mg/s)",
    "Air Rate (cm/s)",
)
PSI107_CLASSES: Tuple[str, ...] = (
    "50%Propylene", "75%Propylene", "Ethylene", "Methane", "Propane", "Propylene",
)
PSI69_FEATURES: Tuple[str, ...] = (
    "Ambient pressure; mmHg",
    "O initial ambient composition; mole fraction",
    "N initial ambient composition; mole fraction",
    "CO initial ambient composition; mole fraction",
    "He initial ambient composition; mole fraction",
    "Droplet initial diameter; mm",
    "Fuel",
)
PSI69_CLASSES: Tuple[str, ...] = ("Completion", "Disruption", "Extinction")

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
    "psi107_fuel_classifier": {
        "path": BACKEND_ROOT / "models/psi107/fuel_classifier_pipeline.joblib",
        "sha256": "ad59bc8564dfbb1f9ba474ffcc4a80cf4b8cc56e84a9456e92be8959d4dadd27",
        "features": PSI107_FEATURES,
        "classes": PSI107_CLASSES,
        "name": "PSI-107 FuelClassifierDNN",
        "target": "Fuel class",
        "metrics": PredictionEvaluationMetrics(
            accuracy=0.8191,
            sample_count_note="Model card reports accuracy 0.8191 but no training/test sample counts or split methodology.",
        ),
        "source_dataset": "PSI-107_Experimental table_Spice.csv",
        "limitations": [
            "The README provides the MLP architecture and the artifact embeds the scaler, class sequence, and state dictionary; no training code or split details are supplied.",
            "Output labels use the ordered classes embedded in the model artifact; raw dataset labels contain whitespace variants which are not emitted as separate model classes.",
            "The reported 0.8191 accuracy is from the supplied model card and has not been independently reproduced.",
            "The model/dataset NASA provenance has not been independently verified and predictions are not mission-safety findings.",
        ],
    },
    "psi69_flame_outcome": {
        "path": BACKEND_ROOT / "models/psi69/dnn_cls_model.keras",
        "sha256": "8ef25ddfaab4f5899723c60c35df5dde75d11aa97d679f40e6971f02bf1fb32c",
        "preprocessor_path": BACKEND_ROOT / "models/psi69/dnn_cls_preprocessor.joblib",
        "preprocessor_sha256": "8f0ef7e8f76702f06f4b20455b0e4df57ec2dbff0de31a4e8a181d1ba392af57",
        "features": PSI69_FEATURES,
        "classes": PSI69_CLASSES,
        "name": "PSI-69 FLEX Flame Outcome DNN Classifier",
        "target": "Test end outcome",
        "metrics": PredictionEvaluationMetrics(
            documented_sample_count=274,
            documented_test_split_ratio=0.20,
            sample_count_note="Model card documents 274 total samples, an 80/20 split, and SMOTE-balanced training data of 447 rows; no numerical test performance metric is provided.",
        ),
        "source_dataset": "PSI-69_Experimental table_FLEX.csv",
        "limitations": [
            "The model card feature names O2/N2/CO2 conflict with the dataset and serialized preprocessor, which use O/N/CO. The API follows the actual preprocessor feature names.",
            "Class order comes from the supplied inference README and has not been confirmed from training code; dataset labels are Completion, Disruption, and Extinction.",
            "The model card provides no numerical test metric. Its 80/20/SMOTE details are documentation-reported and not independently reproduced.",
            "The serialized artifact was saved with Keras 3.13.2; its config contains a nullable key unsupported by the compatible Python 3.9 Keras 3.10 runtime. The loader removes only that null serialization key in memory before reconstructing and loading the unchanged weights.",
            "NASA provenance and experimental generalization are unverified. Model outputs are not spacecraft safety predictions.",
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


@lru_cache(maxsize=1)
def load_psi107_bundle():
    """Verify then load the PSI-107 model packet and its embedded class mapping."""
    config = MODEL_REGISTRY["psi107_fuel_classifier"]
    path = config["path"]
    if not path.is_file():
        raise PredictionModelUnavailable("PSI-107 fuel-classifier artifact is unavailable.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["sha256"]:
        raise PredictionModelUnavailable("PSI-107 fuel-classifier artifact failed its integrity check.")
    try:
        import torch
        import torch.nn as nn

        packet = joblib.load(path)
        required = {"model_state_dict", "scaler", "classes", "input_features"}
        if not isinstance(packet, dict) or not required.issubset(packet):
            raise PredictionModelUnavailable("PSI-107 artifact is missing required model metadata.")
        if tuple(packet["input_features"]) != PSI107_FEATURES:
            raise PredictionModelUnavailable("PSI-107 feature metadata is incompatible.")
        classes = tuple(str(value) for value in packet["classes"])
        if classes != PSI107_CLASSES:
            raise PredictionModelUnavailable("PSI-107 embedded class mapping is incompatible.")
        scaler = packet["scaler"]
        if getattr(scaler, "n_features_in_", None) != len(PSI107_FEATURES):
            raise PredictionModelUnavailable("PSI-107 scaler feature count is incompatible.")

        class FuelClassifierDNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(4, 64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(32, len(classes)),
                )

            def forward(self, inputs):
                return self.network(inputs)

        model = FuelClassifierDNN()
        model.load_state_dict(packet["model_state_dict"], strict=True)
        model.eval()
        return model, scaler, classes, torch
    except PredictionModelUnavailable:
        raise
    except Exception as exc:
        raise PredictionModelUnavailable("PSI-107 model packet could not be loaded or validated.") from exc


@lru_cache(maxsize=1)
def load_psi69_bundle():
    """Load PSI-69 preprocessing and Keras weights with a narrow v3 config shim."""
    config = MODEL_REGISTRY["psi69_flame_outcome"]
    artifact = config["path"]
    preprocessor_path = config["preprocessor_path"]
    if not artifact.is_file() or not preprocessor_path.is_file():
        raise PredictionModelUnavailable("PSI-69 model or preprocessor artifact is unavailable.")
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != config["sha256"]:
        raise PredictionModelUnavailable("PSI-69 model artifact failed its integrity check.")
    if hashlib.sha256(preprocessor_path.read_bytes()).hexdigest() != config["preprocessor_sha256"]:
        raise PredictionModelUnavailable("PSI-69 preprocessor artifact failed its integrity check.")
    try:
        import tensorflow as tf

        preprocessor = joblib.load(preprocessor_path)
        if tuple(getattr(preprocessor, "feature_names_in_", ())) != PSI69_FEATURES:
            raise PredictionModelUnavailable("PSI-69 preprocessor feature schema is incompatible.")
        with zipfile.ZipFile(artifact, "r") as bundle:
            if "config.json" not in bundle.namelist() or "model.weights.h5" not in bundle.namelist():
                raise PredictionModelUnavailable("PSI-69 Keras archive is missing config or weights.")
            model_config = json.loads(bundle.read("config.json").decode("utf-8"))
            weights = bundle.read("model.weights.h5")

        def remove_null_quantization_config(value):
            if isinstance(value, dict):
                clean = {}
                for key, child in value.items():
                    if key == "quantization_config" and child is None:
                        continue
                    clean[key] = remove_null_quantization_config(child)
                return clean
            if isinstance(value, list):
                return [remove_null_quantization_config(item) for item in value]
            return value

        model = tf.keras.models.model_from_json(
            json.dumps(remove_null_quantization_config(model_config))
        )
        # The preprocessor accepts seven raw columns and one-hot encodes Fuel,
        # producing eight numerical model inputs.
        if model.input_shape[-1] != 8 or model.output_shape[-1] != len(PSI69_CLASSES):
            raise PredictionModelUnavailable("PSI-69 model input/output dimensions are incompatible.")
        if getattr(model.layers[-1].activation, "__name__", "") != "softmax":
            raise PredictionModelUnavailable("PSI-69 output layer is not the documented softmax classifier.")
        with tempfile.TemporaryDirectory(prefix="pyronull-psi69-") as temp_dir:
            weights_path = Path(temp_dir) / "model.weights.h5"
            weights_path.write_bytes(weights)
            model.load_weights(str(weights_path))
        return preprocessor, model, tf
    except PredictionModelUnavailable:
        raise
    except Exception as exc:
        raise PredictionModelUnavailable("PSI-69 model bundle could not be loaded or validated.") from exc


def _psi107_input_frame(request: Psi107FuelClassificationRequest) -> pd.DataFrame:
    return pd.DataFrame([[
        request.nozzle_mm,
        request.fuel_rate_mg_s,
        request.hydrocarbon_fuel_rate_mg_s,
        request.air_rate_cm_s,
    ]], columns=PSI107_FEATURES)


def predict_psi107_fuel(request: Psi107FuelClassificationRequest) -> AdditionalClassificationResponse:
    config = MODEL_REGISTRY["psi107_fuel_classifier"]
    try:
        model, scaler, classes, torch = load_psi107_bundle()
        scaled = scaler.transform(_psi107_input_frame(request).to_numpy(dtype=np.float64))
        inputs = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = model(inputs)
        if tuple(logits.shape) != (1, len(classes)) or not bool(torch.isfinite(logits).all()):
            raise PredictionInferenceError("PSI-107 returned an invalid class-score vector.")
        class_index = int(torch.argmax(logits, dim=1).item())
    except (PredictionModelUnavailable, PredictionInferenceError):
        raise
    except Exception as exc:
        raise PredictionInferenceError("PSI-107 model failed to process the supplied input.") from exc
    return AdditionalClassificationResponse(
        investigation_id="PSI-107",
        model_name=config["name"],
        prediction_target=config["target"],
        predicted_class=classes[class_index],
        predicted_class_index=class_index,
        evaluation_metrics=config["metrics"],
        input_validation=_input_validation(
            PSI107_FEATURES,
            {"Nozzle (mm)": "mm", "Fuel Rate (mg/s)": "mg/s", "HC Fuel Rate (mg/s)": "mg/s", "Air Rate (cm/s)": "cm/s; signed values occur in source data"},
        ),
        provenance=_provenance(config),
        limitations=config["limitations"],
        scientific_warning=(
            SCIENTIFIC_WARNING
            + " The predicted fuel label is a model classification, not a real-world fuel-detection probability."
        ),
    )


def _psi69_input_frame(request: Psi69FlameOutcomeRequest) -> pd.DataFrame:
    return pd.DataFrame([[
        request.ambient_pressure_mmhg,
        request.oxygen_mole_fraction,
        request.nitrogen_mole_fraction,
        request.carbon_monoxide_mole_fraction,
        request.helium_mole_fraction,
        request.droplet_initial_diameter_mm,
        request.fuel,
    ]], columns=PSI69_FEATURES)


def predict_psi69_flame_outcome(request: Psi69FlameOutcomeRequest) -> AdditionalClassificationResponse:
    config = MODEL_REGISTRY["psi69_flame_outcome"]
    try:
        preprocessor, model, tf = load_psi69_bundle()
        transformed = preprocessor.transform(_psi69_input_frame(request))
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=np.float32)
        if transformed.shape != (1, 8) or not np.isfinite(transformed).all():
            raise PredictionInferenceError("PSI-69 preprocessor returned invalid features.")
        scores = np.asarray(model(transformed, training=False))[0]
        if scores.shape != (len(PSI69_CLASSES),) or not np.isfinite(scores).all():
            raise PredictionInferenceError("PSI-69 returned an invalid class-score vector.")
        class_index = int(np.argmax(scores))
    except (PredictionModelUnavailable, PredictionInferenceError):
        raise
    except Exception as exc:
        raise PredictionInferenceError("PSI-69 model failed to process the supplied input.") from exc
    return AdditionalClassificationResponse(
        investigation_id="PSI-69",
        model_name=config["name"],
        prediction_target=config["target"],
        predicted_class=PSI69_CLASSES[class_index],
        predicted_class_index=class_index,
        evaluation_metrics=config["metrics"],
        input_validation=_input_validation(
            PSI69_FEATURES,
            {
                "Ambient pressure; mmHg": "mmHg",
                "O initial ambient composition; mole fraction": "mole fraction (0 to 1)",
                "N initial ambient composition; mole fraction": "mole fraction (0 to 1)",
                "CO initial ambient composition; mole fraction": "mole fraction (0 to 1)",
                "He initial ambient composition; mole fraction": "mole fraction (0 to 1)",
                "Droplet initial diameter; mm": "mm",
                "Fuel": "Methanol or Heptane",
            },
        ),
        provenance=_provenance(config),
        limitations=config["limitations"],
        scientific_warning=(
            SCIENTIFIC_WARNING
            + " The outcome class is not a calibrated probability or fire-risk assessment."
        ),
    )
