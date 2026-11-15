import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import additional_predictions as prediction_service


client = TestClient(app)
PSI107_PAYLOAD = {
    "nozzle_mm": 1.6002,
    "fuel_rate_mg_s": 1.971,
    "hydrocarbon_fuel_rate_mg_s": 1.971,
    "air_rate_cm_s": 36.39,
}
PSI69_PAYLOAD = {
    "ambient_pressure_mmhg": 757.7,
    "oxygen_mole_fraction": 0.21,
    "nitrogen_mole_fraction": 0.79,
    "carbon_monoxide_mole_fraction": 0.0,
    "helium_mole_fraction": 0.0,
    "droplet_initial_diameter_mm": 2.10,
    "fuel": "Methanol",
}


def _clear_caches():
    prediction_service.load_psi107_bundle.cache_clear()
    prediction_service.load_psi69_bundle.cache_clear()


def test_psi107_actual_model_uses_embedded_class_mapping_and_documented_sample():
    _clear_caches()
    try:
        response = client.post("/api/v1/predictions/psi107/fuel-classification", json=PSI107_PAYLOAD)
    finally:
        _clear_caches()
    assert response.status_code == 200
    result = response.json()
    assert result["investigation_id"] == "PSI-107"
    assert result["model_name"] == "PSI-107 FuelClassifierDNN"
    assert result["prediction_target"] == "Fuel class"
    assert result["predicted_class"] == "Propane"
    assert result["predicted_class_index"] == 4
    assert result["evaluation_metrics"]["accuracy"] == 0.8191
    assert result["provenance"]["verification_status"] == "UNVERIFIED"
    assert "class_probabilities" not in result
    assert result["model_status"] == "AVAILABLE"


def test_psi107_embedded_features_and_labels_match_api_mapping():
    _clear_caches()
    try:
        _, scaler, classes, _ = prediction_service.load_psi107_bundle()
    finally:
        _clear_caches()
    assert tuple(classes) == prediction_service.PSI107_CLASSES
    assert scaler.n_features_in_ == len(prediction_service.PSI107_FEATURES)
    assert prediction_service.PSI107_FEATURES == (
        "Nozzle (mm)", "Fuel Rate (mg/s)", "HC Fuel Rate (mg/s)", "Air Rate (cm/s)"
    )


def test_psi69_actual_classifier_and_serialized_preprocessor_use_dataset_schema():
    _clear_caches()
    try:
        response = client.post("/api/v1/predictions/psi69/flame-outcome", json=PSI69_PAYLOAD)
        preprocessor, model, _ = prediction_service.load_psi69_bundle()
    finally:
        _clear_caches()
    assert response.status_code == 200
    result = response.json()
    assert result["investigation_id"] == "PSI-69"
    assert result["model_name"] == "PSI-69 FLEX Flame Outcome DNN Classifier"
    assert result["prediction_target"] == "Test end outcome"
    assert result["predicted_class"] == "Extinction"
    assert result["predicted_class_index"] == 2
    assert result["evaluation_metrics"]["documented_sample_count"] == 274
    assert result["evaluation_metrics"]["accuracy"] is None
    assert result["provenance"]["verification_status"] == "UNVERIFIED"
    assert "class_probabilities" not in result
    assert list(preprocessor.feature_names_in_) == list(prediction_service.PSI69_FEATURES)
    assert model.input_shape == (None, 8)
    assert model.output_shape == (None, 3)


def test_psi69_feature_order_and_mapping(monkeypatch):
    captured = {}

    class MockPreprocessor:
        def transform(self, frame):
            captured["columns"] = list(frame.columns)
            captured["values"] = frame.iloc[0].tolist()
            return np.zeros((1, 8), dtype=np.float32)

    class MockModel:
        def __call__(self, value, training=False):
            return np.array([[0.1, 0.8, 0.1]], dtype=np.float32)

    monkeypatch.setattr(prediction_service, "load_psi69_bundle", lambda: (MockPreprocessor(), MockModel(), None))
    response = client.post("/api/v1/predictions/psi69/flame-outcome", json=PSI69_PAYLOAD)
    assert response.status_code == 200
    assert captured["columns"] == list(prediction_service.PSI69_FEATURES)
    assert captured["values"] == [757.7, 0.21, 0.79, 0.0, 0.0, 2.1, "Methanol"]
    assert response.json()["predicted_class"] == "Disruption"
    assert response.json()["predicted_class_index"] == 1


@pytest.mark.parametrize(
    "path,payload,missing",
    [
        ("/api/v1/predictions/psi107/fuel-classification", PSI107_PAYLOAD, "nozzle_mm"),
        ("/api/v1/predictions/psi69/flame-outcome", PSI69_PAYLOAD, "fuel"),
    ],
)
def test_required_features_are_enforced(path, payload, missing):
    incomplete = dict(payload)
    incomplete.pop(missing)
    assert client.post(path, json=incomplete).status_code == 422


@pytest.mark.parametrize(
    "path,payload,field,value",
    [
        ("/api/v1/predictions/psi107/fuel-classification", PSI107_PAYLOAD, "nozzle_mm", 0),
        ("/api/v1/predictions/psi107/fuel-classification", PSI107_PAYLOAD, "fuel_rate_mg_s", float("nan")),
        ("/api/v1/predictions/psi107/fuel-classification", PSI107_PAYLOAD, "hydrocarbon_fuel_rate_mg_s", "bad"),
        ("/api/v1/predictions/psi107/fuel-classification", PSI107_PAYLOAD, "air_rate_cm_s", float("inf")),
        ("/api/v1/predictions/psi69/flame-outcome", PSI69_PAYLOAD, "ambient_pressure_mmhg", -1),
        ("/api/v1/predictions/psi69/flame-outcome", PSI69_PAYLOAD, "oxygen_mole_fraction", 1.1),
        ("/api/v1/predictions/psi69/flame-outcome", PSI69_PAYLOAD, "droplet_initial_diameter_mm", float("inf")),
        ("/api/v1/predictions/psi69/flame-outcome", PSI69_PAYLOAD, "fuel", "UnknownFuel"),
    ],
)
def test_invalid_types_and_values_return_422(path, payload, field, value):
    invalid = dict(payload)
    invalid[field] = value
    response = client.post(
        path,
        content=json.dumps(invalid, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "model_name,model_key,path,payload,cache_clear",
    [
        ("psi107", "psi107_fuel_classifier", "/api/v1/predictions/psi107/fuel-classification", PSI107_PAYLOAD, "load_psi107_bundle"),
        ("psi69", "psi69_flame_outcome", "/api/v1/predictions/psi69/flame-outcome", PSI69_PAYLOAD, "load_psi69_bundle"),
    ],
)
def test_missing_or_changed_artifacts_return_503(model_name, model_key, path, payload, cache_clear, tmp_path, monkeypatch):
    del model_name
    original = prediction_service.MODEL_REGISTRY[model_key]
    path_key = "path"
    altered = dict(original, **{path_key: tmp_path / "missing-artifact"})
    monkeypatch.setitem(prediction_service.MODEL_REGISTRY, model_key, altered)
    getattr(prediction_service, cache_clear).cache_clear()
    try:
        response = client.post(path, json=payload)
    finally:
        getattr(prediction_service, cache_clear).cache_clear()
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_psi69_corrupt_preprocessor_is_controlled_503(tmp_path, monkeypatch):
    original = prediction_service.MODEL_REGISTRY["psi69_flame_outcome"]
    fake = tmp_path / "corrupt-preprocessor.joblib"
    fake.write_bytes(b"not a preprocessor")
    config = dict(original, preprocessor_path=fake, preprocessor_sha256=hashlib.sha256(fake.read_bytes()).hexdigest())
    monkeypatch.setitem(prediction_service.MODEL_REGISTRY, "psi69_flame_outcome", config)
    prediction_service.load_psi69_bundle.cache_clear()
    try:
        response = client.post("/api/v1/predictions/psi69/flame-outcome", json=PSI69_PAYLOAD)
    finally:
        prediction_service.load_psi69_bundle.cache_clear()
    assert response.status_code == 503
    assert "could not be loaded" in response.json()["detail"]


def test_model_cache_loads_both_additional_model_bundles_once():
    _clear_caches()
    try:
        first = prediction_service.load_psi107_bundle()
        assert prediction_service.load_psi107_bundle() is first
        second = prediction_service.load_psi69_bundle()
        assert prediction_service.load_psi69_bundle() is second
    finally:
        _clear_caches()


def test_all_prediction_routes_and_typed_schemas_are_in_openapi():
    paths = app.openapi()["paths"]
    required = {
        "/api/v1/predictions/burn-time",
        "/api/v1/predictions/psi10/adiabatic-flame-temperature",
        "/api/v1/predictions/psi10/stoichiometric-mixture-fraction",
        "/api/v1/predictions/psi10/pressure-rise",
        "/api/v1/predictions/psi159/ignition",
        "/api/v1/predictions/psi107/fuel-classification",
        "/api/v1/predictions/psi69/flame-outcome",
    }
    assert required.issubset(paths)
    for path in required:
        operation = paths[path]["post"]
        assert "$ref" in operation["requestBody"]["content"]["application/json"]["schema"]
        assert "$ref" in operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert client.get("/docs").status_code == 200


def test_existing_prediction_and_mission_routes_remain_registered():
    paths = app.openapi()["paths"]
    assert "/api/v1/predictions/burn-time" in paths
    assert "/api/v1/mission-intelligence/profiles" in paths
    assert "/api/v1/mission-intelligence/report" in paths
    assert "/api/v1/investigations" in paths
    assert "/api/v1/compare" in paths
    assert "/api/v1/assistant/query" in paths
    assert "/api/v1/rank" in paths
