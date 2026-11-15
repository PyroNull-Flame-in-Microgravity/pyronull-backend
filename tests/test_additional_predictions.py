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
PSI10_PAYLOAD = {
    "oxygen_mole_fraction": 0.37,
    "oxygen_mass_flow_mg_s": 5.11,
    "nitrogen_mass_flow_mg_s": 7.59,
    "total_mass_flow_mg_s": 12.71,
    "initial_pressure_atm": 0.99,
    "mean_pressure_atm": 1.00,
}
PSI159_IGNITION_PAYLOAD = {
    "flame_type": "Inverse",
    "fuel": "ethane",
    "pressure_bar": 1.1033,
    "adiabatic_flame_temperature_k": 2120.0,
    "hydrocarbon_burn_rate_mg_s": 1.50,
    "oxygen_flow_rate_mg_s": 0.85,
    "oxygen_mole_fraction": 0.15,
    "fuel_fraction": 0.2981,
}


def _clear_model_cache():
    prediction_service.load_additional_model.cache_clear()


def _assert_response_contract(result):
    assert {
        "investigation_id", "model_name", "prediction_target", "evaluation_metrics",
        "model_status", "input_validation", "provenance", "limitations", "scientific_warning",
    }.issubset(result)
    assert result["model_status"] == "AVAILABLE"
    assert result["input_validation"]["status"] == "VALIDATED"
    assert result["provenance"]["verification_status"] == "UNVERIFIED"


def test_psi10_tad_endpoint_uses_actual_model_and_reports_units_and_metrics():
    _clear_model_cache()
    try:
        response = client.post(
            "/api/v1/predictions/psi10/adiabatic-flame-temperature",
            json=PSI10_PAYLOAD,
        )
    finally:
        _clear_model_cache()
    assert response.status_code == 200
    result = response.json()
    _assert_response_contract(result)
    assert result["investigation_id"] == "PSI-10"
    assert result["predicted_value"] == pytest.approx(2531.90725086)
    assert result["output_unit"] == "K"
    assert result["evaluation_metrics"]["mae"] == 53.50
    assert result["evaluation_metrics"]["r_squared"] == 0.6218
    assert result["provenance"]["verification_status"] == "UNVERIFIED"


def test_psi10_zst_endpoint_uses_actual_model_and_marks_dimensionless_target():
    _clear_model_cache()
    try:
        response = client.post(
            "/api/v1/predictions/psi10/stoichiometric-mixture-fraction",
            json=PSI10_PAYLOAD,
        )
    finally:
        _clear_model_cache()
    assert response.status_code == 200
    result = response.json()
    _assert_response_contract(result)
    assert result["predicted_value"] == pytest.approx(0.32813100724)
    assert result["output_unit"] == "dimensionless fraction (unit implied by target name)"
    assert result["evaluation_metrics"]["mae"] == 0.024039
    assert result["provenance"]["verification_status"] == "UNVERIFIED"


def test_psi10_pressure_rise_endpoint_uses_actual_model_and_reports_atm():
    _clear_model_cache()
    try:
        response = client.post("/api/v1/predictions/psi10/pressure-rise", json=PSI10_PAYLOAD)
    finally:
        _clear_model_cache()
    assert response.status_code == 200
    result = response.json()
    _assert_response_contract(result)
    assert result["predicted_value"] == pytest.approx(0.0118195902)
    assert result["output_unit"] == "atm"
    assert result["evaluation_metrics"]["mae"] == 0.001406
    assert result["limitations"]
    assert "not a certified spacecraft fire-safety" in result["scientific_warning"]


def test_psi159_ignition_endpoint_uses_actual_model_without_probability_claims():
    _clear_model_cache()
    try:
        response = client.post("/api/v1/predictions/psi159/ignition", json=PSI159_IGNITION_PAYLOAD)
    finally:
        _clear_model_cache()
    assert response.status_code == 200
    result = response.json()
    _assert_response_contract(result)
    assert result["investigation_id"] == "PSI-159"
    assert result["predicted_class_code"] == 1
    assert result["predicted_class"] == "Hot Ignition/Flash"
    assert "class_probabilities" not in result
    assert result["evaluation_metrics"]["accuracy"] == 0.9048
    assert "not be interpreted as a real-world fire probability" in result["scientific_warning"]


def test_psi10_feature_mapping_and_order(monkeypatch):
    captured = {}

    class MockModel:
        feature_names_in_ = prediction_service.PSI10_FEATURES
        n_features_in_ = 6

        def predict(self, frame):
            captured["columns"] = list(frame.columns)
            captured["values"] = frame.iloc[0].tolist()
            return np.array([100.0])

    monkeypatch.setattr(prediction_service, "load_additional_model", lambda key: MockModel())
    response = client.post("/api/v1/predictions/psi10/pressure-rise", json=PSI10_PAYLOAD)
    assert response.status_code == 200
    assert captured["columns"] == list(prediction_service.PSI10_FEATURES)
    assert captured["values"] == [0.37, 5.11, 7.59, 12.71, 0.99, 1.0]


def test_psi159_ignition_feature_mapping_and_order(monkeypatch):
    captured = {}

    class MockModel:
        feature_names_in_ = prediction_service.PSI159_IGNITION_FEATURES
        n_features_in_ = 8
        classes_ = np.array([0, 1])

        def predict(self, frame):
            captured["columns"] = list(frame.columns)
            captured["values"] = frame.iloc[0].tolist()
            return np.array([0])

    monkeypatch.setattr(prediction_service, "load_additional_model", lambda key: MockModel())
    response = client.post("/api/v1/predictions/psi159/ignition", json=PSI159_IGNITION_PAYLOAD)
    assert response.status_code == 200
    assert captured["columns"] == list(prediction_service.PSI159_IGNITION_FEATURES)
    assert captured["values"] == ["Inverse", "ethane", 1.1033, 2120.0, 1.5, 0.85, 0.15, 0.2981]
    assert response.json()["predicted_class"] == "No Ignition"


@pytest.mark.parametrize(
    "path,payload,missing_key",
    [
        ("/api/v1/predictions/psi10/adiabatic-flame-temperature", PSI10_PAYLOAD, "oxygen_mole_fraction"),
        ("/api/v1/predictions/psi10/stoichiometric-mixture-fraction", PSI10_PAYLOAD, "oxygen_mole_fraction"),
        ("/api/v1/predictions/psi10/pressure-rise", PSI10_PAYLOAD, "oxygen_mole_fraction"),
        ("/api/v1/predictions/psi159/ignition", PSI159_IGNITION_PAYLOAD, "fuel"),
    ],
)
def test_required_features_are_enforced(path, payload, missing_key):
    incomplete = dict(payload)
    del incomplete[missing_key]
    assert client.post(path, json=incomplete).status_code == 422


@pytest.mark.parametrize(
    "path,payload,field,value",
    [
        ("/api/v1/predictions/psi10/adiabatic-flame-temperature", PSI10_PAYLOAD, "oxygen_mass_flow_mg_s", "not numeric"),
        ("/api/v1/predictions/psi10/stoichiometric-mixture-fraction", PSI10_PAYLOAD, "oxygen_mole_fraction", float("nan")),
        ("/api/v1/predictions/psi10/pressure-rise", PSI10_PAYLOAD, "mean_pressure_atm", float("inf")),
        ("/api/v1/predictions/psi10/pressure-rise", PSI10_PAYLOAD, "nitrogen_mass_flow_mg_s", float("-inf")),
        ("/api/v1/predictions/psi10/adiabatic-flame-temperature", PSI10_PAYLOAD, "initial_pressure_atm", "bad"),
        ("/api/v1/predictions/psi10/stoichiometric-mixture-fraction", PSI10_PAYLOAD, "total_mass_flow_mg_s", float("inf")),
        ("/api/v1/predictions/psi159/ignition", PSI159_IGNITION_PAYLOAD, "pressure_bar", "bad"),
        ("/api/v1/predictions/psi159/ignition", PSI159_IGNITION_PAYLOAD, "fuel_fraction", float("nan")),
        ("/api/v1/predictions/psi159/ignition", PSI159_IGNITION_PAYLOAD, "oxygen_flow_rate_mg_s", float("inf")),
        ("/api/v1/predictions/psi159/ignition", PSI159_IGNITION_PAYLOAD, "adiabatic_flame_temperature_k", float("-inf")),
        ("/api/v1/predictions/psi159/ignition", PSI159_IGNITION_PAYLOAD, "flame_type", 45),
    ],
)
def test_invalid_types_and_non_finite_values_return_422(path, payload, field, value):
    invalid = dict(payload)
    invalid[field] = value
    response = client.post(
        path,
        content=json.dumps(invalid, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_model_cache_loads_registered_artifact_once(tmp_path, monkeypatch):
    calls = []
    model = SimpleNamespace(feature_names_in_=prediction_service.PSI10_FEATURES, n_features_in_=6)
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"mock trusted test artifact")
    config = dict(prediction_service.MODEL_REGISTRY["psi10_tad"], path=artifact)
    config["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setitem(prediction_service.MODEL_REGISTRY, "psi10_tad", config)
    monkeypatch.setattr(prediction_service.joblib, "load", lambda path: calls.append(path) or model)
    _clear_model_cache()
    try:
        assert prediction_service.load_additional_model("psi10_tad") is model
        assert prediction_service.load_additional_model("psi10_tad") is model
        assert len(calls) == 1
    finally:
        _clear_model_cache()


@pytest.mark.parametrize(
    "model_key,path,payload",
    [
        ("psi10_tad", "/api/v1/predictions/psi10/adiabatic-flame-temperature", PSI10_PAYLOAD),
        ("psi10_zst", "/api/v1/predictions/psi10/stoichiometric-mixture-fraction", PSI10_PAYLOAD),
        ("psi10_pressure_rise", "/api/v1/predictions/psi10/pressure-rise", PSI10_PAYLOAD),
        ("psi159_ignition", "/api/v1/predictions/psi159/ignition", PSI159_IGNITION_PAYLOAD),
    ],
)
def test_missing_model_artifact_returns_503(model_key, path, payload, tmp_path, monkeypatch):
    config = dict(prediction_service.MODEL_REGISTRY[model_key], path=tmp_path / "missing.joblib")
    monkeypatch.setitem(prediction_service.MODEL_REGISTRY, model_key, config)
    _clear_model_cache()
    try:
        response = client.post(path, json=payload)
    finally:
        _clear_model_cache()
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_model_integrity_mismatch_returns_503(tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"tampered bytes")
    config = dict(prediction_service.MODEL_REGISTRY["psi10_zst"], path=artifact, sha256="incorrect")
    monkeypatch.setitem(prediction_service.MODEL_REGISTRY, "psi10_zst", config)
    _clear_model_cache()
    try:
        response = client.post(
            "/api/v1/predictions/psi10/stoichiometric-mixture-fraction",
            json=PSI10_PAYLOAD,
        )
    finally:
        _clear_model_cache()
    assert response.status_code == 503
    assert "integrity check" in response.json()["detail"]


@pytest.mark.parametrize(
    "model_key,path,payload",
    [
        ("psi10_tad", "/api/v1/predictions/psi10/adiabatic-flame-temperature", PSI10_PAYLOAD),
        ("psi10_zst", "/api/v1/predictions/psi10/stoichiometric-mixture-fraction", PSI10_PAYLOAD),
        ("psi10_pressure_rise", "/api/v1/predictions/psi10/pressure-rise", PSI10_PAYLOAD),
        ("psi159_ignition", "/api/v1/predictions/psi159/ignition", PSI159_IGNITION_PAYLOAD),
    ],
)
def test_model_deserialization_error_returns_503(model_key, path, payload, tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"mock test artifact")
    config = dict(prediction_service.MODEL_REGISTRY[model_key], path=artifact)
    config["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setitem(prediction_service.MODEL_REGISTRY, model_key, config)
    monkeypatch.setattr(prediction_service.joblib, "load", lambda path: (_ for _ in ()).throw(ValueError("bad model")))
    _clear_model_cache()
    try:
        response = client.post(path, json=payload)
    finally:
        _clear_model_cache()
    assert response.status_code == 503
    assert "could not be loaded" in response.json()["detail"]


def test_all_integrated_routes_have_openapi_request_and_response_models():
    paths = app.openapi()["paths"]
    required = {
        "/api/v1/predictions/burn-time",
        "/api/v1/predictions/psi10/adiabatic-flame-temperature",
        "/api/v1/predictions/psi10/stoichiometric-mixture-fraction",
        "/api/v1/predictions/psi10/pressure-rise",
        "/api/v1/predictions/psi159/ignition",
        "/api/v1/mission-intelligence/profiles",
    }
    assert required.issubset(paths)
    for path in required - {"/api/v1/mission-intelligence/profiles"}:
        operation = paths[path]["post"]
        assert "$ref" in operation["requestBody"]["content"]["application/json"]["schema"]
        assert "$ref" in operation["responses"]["200"]["content"]["application/json"]["schema"]
